#!/usr/bin/env python3
"""PyTorch GPU port of the Astra training loop (LiteLM-equivalent).

MIRRORS the NumPy reference (python/astra) 1:1 for:
  * architecture       pre-norm RMSNorm -> MHA(RoPE) -> SwiGLU, tied head
  * training loop      same config schema, step semantics, val/save cadence
  * optimizer          NumPy-identical AdamW (hand-derived, decoupled wd)
  * schedule           cosine-with-warmup, same formula
  * checkpoints        SAME .npz layout (w:<name>, m:<name>, v:<name>) so a
                       GPU-produced final.npz loads unchanged in the NumPy
                       inference path (inference/generate.py / decoder.py)
  * data stream        reuses astra.training.Corpus/SeqStream verbatim

Usage (identical surface to training/train.py):
    python training/gpu_train.py --config configs/astra5m_word_prose.json --steps 5400 \
        --out checkpoints/astra5m_word_prose --cache-dir /tmp/astra_cache
    python training/gpu_train.py --config configs/astra5m_word_chat.json --steps 3800 \
        --out checkpoints/astra5m_word_chat \
        --resume checkpoints/astra5m_word_prose/resumed/final.npz --reset-step

Determinism: seeding is torch-native (torch.manual_seed + torch.Generator), so
weights are NOT bit-identical to the NumPy run, but same seed => same loss
trajectory on the same device.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from astra.model.config import ModelConfig
from astra.safety import leak_check
from astra.tokenizer import load_tokenizer
from astra.training.data import Corpus, SeqStream
from astra.utils import sha256_file, write_json

# ------------------------------------------------------------------ model


def build_rope_tables(head_dim: int, max_seq: int, theta: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / theta ** (torch.arange(0, head_dim - 1, 2, dtype=torch.float32) / head_dim)
    pos = torch.arange(max_seq, dtype=torch.float32)
    freqs = torch.outer(pos, inv)
    return torch.cos(freqs).to(device), torch.sin(freqs).to(device)


class TorchLiteLM(torch.nn.Module):
    """torch.nn mirror of astra.model.core.LiteLM with the same param names.

    Params are stored in a flat ``self._params`` dict keyed by the dotted
    names used by the checkpoint format (``w:attn0.qkv`` etc). They are plain
    leaf ``nn.Parameter`` tensors (not registered module attributes — PyTorch
    forbids ``.`` in registered names), so autograd fills ``.grad`` normally.
    """

    def __init__(self, cfg: ModelConfig, seed: int = 0, device: torch.device | None = None, dtype: torch.dtype = torch.float32):
        super().__init__()
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dtype = dtype
        self._params: dict[str, torch.nn.Parameter] = {}

        wte = torch.empty(cfg.vocab_size, cfg.d_model, dtype=torch.float32)
        torch.nn.init.normal_(wte, 0.0, 0.02, generator=gen)
        self._params["wte"] = torch.nn.Parameter(wte)

        if cfg.pos_type == "learned":
            pe = torch.empty(cfg.max_seq_len, cfg.d_model, dtype=torch.float32)
            torch.nn.init.normal_(pe, 0.0, 0.02, generator=gen)
            self._params["pos_emb"] = torch.nn.Parameter(pe)

        self.cos, self.sin = build_rope_tables(cfg.d_head, cfg.max_seq_len, cfg.rope_theta, self.device)

        for li in range(cfg.n_layers):
            self._params[f"attn{li}.qkv"] = torch.nn.Parameter(self._init_linear(cfg, cfg.d_model, 3 * cfg.d_model, li, gen))
            self._params[f"attn{li}.out"] = torch.nn.Parameter(self._init_linear(cfg, cfg.d_model, cfg.d_model, li, gen))
            self._params[f"attn{li}.ln1"] = torch.nn.Parameter(torch.ones(cfg.d_model, dtype=torch.float32))
            self._params[f"ffn{li}.wg"] = torch.nn.Parameter(self._init_linear(cfg, cfg.d_model, cfg.d_ffn, li, gen))
            if cfg.ffn_type == "swiglu":
                self._params[f"ffn{li}.wu"] = torch.nn.Parameter(self._init_linear(cfg, cfg.d_model, cfg.d_ffn, li, gen))
            self._params[f"ffn{li}.wd"] = torch.nn.Parameter(self._init_linear(cfg, cfg.d_ffn, cfg.d_model, li, gen))
            self._params[f"ffn{li}.ln2"] = torch.nn.Parameter(torch.ones(cfg.d_model, dtype=torch.float32))
        self._params["ln_f"] = torch.nn.Parameter(torch.ones(cfg.d_model, dtype=torch.float32))
        for name in self._params:
            p_tensor = self._params[name].to(self.device)
            if dtype != torch.float32 and not name.endswith((".ln1", ".ln2", "ln_f")):
                p_tensor = p_tensor.to(dtype)
            self._params[name] = torch.nn.Parameter(p_tensor)

    def _init_linear(self, cfg: ModelConfig, in_f: int, out_f: int, layer_idx: int, gen) -> torch.Tensor:
        limit = math.sqrt(6.0 / (in_f + out_f))
        scale = 1.0 / math.sqrt(layer_idx + 1)
        w = torch.empty(in_f, out_f, dtype=torch.float32)
        torch.nn.init.uniform_(w, -limit, limit, generator=gen)
        return w * (scale / limit)

    def named_params(self) -> list[tuple[str, torch.Tensor]]:
        return list(self._params.items())

    def _rms(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        mean_sq = torch.mean(x * x, dim=-1, keepdim=True)
        return x * g * torch.rsqrt(mean_sq + self.cfg.eps)

    def _rotate(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        d2 = torch.arange(0, x.shape[-1], 2, device=x.device)
        even, odd = x[..., d2], x[..., d2 + 1]
        return torch.stack((-odd * sin + even * cos, even * sin + odd * cos), dim=-1).flatten(-2)

    def forward(self, ids: torch.Tensor):
        cfg = self.cfg
        B, T = ids.shape
        P = self._params
        x = P["wte"][ids]
        if cfg.pos_type == "learned":
            x = x + P["pos_emb"][:T]
        cos, sin = self.cos[:T], self.sin[:T]
        for i in range(cfg.n_layers):
            h = self._rms(x, P[f"attn{i}.ln1"])
            qkv = h @ P[f"attn{i}.qkv"]
            q, k, v = torch.split(qkv, cfg.d_model, dim=-1)
            q = q.reshape(B, T, cfg.n_heads, cfg.d_head).transpose(1, 2)
            k = k.reshape(B, T, cfg.n_heads, cfg.d_head).transpose(1, 2)
            v = v.reshape(B, T, cfg.n_heads, cfg.d_head).transpose(1, 2)
            q, k = self._rotate(q, cos, sin), self._rotate(k, cos, sin)
            scores = (q @ k.transpose(-1, -2)) * (cfg.d_head**-0.5)
            mask = torch.triu(torch.full((T, T), -1e9, dtype=torch.float32, device=x.device), diagonal=1)
            scores = scores + mask
            att = torch.softmax(scores, dim=-1)
            z = att @ v
            z = z.transpose(1, 2).reshape(B, T, cfg.d_model)
            x = x + (z @ P[f"attn{i}.out"])

            h = self._rms(x, P[f"ffn{i}.ln2"])
            if cfg.ffn_type == "swiglu":
                gate = h @ P[f"ffn{i}.wg"]
                up = h @ P[f"ffn{i}.wu"]
                ff_in = torch.nn.functional.silu(gate) * up
            else:
                ff_in = torch.nn.functional.gelu(h @ P[f"ffn{i}.wg"])
            x = x + (ff_in @ P[f"ffn{i}.wd"])
        self._final_act = x
        ln_out = self._rms(x, P["ln_f"])
        self._ln_out = ln_out
        logits = ln_out @ P["wte"].T
        return logits

    def forward_loss(self, ids: torch.Tensor, targets: torch.Tensor):
        logits = self.forward(ids)
        return logits, torch.nn.functional.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1))

    @property
    def num_params(self) -> int:
        return int(sum(p.numel() for _n, p in self.named_params()))


# ------------------------------------------------- optimizer (NumPy-identical)

class AdamW:
    def __init__(self, model: TorchLiteLM, lr: float = 3e-4, betas=(0.9, 0.95), eps: float = 1e-8, weight_decay: float = 0.1):
        self.model = model
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.m: dict[str, torch.Tensor] = {}
        self.v: dict[str, torch.Tensor] = {}
        for name, p in model.named_params():
            self.m[name] = torch.zeros_like(p)
            self.v[name] = torch.zeros_like(p)
        self.t = 0
        self.last_lr = lr

    @staticmethod
    def _decays(name: str) -> bool:
        return not (name.endswith((".ln1", ".ln2")) or name == "ln_f")

    def step(self, lr: float) -> None:
        self.t += 1
        self.last_lr = lr
        with torch.no_grad():
            for name, p in self.model.named_params():
                g = p.grad
                if g is None:
                    continue
                m, v = self.m[name], self.v[name]
                m.mul_(self.beta1).add_(g, alpha=1 - self.beta1)
                v.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
                mhat = m / (1 - self.beta1**self.t)
                vhat = v / (1 - self.beta2**self.t)
                upd = mhat / (torch.sqrt(vhat) + self.eps)
                if self._decays(name):
                    upd = upd + self.wd * p
                p.sub_(lr * upd)

    def state_dict(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name in self.m:
            out[f"m:{name}"] = self.m[name]
            out[f"v:{name}"] = self.v[name]
        return out

    def load_state_dict(self, data: dict) -> None:
        for name in self.m:
            if f"m:{name}" in data:
                self.m[name].copy_(data[f"m:{name}"])
            if f"v:{name}" in data:
                self.v[name].copy_(data[f"v:{name}"])


class CosineSchedule:
    def __init__(self, max_steps: int, warmup_steps: int, peak_lr: float, min_lr: float = 0.0):
        self.max_steps = max(1, max_steps)
        self.warmup_steps = min(warmup_steps, self.max_steps)
        self.peak_lr = peak_lr
        self.min_lr = min_lr

    def lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.peak_lr * (step + 1) / max(1, self.warmup_steps)
        t = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        t = min(1.0, t)
        return self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1 + math.cos(math.pi * t))


# ------------------------------------------------------------ checkpoints

def save_checkpoint_torch(
    path: str,
    model: TorchLiteLM,
    opt: AdamW | None,
    schedule: CosineSchedule | None,
    step: int,
    loss_hist: list[float],
    meta: dict,
) -> None:
    payload: dict[str, np.ndarray] = {}
    for name, p in model.named_params():
        payload[f"w:{name}"] = p.detach().cpu().numpy()
    if opt is not None:
        for key, arr in opt.state_dict().items():
            payload[key] = arr.cpu().numpy()
    np.savez_compressed(path, **payload)
    mpath = str(path).rsplit(".npz", 1)[0] + ".manifest.json"
    write_json(mpath, {**meta, "step": step, "opt_t": opt.t if opt is not None else 0, "loss_hist": loss_hist})


def load_checkpoint_torch(path: str, model: TorchLiteLM, opt: AdamW | None, reset_step: bool = False) -> tuple[int, list[float]]:
    data = np.load(path)
    with torch.no_grad():
        for name, p in model.named_params():
            p.copy_(torch.from_numpy(data[f"w:{name}"]))
    step, loss_hist = 0, []
    mpath = str(path).rsplit(".npz", 1)[0] + ".manifest.json"
    if Path(mpath).exists():
        m = json.loads(Path(mpath).read_text())
        step, loss_hist = m["step"], m["loss_hist"]
        if reset_step:
            step, loss_hist = 0, []
        elif opt is not None:
            opt.t = m.get("opt_t", 0)
            opt.load_state_dict({k: torch.from_numpy(data[k]) for k in data.files if k.startswith(("m:", "v:"))})
    return step, loss_hist


# --------------------------------------------------------------- training

def val_loss(model: TorchLiteLM, corpus: Corpus, cfg: ModelConfig, device) -> dict:
    model.eval()
    stream = SeqStream(corpus, batch_seq=4, seq_len=cfg.max_seq_len, rng=np.random.default_rng(0))
    total, n = 0.0, 0
    amp_dtype = torch.float16 if model.model_dtype == torch.float16 else (torch.bfloat16 if model.model_dtype == torch.bfloat16 else torch.float32)
    with torch.no_grad():
        for x, y in stream:
            xt = torch.from_numpy(np.asarray(x)).long().to(device)
            yt = torch.from_numpy(np.asarray(y)).long().to(device)
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=amp_dtype, enabled=(model.model_dtype != torch.float32)):
                _logits, loss = model.forward_loss(xt, yt)
            total += float(loss) * x.shape[0]
            n += x.shape[0]
    model.train()
    mean = total / max(1, n)
    return {"loss": mean, "ppl": float(math.exp(min(mean, 30.0)))}


def _encoded_ids(tok, text: str, tok_path: str, split: str, corpus_path: Path, cache_dir: str | None) -> list[int]:
    if cache_dir:
        key = f"{split}-{sha256_file(corpus_path)[:16]}-{sha256_file(tok_path)[:16]}"
        cached = Path(cache_dir) / key / f"{split}.npy"
        if cached.exists():
            print(f"[data ] cache hit: {cached}")
            return np.load(cached).tolist()
    ids = tok.encode(text)
    if cache_dir:
        cached = Path(cache_dir) / key / f"{split}.npy"
        cached.parent.mkdir(parents=True, exist_ok=True)
        np.save(cached, np.asarray(ids, dtype=np.int32))
        print(f"[data ] cache write: {cached}")
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/toy_pretrain.json")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--reset-step", action="store_true")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    args = ap.parse_args()

    raw = json.loads(Path(args.config).read_text())
    cfg = ModelConfig.from_dict(raw["model"])
    tr = raw["training"]
    seed = args.seed if args.seed is not None else raw.get("seed", 0)
    if args.steps:
        tr = {**tr, "max_steps": args.steps}
    out_dir = args.out or raw["out_dir"]

    tok = load_tokenizer(raw["tokenizer"])
    cfg.vocab_size = len(tok)
    cfg = ModelConfig.from_dict(cfg.to_dict())

    train_path = Path(raw["data"]["train"])
    val_path = Path(raw["data"]["val"])
    train_ids = _encoded_ids(tok, train_path.read_text(encoding="utf-8"), raw["tokenizer"], "train", train_path, args.cache_dir)
    val_ids = _encoded_ids(tok, val_path.read_text(encoding="utf-8"), raw["tokenizer"], "val", val_path, args.cache_dir)

    safety = raw.get("safety", {})
    if safety.get("leak_check", True):
        leak = leak_check(train_ids, val_ids, n=13)
        print(f"[safety] leak_check(train, val, n=13) = {leak}")
        if not leak["leak_free"]:
            raise SystemExit(f"contamination detected: {leak}")
    else:
        print(f"[safety] n-gram leak gate disabled: {safety.get('note', '')}")

    train_corpus = Corpus(ids=np.array(train_ids, dtype=np.int32), manifest={"name": "train", "kind": "training"})
    val_corpus = Corpus(ids=np.array(val_ids, dtype=np.int32), manifest={"name": "val", "kind": "validation"})

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[torch] device={device} cuda_cores={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'n/a'}")

    pt_dtype = torch.float32
    if args.dtype == "fp16":
        pt_dtype = torch.float16
    elif args.dtype == "bf16":
        pt_dtype = torch.bfloat16

    model = TorchLiteLM(cfg, seed=seed, device=device, dtype=pt_dtype)

    opt = AdamW(model, lr=tr.get("peak_lr", 3e-4), weight_decay=tr.get("weight_decay", 0.1))
    bsz = tr.get("batch_seq", 8)
    max_steps = int(tr.get("max_steps", 2000))
    accum_steps = max(1, int(tr.get("accum_steps", 1)))
    warmup = int(tr.get("warmup_steps", max(1, int(0.02 * max_steps))))
    schedule = CosineSchedule(max_steps, warmup, tr.get("peak_lr", 3e-4), tr.get("min_lr", 1e-5))
    val_every = int(tr.get("val_every", 250))
    save_every = tr.get("save_every")

    step, hist = 0, []
    lr_start = 0
    if args.resume:
        step, hist = load_checkpoint_torch(args.resume, model, opt, reset_step=args.reset_step)
        lr_start = step
        out_dir = str(Path(out_dir) / "resumed")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"[model] params={model.num_params} vocab={cfg.vocab_size}")
    print(f"[torch] train_tokens={len(train_ids)} val_tokens={len(val_ids)}")

    rng = np.random.default_rng(seed)
    start = time.time()
    val_hist: list[dict] = []
    opt_step = 0
    micro_in_accum = 0

    def _snapshot(tag: str) -> None:
        save_checkpoint_torch(
            f"{out_dir}/{tag}-{step}.npz", model, opt, schedule, step=step, loss_hist=list(hist),
            meta={"seed": seed, "model_config": cfg.to_dict(), "train_config": tr, "params": model.num_params},
        )

    amp_dtype = torch.float16 if model.model_dtype == torch.float16 else (torch.bfloat16 if model.model_dtype == torch.bfloat16 else torch.float32)
    while step < max_steps:
        stream = SeqStream(train_corpus, batch_seq=bsz, seq_len=cfg.max_seq_len, rng=rng)
        for x, y in stream:
            if step >= max_steps:
                break
            xt = torch.from_numpy(np.asarray(x)).long().to(device)
            yt = torch.from_numpy(np.asarray(y)).long().to(device)
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=amp_dtype, enabled=(model.model_dtype != torch.float32)):
                loss = model.forward_loss(xt, yt)[1]
            loss.backward()
            hist.append(float(loss))
            step += 1
            micro_in_accum += 1
            if micro_in_accum < accum_steps:
                continue
            for _n, p in model.named_params():
                if p.grad is not None:
                    p.grad.mul_(1.0 / accum_steps)
            nrm = float(torch.sqrt(sum(p.grad.float().pow(2).sum() for _n, p in model.named_params() if p.grad is not None)))
            clip = tr.get("grad_clip", 1.0)
            if nrm > clip:
                for _n, p in model.named_params():
                    if p.grad is not None:
                        p.grad.mul_(clip / (nrm + 1e-6))
            opt.step(schedule.lr(lr_start + opt_step))
            opt_step += 1
            for _n, p in model.named_params():
                p.grad = None
            micro_in_accum = 0
            if step % val_every == 0:
                v = val_loss(model, val_corpus, cfg, device)
                v["step"] = step
                val_hist.append(v)
                print(f"[step {step:5d}] train={float(loss):.4f} "
                      f"val_loss={v['loss']:.4f} ppl={v['ppl']:.2f} lr={schedule.lr(lr_start + opt_step - 1):.2e}", flush=True)
            if save_every and step % save_every == 0:
                _snapshot("checkpoint")

    elapsed = time.time() - start
    final = val_loss(model, val_corpus, cfg, device)
    ckpt = f"{out_dir}/final.npz"
    save_checkpoint_torch(
        ckpt, model, opt, schedule, step=step, loss_hist=hist,
        meta={"seed": seed, "model_config": cfg.to_dict(), "train_config": tr, "params": model.num_params},
    )
    write_json(f"{out_dir}/report.json", {"steps": step, "final_val": final, "elapsed_s": elapsed, "val_history": val_hist})
    print(f"[final val] {json.dumps(final)}")
    print(f"checkpoint -> {ckpt}")


if __name__ == "__main__":
    main()
"""Training loop (docs/TRAINING.md). NumPy reference implementation.

Contract: given a config dict and (train) seed, two runs produce the same loss
trajectory. Evaluation on the frozen validation split happens on a cadence and
at the end; training loss alone is never the improvement metric.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from astra.model.core import LiteLM
from astra.model.config import ModelConfig
from astra.training.checkpoint import load_checkpoint, save_checkpoint
from astra.training.data import Corpus, SeqStream
from astra.training.optim import AdamW, CosineSchedule, clip_grad_norm
from astra.utils import write_json


@dataclass
class TrainReport:
    steps: int = 0
    loss_hist: list[float] = field(default_factory=list)
    val_hist: list[dict] = field(default_factory=list)   # {step, loss, ppl}
    final_val: dict = field(default_factory=dict)
    elapsed_s: float = 0.0
    tokens_per_sec: float = 0.0
    manifest: dict = field(default_factory=dict)


def val_loss(model: LiteLM, corpus: Corpus, cfg: ModelConfig) -> dict:
    """Mean cross-entropy + perplexity over the whole val corpus (deterministic)."""
    stream = SeqStream(corpus, batch_seq=4, seq_len=cfg.max_seq_len, rng=np.random.default_rng(0))
    total, n = 0.0, 0
    for x, y in stream:
        _logits, loss = model.forward_loss(x, y)
        total += loss * x.shape[0]
        n += x.shape[0]
    mean = total / max(1, n)
    return {"loss": mean, "ppl": float(math.exp(min(mean, 30.0)))}


def train(
    cfg: ModelConfig,
    train_config: dict,
    train_corpus: Corpus,
    val_corpus: Corpus,
    seed: int = 0,
    out_dir: str = "checkpoints/phase0",
    resume_from: str | None = None,
    val_every: int | None = None,
) -> TrainReport:
    model = LiteLM(cfg, seed=seed)
    opt = AdamW(model, lr=train_config.get("peak_lr", 3e-4),
                weight_decay=train_config.get("weight_decay", 0.1))
    bsz = train_config.get("batch_seq", 8)
    max_steps = int(train_config.get("max_steps", 2000))
    warmup = int(train_config.get("warmup_steps", max(1, int(0.02 * max_steps))))
    schedule = CosineSchedule(
        max_steps=max_steps,
        warmup_steps=warmup,
        peak_lr=train_config.get("peak_lr", 3e-4),
        min_lr=train_config.get("min_lr", 1e-5),
    )
    val_every = val_every or int(train_config.get("val_every", 250))
    hist: list[float] = []
    val_hist: list[dict] = []
    step = 0
    start = time.time()
    toks = 0
    rng = np.random.default_rng(seed)

    if resume_from:
        step, hist, _meta = load_checkpoint(resume_from, model, opt, schedule)
        out_dir = str(Path(out_dir) / "resumed")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    window: list[float] = []
    while step < max_steps:
        stream = SeqStream(train_corpus, batch_seq=bsz, seq_len=cfg.max_seq_len, rng=rng)
        for x, y in stream:
            if step >= max_steps:
                break
            model.zero_grad()
            logits, loss = model.forward_loss(x, y)
            model.backward(logits, y)
            clip_grad_norm(model, train_config.get("grad_clip", 1.0))
            opt.step(schedule.lr(step))
            toks += int(x.shape[0] * x.shape[1])
            hist.append(float(loss))
            window.append(float(loss))
            step += 1
            if step % val_every == 0:
                v = val_loss(model, val_corpus, cfg)
                v["step"] = step
                val_hist.append(v)
                print(
                    f"[step {step:5d}] train={loss:.4f} "
                    f"val_loss={v['loss']:.4f} ppl={v['ppl']:.2f} "
                    f"lr={schedule.lr(step - 1):.2e}"
                )
    # drain a trailing partial epoch so final metrics are deterministic from config
    elapsed = time.time() - start
    final = val_loss(model, val_corpus, cfg)
    manifest = {
        "seed": seed,
        "model_config": cfg.to_dict(),
        "train_config": train_config,
        "train_data": train_corpus.manifest,
        "val_data": val_corpus.manifest,
        "max_steps": max_steps,
        "params": model.num_params,
        "threads": int(os.environ.get("OPENBLAS_NUM_THREADS", "0") or 0),
    }
    report = TrainReport(
        steps=step,
        loss_hist=hist,
        val_hist=val_hist,
        final_val=final,
        elapsed_s=elapsed,
        tokens_per_sec=toks / max(1e-6, elapsed),
        manifest=manifest,
    )
    ckpt = f"{out_dir}/final.npz"
    save_checkpoint(
        ckpt,
        model,
        opt,
        schedule,
        step=step,
        loss_hist=hist,
        meta={
            "seed": seed,
            "model_config": cfg.to_dict(),
            "train_config": train_config,
            "data_manifest": {"train": train_corpus.manifest, "val": val_corpus.manifest},
            "params": model.num_params,
        },
    )
    write_json(
        f"{out_dir}/report.json",
        {
            "steps": step,
            "final_val": final,
            "elapsed_s": elapsed,
            "tokens_per_sec": toks / max(1e-6, elapsed),
            "loss_history": hist,
            "val_history": val_hist,
            "manifest": manifest,
            "checkpoint": ckpt,
        },
    )
    return report
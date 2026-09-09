"""Candidate trainer (docs/LEARNING.md § 1.9).

A *candidate* model is trained off the active checkpoint (never in place) on a
small, validated, safe set of experience examples. The candidate is a separate
artifact; it only becomes the active model after gates pass and a promotion
decision (docs/LEARNING.md § 4.4).

Mechanics:
- Load active checkpoint weights into a fresh LiteLM (same architecture).
- Build a token stream from validated *sft* experience examples, interspersed
  with a replay corpus to prevent catastrophic forgetting (docs/LEARNING.md
  § 5) and keep data volume small.
- Low LR, warmup, cosine decay — tuned for fine-tuning a tiny model.
- Save candidate to a *separate* out dir with a manifest recording origin
  (base checkpoint sha256, experience ids consumed, git commit).

The trainer consumes SFT payloads ``{input, output}`` as text and tokenizes
them into one contiguous sequence per example with the same ByteLevelBPE used
at pretraining. Deterministic given (seed, config, replay corpus, experience
set), mirroring the reproducibility contract of the base training loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from astra.model.config import ModelConfig
from astra.model.core import LiteLM, all_params
from astra.tokenizer import ByteLevelBPE
from astra.training.checkpoint import load_checkpoint, save_checkpoint
from astra.training.data import Corpus, SeqStream
from astra.training.optim import build_optimizer, build_schedule, clip_grad_norm
from astra.utils import git_commit, sha256_file, write_json


@dataclass
class CandidateConfig:
    max_steps: int = 120
    peak_lr: float = 1e-4
    min_lr: float = 2e-5
    warmup_steps: int = 10
    batch_seq: int = 4
    accum_steps: int = 1
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    replay_ratio: float = 0.5  # fraction of batches sampled from replay corpus
    loss_only_on_output: bool = True

    def to_dict(self) -> dict:
        return {
            "max_steps": self.max_steps,
            "peak_lr": self.peak_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "batch_seq": self.batch_seq,
            "accum_steps": self.accum_steps,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "replay_ratio": self.replay_ratio,
            "loss_only_on_output": self.loss_only_on_output,
        }


@dataclass
class CandidateResult:
    checkpoint: str
    steps: int
    loss_hist: list[float]
    final_loss: float
    elapsed_s: float
    manifest: dict


def _sequence_from_payload(tok: ByteLevelBPE, payload: dict, kind: str) -> list[int]:
    """Tokenize an experience payload into a single training sequence."""
    if kind == "sft":
        text = f"{payload['input']} {payload['output']}"
    elif kind == "preference":
        # We train on the good path; the bad path is only used for ranking
        # losses later (Phase 6+). For LM-style learning we use the good pair.
        text = f"{payload['input']} {payload['good']}"
    else:  # fact
        text = payload.get("text", payload.get("output", ""))
    return tok.encode(text)


def train_candidate(
    *,
    base_checkpoint: str,
    tokenizer: ByteLevelBPE,
    model_config: ModelConfig,
    experiences: list[dict],
    replay_ids: np.ndarray | None,
    config: CandidateConfig,
    out_dir: str = "checkpoints/candidate",
    seed: int = 0,
) -> CandidateResult:
    """Train a candidate off ``base_checkpoint`` on validated experiences.

    ``experiences`` must already be validated + deduped (output of the
    ExperienceStore, see submodule). ``replay_ids`` is the token array of the
    original corpus so the candidate does not forget what it already knew.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    model = LiteLM(model_config, seed=seed)
    load_checkpoint(base_checkpoint, model, None, None)

    # Build corpus artifacts.
    sft_seq: list[int] = []
    for ex in experiences:
        sft_seq += _sequence_from_payload(tokenizer, ex["payload"], ex["kind"])
        sft_seq += [tokenizer.encode("<bos>")[0]]
    # A very small experience set (e.g. a single short correction) may be
    # shorter than one context window; tile it so at least one window exists,
    # otherwise SeqStream would produce zero windows and the candidate would
    # never see the experience (silent no-op). Deterministic tiling.
    if len(sft_seq) > 0 and len(sft_seq) < model_config.max_seq_len + 1:
        tile = int(np.ceil((model_config.max_seq_len + 1) / len(sft_seq)))
        sft_seq = sft_seq * tile
    sft_ids = np.array(sft_seq, dtype=np.int32) if sft_seq else np.zeros(0, dtype=np.int32)
    exp_corpus = Corpus(
        ids=sft_ids,
        manifest={
            "name": "experience-finetune",
            "kind": "training",
            "examples": len(experiences),
            "source": "ExperienceStore",
        },
    )
    replay_corpus = None
    if replay_ids is not None and replay_ids.size > 0:
        replay_corpus = Corpus(
            ids=np.array(replay_ids, dtype=np.int32),
            manifest={"name": "replay-corpus", "kind": "training", "source": "base-checkpoint-data"},
        )

    opt = build_optimizer(
        "adamw",
        model,
        lr=config.peak_lr,
        weight_decay=config.weight_decay,
    )
    schedule = build_schedule(
        "cosine",
        max_steps=config.max_steps,
        warmup_steps=config.warmup_steps,
        peak_lr=config.peak_lr,
        min_lr=config.min_lr,
    )

    rng = np.random.default_rng(seed)
    exp_stream = SeqStream(exp_corpus, batch_seq=config.batch_seq, seq_len=model_config.max_seq_len, rng=rng)
    rep_stream = None
    if replay_corpus is not None:
        rep_stream = SeqStream(replay_corpus, batch_seq=config.batch_seq, seq_len=model_config.max_seq_len, rng=rng)

    hist: list[float] = []
    step = 0
    opt_step = 0
    model.zero_grad()
    micro_in_accum = 0
    start = time.time()

    def _train_step(x: np.ndarray, y: np.ndarray) -> None:
        nonlocal micro_in_accum, opt_step, step
        logits, loss = model.forward_loss(x, y)
        model.backward(logits, y)
        hist.append(float(loss))
        step += 1
        micro_in_accum += 1
        if micro_in_accum >= config.accum_steps:
            for _n, _w, g in all_params(model):
                g *= 1.0 / config.accum_steps
            clip_grad_norm(model, config.grad_clip)
            opt.step(schedule.lr(opt_step))
            opt_step += 1
            model.zero_grad()
            micro_in_accum = 0

    try:
        exp_iter = iter(exp_stream)
        rep_iter = iter(rep_stream) if rep_stream else None
        while step < config.max_steps:
            # decide source: replay with replay_ratio probability when available
            use_replay = rep_iter is not None and rng.random() < config.replay_ratio
            try:
                if use_replay:
                    x, y = next(rep_iter)
                else:
                    x, y = next(exp_iter)
            except StopIteration:
                if use_replay:
                    rep_iter = iter(rep_stream)
                else:
                    exp_iter = iter(exp_stream)
                if use_replay:
                    x, y = next(rep_iter)
                else:
                    x, y = next(exp_iter)
            _train_step(x, y)
    except StopIteration:
        # degenerate: no experiences and no replay to train on
        pass
    except KeyError as e:
        raise RuntimeError(f"candidate training failed at step {step}: {e}") from e

    elapsed = time.time() - start
    final_loss = float(np.mean(hist)) if hist else float("nan")

    ckpt_path = f"{out_dir}/final.npz"
    manifest = {
        "base_checkpoint": base_checkpoint,
        "base_sha256": sha256_file(base_checkpoint) if Path(base_checkpoint).exists() else "",
        "base_config": model_config.to_dict(),
        "candidate_config": config.to_dict(),
        "experiences": [
            {"id": e["id"], "kind": e["kind"], "feedback_id": e.get("feedback_id", "")}
            for e in experiences
        ],
        "num_experiences": len(experiences),
        "replay_tokens": int(replay_ids.size) if replay_ids is not None else 0,
        "seed": seed,
        "steps": step,
        "params": model.num_params,
        "git_commit": git_commit(),
        "final_loss": final_loss,
    }
    save_checkpoint(
        ckpt_path,
        model,
        opt,
        schedule,
        step=step,
        loss_hist=hist,
        meta=manifest,
    )
    write_json(
        f"{out_dir}/report.json",
        {"steps": step, "final_loss": final_loss, "elapsed_s": elapsed, "manifest": manifest},
    )
    return CandidateResult(
        checkpoint=ckpt_path,
        steps=step,
        loss_hist=hist,
        final_loss=final_loss,
        elapsed_s=elapsed,
        manifest=manifest,
    )
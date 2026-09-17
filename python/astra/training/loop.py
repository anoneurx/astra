"""Training loop (docs/TRAINING.md). NumPy reference implementation.

Contract: given a config dict and (train) seed, two runs produce the same loss
trajectory. Evaluation on the frozen validation split happens on a cadence and
at the end; training loss alone is never the improvement metric.
"""

from __future__ import annotations

import math
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from astra.experiments import ExperimentStore
from astra.model.config import ModelConfig
from astra.model.core import LiteLM, all_params
from astra.training.checkpoint import load_checkpoint, save_checkpoint
from astra.training.data import Corpus, SeqStream
from astra.training.optim import (
    build_optimizer,
    build_schedule,
    clip_grad_norm,
)
from astra.utils import environment, git_commit, write_json


@dataclass
class TrainReport:
    steps: int = 0
    loss_hist: list[float] = field(default_factory=list)
    val_hist: list[dict] = field(default_factory=list)   # {step, loss, ppl}
    final_val: dict = field(default_factory=dict)
    elapsed_s: float = 0.0
    tokens_per_sec: float = 0.0
    manifest: dict = field(default_factory=dict)
    store_run_id: str = ""


def val_loss(model: LiteLM, corpus: Corpus, cfg: ModelConfig, n_shards: int = 1) -> dict:
    """Mean cross-entropy + perplexity over the whole val corpus (deterministic).

    When ``n_shards > 1``, the per-shard breakdown is included as ``shards``
    (docs/TRAINING.md § 2.8: mean and per-shard loss on a sliding window).
    """
    stream = SeqStream(corpus, batch_seq=4, seq_len=cfg.max_seq_len, rng=np.random.default_rng(0))
    total, n = 0.0, 0
    window_losses: list[float] = []
    for x, y in stream:
        _logits, loss = model.forward_loss(x, y)
        total += loss * x.shape[0]
        n += x.shape[0]
        window_losses.append(float(loss))
    mean = total / max(1, n)
    report: dict = {"loss": mean, "ppl": float(math.exp(min(mean, 30.0)))}
    if n_shards > 1 and window_losses:
        report["shards"] = loss_by_shard(window_losses, n_shards)
    return report


def loss_by_shard(step_losses: list[float], n_shards: int) -> list[dict]:
    """Partition a per-step/batch loss trajectory into contiguous shards.

    Returns one entry per shard: ``{"shard": i, "loss": mean, "ppl": ...}``.
    Shard boundaries are contiguous in trajectory order; shards shorter than
    one step are dropped so counts stay truthful.
    """
    n = len(step_losses)
    per_shard = max(1, n // n_shards)
    shards: list[dict] = []
    for i in range(n_shards):
        start, end = i * per_shard, min(n, (i + 1) * per_shard)
        if start >= end:
            continue
        mean = float(np.mean(step_losses[start:end]))
        shards.append({"shard": i, "loss": mean, "ppl": float(math.exp(min(mean, 30.0)))})
    return shards


def train(
    cfg: ModelConfig,
    train_config: dict,
    train_corpus: Corpus,
    val_corpus: Corpus,
    seed: int = 0,
    out_dir: str = "checkpoints/phase0",
    resume_from: str | None = None,
    reset_step: bool = False,
    val_every: int | None = None,
    experiment_store: str | None = None,
) -> TrainReport:
    store = ExperimentStore(
        root=experiment_store or train_config.get("experiment_store", "experiments/runs")
    )
    val_shards = int(train_config.get("val_shards", 1))
    model = LiteLM(cfg, seed=seed)
    opt = build_optimizer(
        train_config.get("optimizer", "adamw"),
        model,
        lr=train_config.get("peak_lr", 3e-4),
        weight_decay=train_config.get("weight_decay", 0.1),
    )
    bsz = train_config.get("batch_seq", 8)
    max_steps = int(train_config.get("max_steps", 2000))
    accum_steps = max(1, int(train_config.get("accum_steps", 1)))
    warmup = int(train_config.get("warmup_steps", max(1, int(0.02 * max_steps))))
    schedule = build_schedule(
        train_config.get("scheduler", "cosine"),
        max_steps=max_steps,
        warmup_steps=warmup,
        peak_lr=train_config.get("peak_lr", 3e-4),
        min_lr=train_config.get("min_lr", 1e-5),
    )
    val_every = val_every or int(train_config.get("val_every", 250))
    save_every = train_config.get("save_every")
    hist: list[float] = []
    val_hist: list[dict] = []
    step = 0
    start = time.time()
    toks = 0
    rng = np.random.default_rng(seed)

    lr_start = 0  # schedule index resumes from a loaded checkpoint when present
    if resume_from:
        step, hist, _meta = load_checkpoint(resume_from, model, opt, schedule, reset_step=reset_step)
        lr_start = step
        out_dir = str(Path(out_dir) / "resumed")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    window: list[float] = []
    opt_step = 0  # completed macro optimizer steps (lr index)
    micro_in_accum = 0
    model.zero_grad()

    def _snapshot(tag: str) -> None:
        save_checkpoint(
            f"{out_dir}/{tag}-{step}.npz",
            model,
            opt,
            schedule,
            step=step,
            loss_hist=list(hist),
            meta={
                "seed": seed,
                "model_config": cfg.to_dict(),
                "train_config": train_config,
                "data_manifest": {"train": train_corpus.manifest, "val": val_corpus.manifest},
                "params": model.num_params,
                "git_commit": git_commit(),
                "environment": environment(),
            },
        )

    def _on_signal(signum: int, _frame: object) -> None:
        print(f"[signal] caught {signum} at step {step}; saving checkpoint-{step}.npz",
              flush=True)
        _snapshot("checkpoint")
        os._exit(128 + signum)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    while step < max_steps:
        stream = SeqStream(train_corpus, batch_seq=bsz, seq_len=cfg.max_seq_len, rng=rng)
        for x, y in stream:
            if step >= max_steps:
                break
            logits, loss = model.forward_loss(x, y)
            model.backward(logits, y)  # accumulates into grads (uses +=)
            toks += int(x.shape[0] * x.shape[1])
            hist.append(float(loss))
            window.append(float(loss))
            step += 1
            micro_in_accum += 1
            if micro_in_accum < accum_steps:
                continue  # keep accumulating this macro batch
            # macro step: average the accumulated micro-batch gradients
            for _n, _w, g in all_params(model):
                g *= 1.0 / accum_steps
            clip_grad_norm(model, train_config.get("grad_clip", 1.0))
            opt.step(schedule.lr(lr_start + opt_step))
            opt_step += 1
            model.zero_grad()
            micro_in_accum = 0
            if step % val_every == 0:
                v = val_loss(model, val_corpus, cfg)
                v["step"] = step
                val_hist.append(v)
                print(
                    f"[step {step:5d}] train={loss:.4f} "
                    f"val_loss={v['loss']:.4f} ppl={v['ppl']:.2f} "
                    f"lr={schedule.lr(lr_start + opt_step - 1):.2e}"
                )
            if save_every and step % save_every == 0:
                _snapshot("checkpoint")  # crash-safe resume point
    # drain a trailing partial epoch so final metrics are deterministic from config
    elapsed = time.time() - start
    final = val_loss(model, val_corpus, cfg, n_shards=val_shards)
    manifest = {
        "seed": seed,
        "model_config": cfg.to_dict(),
        "train_config": train_config,
        "train_data": train_corpus.manifest,
        "val_data": val_corpus.manifest,
        "max_steps": max_steps,
        "params": model.num_params,
        "threads": int(os.environ.get("OPENBLAS_NUM_THREADS", "0") or 0),
        "git_commit": git_commit(),
        "environment": environment(),
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
            "git_commit": git_commit(),
            "environment": environment(),
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
    run_id = store.new_run_id(seed)
    store.log(
        run_id,
        {
            "seed": seed,
            "git_commit": manifest["git_commit"],
            "environment": manifest["environment"],
            "manifest": manifest,
            "train_config": train_config,
            "final_val": final,
            "elapsed_s": elapsed,
            "tokens_per_sec": toks / max(1e-6, elapsed),
            "val_history": val_hist,
            "checkpoint": ckpt,
            "report": f"{out_dir}/report.json",
        },
    )
    report.store_run_id = run_id
    return report
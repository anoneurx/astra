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
- ``preference`` experiences train with a DPO-style margin loss on the
  (input, good, bad) triplets (docs/LEARNING.md § 1.8) instead of plain CE.
- Low LR, warmup, cosine decay — tuned for fine-tuning a tiny model.
- Save candidate to a *separate* out dir with a manifest recording origin
  (base checkpoint sha256, experience ids consumed, git commit).

The trainer consumes SFT payloads ``{input, output}`` as text and tokenizes
them into one contiguous sequence per example with the same ByteLevelBPE used
at pretraining. Deterministic given (seed, config, replay corpus, experience
set), mirroring the reproducibility contract of the base training loop.

When ``memory_context`` is provided, the ``<|memory|>`` block is prepended to
every experience sequence so retrieved long-term memories condition the
candidate training, and the block hash/records are recorded in the manifest
for provenance (docs/MEMORY.md § 9).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from astra.model.config import ModelConfig
from astra.model.core import LiteLM, all_params, preference_backward
from astra.tokenizer import ByteLevelBPE
from astra.training.checkpoint import load_checkpoint, save_checkpoint
from astra.training.data import Corpus, SeqStream
from astra.training.optim import build_optimizer, build_schedule, clip_grad_norm
from astra.utils import git_commit, sha256_file, write_json

MEMORY_BLOCK_OPEN = "<|memory|>"
MEMORY_BLOCK_CLOSE = "<|/memory|>"


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
    preference_beta: float = 0.1  # DPO temperature for preference examples

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
            "preference_beta": self.preference_beta,
        }


@dataclass
class CandidateResult:
    checkpoint: str
    steps: int
    loss_hist: list[float]
    final_loss: float
    elapsed_s: float
    manifest: dict


def _memory_block_tokens(tok: ByteLevelBPE, memory_context: str | None) -> list[int]:
    """Tokenized ``<|memory|>`` block (empty when no memory context).

    Accepts raw content (wrapped here) or an already-rendered ``<|memory|>``
    block as produced by ``astra.memory.injection.build_memory_block``; the
    block is never double-wrapped.
    """
    if not memory_context or not memory_context.strip():
        return []
    text = memory_context.strip()
    if not text.startswith(MEMORY_BLOCK_OPEN):
        text = f"{MEMORY_BLOCK_OPEN}\n{text}\n{MEMORY_BLOCK_CLOSE}"
    return tok.encode(text) + [tok.encode("<bos>")[0]]


def _sequence_from_payload(tok: ByteLevelBPE, payload: dict, kind: str) -> list[int]:
    """Tokenize an experience payload into a single training sequence."""
    if kind == "sft":
        text = f"{payload['input']} {payload['output']}"
    elif kind == "preference":
        # The good path becomes the supervised sequence; the bad path is used
        # by ``_preference_pair`` in the DPO step below.
        text = f"{payload['input']} {payload['good']}"
    else:  # fact
        text = payload.get("text", payload.get("output", ""))
    return tok.encode(text)


def _preference_pair(tok: ByteLevelBPE, payload: dict, seq_len: int, prefix: list[int]) -> tuple[list[int], list[int]]:
    """Build (good, bad) token streams for one preference example.

    ``prefix`` is the shared context (input + optional memory block). Good and
    bad completions are appended to the same prefix so DPO compares both paths
    from the identical conditioning context.
    """
    good_tail = tok.encode(str(payload.get("good", "")))
    bad_tail = tok.encode(str(payload.get("bad", "")))
    good = prefix + good_tail
    bad = prefix + bad_tail
    cap = max(seq_len, 1)
    if len(good) > cap:
        good = good[:cap]
    if len(bad) > cap:
        bad = bad[:cap]
    return good, bad


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
    memory_context: str | None = None,
) -> CandidateResult:
    """Train a candidate off ``base_checkpoint`` on validated experiences.

    ``experiences`` must already be validated + deduped (output of the
    ExperienceStore, see submodule). ``replay_ids`` is the token array of the
    original corpus so the candidate does not forget what it already knew.

    ``memory_context`` (optional) is a rendered ``<|memory|>`` block (see
    ``astra.memory.injection``) prepended to every experience sequence; its
    content + hash are recorded in the candidate manifest.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    seq_len = model_config.max_seq_len
    bos = tokenizer.encode("<bos>")[0]

    memory_prefix = _memory_block_tokens(tokenizer, memory_context)

    model = LiteLM(model_config, seed=seed)
    load_checkpoint(base_checkpoint, model, None, None)

    # Build corpus artifacts: SFT/fact sequences (with optional memory block)
    # and preference pairs for the DPO step.
    sft_seq: list[int] = []
    pref_pairs: list[tuple[list[int], list[int]]] = []
    n_preference = 0
    n_lm = 0
    for ex in experiences:
        kind = ex["kind"]
        if kind == "preference":
            ctx = memory_prefix + tokenizer.encode(str(ex["payload"].get("input", "")))
            good, bad = _preference_pair(tokenizer, ex["payload"], seq_len, ctx)
            # good/bad both start from the memory prefix; ``good`` is also fed
            # through the LM stream so a pure-CE signal exists even with zero
            # replay batches.
            sft_seq += good + [bos]
            pref_pairs.append((good, bad))
            n_preference += 1
            n_lm += 1
        else:
            prefix = memory_prefix + _sequence_from_payload(tokenizer, ex["payload"], "sft" if kind == "sft" else "fact")
            sft_seq += prefix + [bos]
            n_lm += 1
    # A very small experience set (e.g. a single short correction) may be
    # shorter than one context window; tile it so at least one window exists,
    # otherwise SeqStream would produce zero windows and the candidate would
    # never see the experience (silent no-op). Deterministic tiling.
    if len(sft_seq) > 0 and len(sft_seq) < seq_len + 1:
        tile = int(np.ceil((seq_len + 1) / len(sft_seq)))
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
    exp_stream = SeqStream(exp_corpus, batch_seq=config.batch_seq, seq_len=seq_len, rng=rng)
    rep_stream = None
    if replay_corpus is not None:
        rep_stream = SeqStream(replay_corpus, batch_seq=config.batch_seq, seq_len=seq_len, rng=rng)

    hist: list[float] = []
    step = 0
    opt_step = 0
    model.zero_grad()
    micro_in_accum = 0
    start = time.time()

    def _scale_accum() -> None:
        for _n, _w, g in all_params(model):
            g *= 1.0 / config.accum_steps

    def _train_step(x: np.ndarray, y: np.ndarray) -> None:
        nonlocal micro_in_accum, opt_step, step
        logits, loss = model.forward_loss(x, y)
        model.backward(logits, y)
        hist.append(float(loss))
        step += 1
        micro_in_accum += 1
        if micro_in_accum >= config.accum_steps:
            _scale_accum()
            clip_grad_norm(model, config.grad_clip)
            opt.step(schedule.lr(opt_step))
            opt_step += 1
            model.zero_grad()
            micro_in_accum = 0

    def _train_step_preference(good_ids: list[int], bad_ids: list[int]) -> None:
        """One DPO-style step on a single (good, bad) preference pair."""
        nonlocal micro_in_accum, opt_step, step
        if len(good_ids) < 2 or len(bad_ids) < 2:
            step += 1  # degenerate pair; count the step, no gradient
            return
        good_arr = np.asarray([good_ids], dtype=np.int64)
        bad_arr = np.asarray([bad_ids], dtype=np.int64)
        loss = preference_backward(
            model,
            good_arr[:, :-1], good_arr[:, 1:],
            bad_arr[:, :-1], bad_arr[:, 1:],
            beta=config.preference_beta,
        )
        hist.append(float(loss))
        step += 1
        micro_in_accum += 1
        if micro_in_accum >= config.accum_steps:
            _scale_accum()
            clip_grad_norm(model, config.grad_clip)
            opt.step(schedule.lr(opt_step))
            opt_step += 1
            model.zero_grad()
            micro_in_accum = 0

    try:
        exp_iter = iter(exp_stream)
        rep_iter = iter(rep_stream) if rep_stream else None
        pref_iter = (p for p in pref_pairs)
        while step < config.max_steps:
            # interleave replay with experience batches; preference pairs are
            # consumed before replay batches when present (they are small).
            use_replay = rep_iter is not None and rng.random() < config.replay_ratio
            try:
                if use_replay:
                    assert rep_iter is not None
                    x, y = next(rep_iter)
                    _train_step(x, y)
                else:
                    try:
                        good, bad = next(pref_iter)
                        _train_step_preference(good, bad)
                    except StopIteration:
                        x, y = next(exp_iter)
                        _train_step(x, y)
            except StopIteration:
                if use_replay:
                    assert rep_stream is not None
                    rep_iter = iter(rep_stream)
                else:
                    exp_iter = iter(exp_stream)
                if use_replay:
                    assert rep_iter is not None
                    x, y = next(rep_iter)
                    _train_step(x, y)
                else:
                    try:
                        good, bad = next(pref_iter)
                        _train_step_preference(good, bad)
                    except StopIteration:
                        pref_iter = (p for p in pref_pairs)
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
        "n_preference": n_preference,
        "n_lm": n_lm,
        "replay_tokens": int(replay_ids.size) if replay_ids is not None else 0,
        "seed": seed,
        "steps": step,
        "params": model.num_params,
        "git_commit": git_commit(),
        "final_loss": final_loss,
        "memory_context": {
            "hash": hashlib.sha256((memory_context or "").encode()).hexdigest()[:12],
            "length": len((memory_context or "").strip()),
            "tokens": len(memory_prefix),
        },
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
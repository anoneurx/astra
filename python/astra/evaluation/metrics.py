"""Metrics used by the evaluation harness (docs/EVALUATION.md § 2)."""

from __future__ import annotations

import math

import numpy as np

from astra.model.core import LiteLM, mean_cross_entropy  # noqa: F401  (re-export convention)


def perplexity_of_loss(loss: float) -> float:
    return float(math.exp(min(loss, 30.0)))


def next_token_accuracy(logits: np.ndarray, targets: np.ndarray) -> float:
    """Fraction of positions where argmax logits == target."""
    pred = np.argmax(logits.reshape(-1, logits.shape[-1]), axis=-1)
    return float(np.mean(pred == targets.reshape(-1)))


def generate(
    model: LiteLM,
    tokenizer,
    seed_ids: list[int],
    max_new: int = 128,
    temperature: float = 1.0,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """Greedy-with-temperature autoregressive sampling (reference inference)."""
    rng = rng or np.random.default_rng(0)
    v = model.cfg.vocab_size
    ctx = np.array([seed_ids], dtype=np.int64).reshape(1, -1)
    for _ in range(max_new):
        logits, _ = model.forward_loss(ctx[:, -model.cfg.max_seq_len :])
        last = logits[:, -1, :] / temperature
        last = last - last.max(axis=-1, keepdims=True)
        p = np.exp(last)
        p = p / p.sum(axis=-1, keepdims=True)
        nxt = np.array([rng.choice(v, p=p[0])])[None]
        ctx = np.concatenate([ctx, nxt], axis=1)
    return ctx[0].tolist()[len(seed_ids):]


def repetition_fraction(tokens: list[int], n: int = 4) -> float:
    """Fraction of overlapping n-grams (repeated) — detection of degenerate text."""
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    seen: set = set()
    repeats = 0
    for g in grams:
        if g in seen:
            repeats += 1
        seen.add(g)
    return repeats / len(grams)
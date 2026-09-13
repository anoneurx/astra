"""Candidate-vs-active measurement (docs/LEARNING.md § 1.10).

Shared by the Phase-5 loop driver (``tools/learning_loop.py``) and the Phase-6
self-improvement driver (``tools/self_improve.py``). The plan is
deterministic: identical (checkpoint, tokenizer, config, corpus, seed) produce
identical CE numbers, so accept/reject decisions are reproducible.
"""

from __future__ import annotations

import numpy as np

from astra.model import LiteLM, ModelConfig
from astra.tokenizer import ByteLevelBPE
from astra.training.checkpoint import load_checkpoint


def load_model(cfg: ModelConfig, ckpt: str | None) -> LiteLM:
    m = LiteLM(cfg, seed=0)
    if ckpt is not None:
        load_checkpoint(ckpt, m, opt=None, schedule=None)
    return m


def corpus_loss(model: LiteLM, tok: ByteLevelBPE, sentences: list[str]) -> float:
    """Mean next-token CE over the sentences (predict-each-token-after-first)."""
    tot = n = 0.0
    for s in sentences:
        ids = tok.encode(s)
        if len(ids) < 2:
            continue
        x = np.array([ids[:-1]], dtype=np.int64)
        y = np.array([ids[1:]], dtype=np.int64)
        _logits, loss = model.forward_loss(x, y)
        tot += float(loss) * len(ids)
        n += len(ids)
    return tot / max(1.0, n)


def partition_metrics(
    base_ckpt: str,
    candidate_ckpt: str,
    tok: ByteLevelBPE,
    cfg: ModelConfig,
    partitions: dict[str, list[str]],
) -> dict[str, dict[str, float]]:
    """Base-vs-candidate mean CE per partition.

    ``partitions`` = {metric_name: [sentences...]}. Each metric dict is
    ``{"base": .., "candidate": ..}`` consumed by the GateEngine.
    """
    base = load_model(cfg, base_ckpt)
    cand = load_model(cfg, candidate_ckpt)
    out: dict[str, dict[str, float]] = {}
    for name, sentences in partitions.items():
        out[name] = {
            "base": corpus_loss(base, tok, sentences),
            "candidate": corpus_loss(cand, tok, sentences),
        }
    return out
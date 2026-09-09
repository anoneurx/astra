"""Flat exact retrieval over the memory store (docs/MEMORY.md § 6).

Exact cosine over L2-normalised embeddings + hybrid re-rank + token budget.
No ANN dependency for v1 (small stores); Rust ``astra-memory`` HNSW is a
deferred milestone.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from astra.memory.ranking import component_scores, hybrid_score
from astra.memory.records import MemoryRecord


@dataclass
class RetrievalHit:
    record: MemoryRecord
    cosine: float
    score: float
    components: dict[str, float]
    tokens: int = 0

    @property
    def id(self) -> str:
        return self.record.id


def _normalize(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    return vec.astype(np.float32) / n if n > 0 else np.zeros(vec.shape[-1], dtype=np.float32)


def retrieve(
    records: Sequence[MemoryRecord],
    query_emb: np.ndarray,
    k: int = 8,
    budget_tokens: int | None = None,
    tokenizer: Any = None,
    weights: dict[str, float] | None = None,
    now: datetime | None = None,
) -> list[RetrievalHit]:
    """Rank ``records`` (already filtered/active) by hybrid score, top-k.

    ``budget_tokens`` caps the total token-length of returned content so the
    assembled memory block cannot outsize the prompt. If no record fits, the
    top-ranked record is still returned so retrieval never silently returns
    empty (docs/MEMORY.md § 9).
    """
    if k < 1:
        return []
    now = now or datetime.now(UTC)
    q = _normalize(query_emb)

    candidates = [r for r in records if r.embedding is not None]
    if not candidates:
        return []

    embs = np.stack([np.asarray(r.embedding, dtype=np.float32) for r in candidates])
    embs = np.vstack([_normalize(e) for e in embs])
    cosines = np.clip((embs @ q).astype(np.float64), -1.0, 1.0)

    hits: list[RetrievalHit] = []
    for rec, cos in zip(candidates, cosines):
        comps = component_scores(rec, float(cos), now)
        score, _ = hybrid_score(rec, float(cos), weights, now, components=comps)
        hits.append(RetrievalHit(rec, cosine=float(cos), score=score, components=comps))

    hits.sort(key=lambda h: h.score, reverse=True)
    hits = hits[:k]

    if budget_tokens is not None and tokenizer is not None:
        for h in hits:
            h.tokens = len(tokenizer.encode(h.record.content))
        kept: list[RetrievalHit] = []
        used = 0
        for h in hits:
            if used + h.tokens <= budget_tokens or not kept:
                kept.append(h)
                used += h.tokens
        hits = kept

    return hits
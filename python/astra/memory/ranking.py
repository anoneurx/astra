"""Baseline hybrid ranking for retrieval (docs/MEMORY.md § 7).

``score = w_cos·cosine + w_recency·recency + w_conf·confidence + w_kind·kind``
with each component normalised to [0, 1]. Proposed default weights; a ranking
ablation is Phase-4 research.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

from astra.memory.records import MemoryRecord

DEFAULT_WEIGHTS = {"cosine": 1.0, "recency": 0.15, "confidence": 0.25, "kind": 0.10}

KIND_WEIGHTS = {"fact": 1.0, "semantic": 1.0, "episode": 0.8, "session": 0.3}


def _age_days(created_at: str, now: datetime) -> float:
    try:
        ts = datetime.fromisoformat(created_at)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return max(0.0, (now - ts).total_seconds()) / 86400.0
    except ValueError:
        return 0.0


def recency_score(created_at: str, half_life_days: float, now: datetime) -> float:
    return float(math.exp(-_age_days(created_at, now) / max(half_life_days, 1e-9)))


def component_scores(
    record: MemoryRecord,
    cosine: float,
    now: datetime | None = None,
    half_life_days: float = 30.0,
) -> dict[str, float]:
    now = now or datetime.now(UTC)
    return {
        "cosine": float(max(0.0, min(1.0, cosine))),
        "recency": recency_score(record.created_at, half_life_days, now),
        "confidence": float(max(0.0, min(1.0, record.confidence))),
        "kind": KIND_WEIGHTS.get(record.kind, 0.5),
    }


def hybrid_score(
    record: MemoryRecord,
    cosine: float,
    weights: dict[str, Any] | None = None,
    now: datetime | None = None,
    half_life_days: float = 30.0,
    components: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Return (hybrid score, per-component breakdown)."""
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    comps = components or component_scores(record, cosine, now, half_life_days)
    score = sum(float(weights.get(k, 0.0)) * comps[k] for k in ("cosine", "recency", "confidence", "kind"))
    return float(score), comps
"""Continuous-evolution operator tooling (docs/ROADMAP.md § Phase 8).

Phase 8 = operated, controlled continuous improvement: the Astra 1.x series
demonstrates monotonic non-regression on the core suite with rollback always
available. This module is the *coding* deliverable of that phase:

- ``MetricsStore`` — append-only, long-term store of per-promotion metric
  snapshots (the regression watch). One file per model name so the promotion
  series is cheap to scan; rows are immutable, corrections append, never
  rewrite.
- ``DriftDetector`` — classifies the latest promotion in the series as
  ``improving`` / ``stable`` / ``regressing`` and flags *silent drift*: a
  promotion that still passed its gates but trends worse than the series
  best outside the drift tolerance (a slow degradation gates can miss).
- ``lineage`` — the promote chain for a name (parent ``superseded_by`` links)
  used for model-lineage graphing and rollback context.

It sits on the Phase-6/7 machinery: accept/reject and rollback stay in
``astra.registry`` and ``astra.learning.audit``; this module only observes and
alerts on the promotion series. Built for ``tools/continuous.py``.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

MIN_WINDOW = 3


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class PromotionSnapshot:
    name: str
    sha256: str
    semver: str
    metrics: dict
    report_id: str = ""
    created_at: str = ""
    drift: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "sha256": self.sha256,
            "semver": self.semver,
            "metrics": self.metrics,
            "report_id": self.report_id,
            "created_at": self.created_at or _now(),
            "drift": self.drift,
        }


class MetricsStore:
    """Append-only long-term metrics store, one JSON object per line per name."""

    def __init__(self, root: str = "learning/metrics"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        safe = name.replace("/", "_").replace("\\", "_")
        return self.root / f"{safe}_metrics.jsonl"

    def record(self, snapshot: PromotionSnapshot) -> dict:
        row = snapshot.to_dict()
        with open(self._path(snapshot.name), "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def series(self, name: str) -> list[dict]:
        path = self._path(name)
        if not path.exists():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    def last(self, name: str) -> dict | None:
        rows = self.series(name)
        return rows[-1] if rows else None

    def metric_names(self, name: str) -> list[str]:
        seen: dict[str, None] = {}
        for r in self.series(name):
            for key in r.get("metrics", {}):
                seen[key] = None
        return sorted(seen)


@dataclass
class DriftReport:
    name: str
    metric: str
    window: int
    trend: str            # improving | stable | regressing | needs_more_data
    drift: bool           # silent drift flag (independent of trend)
    message: str
    points: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "metric": self.metric,
            "window": self.window,
            "trend": self.trend,
            "drift": self.drift,
            "message": self.message,
        }


class DriftDetector:
    """Classify the latest promotion versus the series trailing window.

    ``needs_more_data`` until at least ``min_window`` promotions have values;
    afterward the per-step mean delta vs ``regression_threshold`` decides the
    trend, and a cumulative move beyond ``drift_tolerance`` from the series
    best flags silent drift even when single steps looked fine.
    """

    def __init__(
        self,
        regression_threshold: float = 0.1,
        drift_tolerance: float = 0.2,
        min_window: int = MIN_WINDOW,
        lower_is_better: bool = True,
    ):
        self.regression_threshold = regression_threshold
        self.drift_tolerance = drift_tolerance
        self.min_window = min_window
        self.lower_is_better = lower_is_better

    def _direction(self) -> int:
        return -1 if self.lower_is_better else 1

    def _values(self, rows: list[dict], metric: str) -> list[dict]:
        return [r for r in rows if metric in r.get("metrics", {})]

    def check(self, rows: list[dict], metric: str, window: int = 5) -> DriftReport:
        points = self._values(rows, metric)
        name = rows[-1]["name"] if rows else ""
        if len(points) < self.min_window:
            return DriftReport(name, metric, window, "needs_more_data", False,
                               f"only {len(points)}/{self.min_window} promotions with {metric!r}")
        if window > 0:
            points = points[-window:]
        d = self._direction()
        vals = [r["metrics"][metric] for r in points]
        steps = [d * (b - a) for a, b in itertools.pairwise(vals)]
        mean_step = sum(steps) / len(steps)

        best = min(vals[:-1]) if self.lower_is_better else max(vals[:-1])
        last = vals[-1]
        worsening = (last - best) * -d
        drift = False

        if mean_step > self.regression_threshold:
            trend = "improving"
            message = f"monotonic improvement on {metric}: mean step {mean_step:+.4f}"
        elif mean_step < -self.regression_threshold:
            trend = "regressing"
            drift = True
            message = f"regressing {metric}: mean step {mean_step:+.4f} beyond ±{self.regression_threshold}"
        else:
            trend = "stable"
            if worsening > self.drift_tolerance:
                drift = True
                message = (f"silent drift on {metric}: now {worsening:.4f} worse than series best "
                           f"(> {self.drift_tolerance})"
                           )
            else:
                message = f"stable on {metric}: mean step {mean_step:+.4f}"

        return DriftReport(name, metric, window, trend, drift, message,
                           points=[{"sha256": p["sha256"][:12], "value": p["metrics"][metric]} for p in points])


def lineage(name: str, registry) -> list[dict]:
    """Promotion lineage for a name, oldest -> newest (active last).

    Uses the registry's immutable history plus the ``superseded_by`` parent
    pointer written at promote time, so rollbacks/graphing see the same chain
    the audit log records.
    """
    chain: list[dict] = []
    for rec in registry.history(name):
        parent = rec.extra.get("superseded_by", "")
        chain.append({
            "sha256": rec.sha256[:12],
            "semver": rec.semver,
            "superseded_by": parent[:12] if parent else "",
            "step": rec.step,
        })
    return chain


__all__ = [
    "DriftDetector",
    "DriftReport",
    "MetricsStore",
    "PromotionSnapshot",
    "lineage",
]
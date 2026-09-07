"""Queryable, JSON-backed experiment store (docs/ROADMAP.md Phase 2).

Each completed training run is recorded as ``<run_id>.json`` under the store
root. The store supports listing, exact-field and range queries, so that any
two runs can be compared deterministically (e.g. seed sweeps, LR sweeps).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from astra.utils import read_json, write_json


class ExperimentStore:
    """Append-only store of one-JSON-file-per-run records."""

    def __init__(self, root: str = "experiments/runs", suffix: str = "run"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.suffix = suffix
        self._lock = threading.Lock()

    def new_run_id(self, seed: int) -> str:
        """Timestamped, seed-tagged run id (unique per store within a second)."""
        ts = time.strftime("%Y%m%d-%H%M%S")
        return f"{self.suffix}-{ts}-s{seed}"

    def _path(self, run_id: str) -> Path:
        return self.root / f"{run_id}.json"

    def log(self, run_id: str, record: dict) -> str:
        """Persist one run record. Returns the run id."""
        payload = {"run_id": run_id, "recorded_at": time.time(), **record}
        with self._lock:
            write_json(str(self._path(run_id)), payload)
        return run_id

    def get(self, run_id: str) -> dict:
        return read_json(str(self._path(run_id)))

    def has(self, run_id: str) -> bool:
        return self._path(run_id).exists()

    def list_ids(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob(f"{self.suffix}-*.json"))

    def list(self) -> list[dict]:
        return [self.get(i) for i in self.list_ids()]

    def query(self, **filters: Any) -> list[dict]:
        """Return runs matching every filter.

        Dot-paths address nested values, e.g. ``final_val.loss=2.5``.
        Suffix ``__gt``/``__lt``/``__ge``/``__le`` turns the filter into a
        comparison, enabling thresholds like ``final_val.loss__lt=3.0``.
        """
        out = []
        for rec in self.list():
            if all(self._match(rec, k, v) for k, v in filters.items()):
                out.append(rec)
        return out

    @staticmethod
    def _match(record: dict, path: str, value: Any) -> bool:
        op = "=="
        for suffix, _op in (("__ge", ">="), ("__le", "<="), ("__gt", ">"), ("__lt", "<")):
            if path.endswith(suffix):
                op = _op
                path = path[: -len(suffix)]
        node: Any = record
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return False
            node = node[part]
        if op == "==":
            return node == value
        try:
            return {"<": node < value, "<=": node <= value, ">": node > value, ">=": node >= value}[op]
        except TypeError:
            return False

    def __len__(self) -> int:
        return len(self.list_ids())
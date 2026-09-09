"""Experience store (docs/LEARNING.md § 1.6-1.8).

Validated feedback becomes *training examples*. This module owns the durable,
append-only store of examples so the candidate trainer can consume exactly the
set the pipeline approved, and every example traces provenance to a source
FeedbackRecord (see feedback.py).

An example is a separate artifact from the raw feedback that produced it:
feedback is quarantined unless validated; only validated feedback yields an
example here. Examples may be retracted (``withdrawn``) without losing the
audit trail.

Training-example kinds (§ 1.8):
- ``sft``         : (input, output) -> next-token supervised pair
- ``preference``  : (input, good, bad) -> preference pair
- ``fact``        : a fact/correction -> routed to the memory store path

Storage: JSONL at ``<root>/store.jsonl`` (git-ignored) + sha256 snapshot
metadata. Deterministic dedup on ``dedup_key``.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


def dedup_key(kind: str, payload: dict) -> str:
    """Canonical dedup identity: content-fingerprint per kind.

    Two examples of the same kind with identical content collapse to one key,
    so re-submitting the same correction is a no-op (docs/LEARNING.md § 3.2).
    """
    canon = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return f"{kind}:{hashlib.sha256(canon.encode()).hexdigest()[:16]}"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def make_example(
    kind: str,
    payload: dict,
    *,
    feedback_id: str,
    source: str,
    confidence: float,
    verification_status: str,
    trust_tier: str,
    example_id: str | None = None,
    created_at: str | None = None,
) -> dict:
    """Build a training-example record from validated feedback."""
    return {
        "id": example_id or uuid.uuid4().hex[:16],
        "kind": kind,
        "payload": payload,
        "feedback_id": feedback_id,
        "source": source,
        "confidence": float(confidence),
        "verification_status": verification_status,
        "trust_tier": trust_tier,
        "status": "active",
        "dedup_key": dedup_key(kind, payload),
        "created_at": created_at or _now(),
    }


@dataclass
class ExperienceStore:
    root: str = "learning/store"

    def __post_init__(self) -> None:
        self.path = Path(self.root)
        self.path.mkdir(parents=True, exist_ok=True)
        self.file = self.path / "store.jsonl"
        if not self.file.exists():
            self.file.write_text("", encoding="utf-8")

    def _read_all(self) -> list[dict]:
        recs: list[dict] = []
        for line in self.file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            recs.append(json.loads(line))
        return recs

    def _write_records(self, recs: list[dict]) -> None:
        lines = "\n".join(json.dumps(r, ensure_ascii=False) for r in recs)
        self.file.write_text(lines + ("\n" if lines else ""), encoding="utf-8")

    # -- write -----------------------------------------------------------------
    def add(self, example: dict) -> tuple[str, bool]:
        """Append a validated example once (dedup by ``dedup_key``).

        Returns ``(example_id, added)`` where ``added`` is False when an
        active example with the same dedup key already exists — the re-submit
        is a no-op that does not duplicate training weight.
        """
        existing = [r for r in self._read_all() if r["dedup_key"] == example["dedup_key"]]
        if existing:
            return existing[-1]["id"], False
        recs = self._read_all()
        recs.append(example)
        self._write_records(recs)
        return example["id"], True

    def withdraw(self, example_id: str) -> bool:
        """Retract an example (status -> withdrawn); kept for audit."""
        recs = self._read_all()
        for r in recs:
            if r["id"] == example_id and r["status"] != "withdrawn":
                r["status"] = "withdrawn"
                self._write_records(recs)
                return True
        return False

    def quarantine(self, example_id: str, reason: str = "") -> bool:
        recs = self._read_all()
        for r in recs:
            if r["id"] == example_id and r["status"] != "quarantined":
                r["status"] = "quarantined"
                r["quarantine_reason"] = reason
                self._write_records(recs)
                return True
        return False

    # -- read ------------------------------------------------------------------
    def active(self, kinds: set[str] | None = None) -> list[dict]:
        """Active (trainable) examples, optionally filtered by kind."""
        out = []
        for r in self._read_all():
            if r["status"] != "active":
                continue
            if kinds and r["kind"] not in kinds:
                continue
            out.append(r)
        return out

    def kinds_active(self) -> dict:
        counts: dict = {}
        for r in self.active():
            counts[r["kind"]] = counts.get(r["kind"], 0) + 1
        return counts

    def get(self, example_id: str) -> dict | None:
        for r in self._read_all():
            if r["id"] == example_id:
                return r
        return None

    def stats(self) -> dict:
        counts: dict = {"total": 0}
        for r in self._read_all():
            counts["total"] += 1
            key = f"by_status::{r['status']}"
            counts[key] = counts.get(key, 0) + 1
        counts["by_kind"] = self.kinds_active()
        return counts

"""Promotion audit log (docs/LEARNING.md § 1.13, docs/ROADMAP.md § Phase 6).

Append-only JSONL recording every promotion decision — accept, reject, promote,
rollback, and (manual-policy) approvals — with enough provenance to reproduce
the decision: base/candidate sha256, gate results verbatim, policy, git commit.

Storage: ``learning/audit.jsonl`` (git-ignored like ``memory/store``). Rows are
never rewritten; ``Rewind`` is a forward-only marker in the same direction as
the audit itself, so the log is the single source of truth for what happened
and in what order.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from astra.utils import git_commit


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class AuditEntry:
    event: str              # decision | promote | rollback | approve | reject
    model_name: str
    base_sha: str = ""
    candidate_sha: str = ""
    decision: dict = field(default_factory=dict)
    notes: str = ""
    created_at: str = ""
    commit: str = ""

    def to_dict(self) -> dict:
        return {
            "event": self.event,
            "model_name": self.model_name,
            "base_sha": self.base_sha,
            "candidate_sha": self.candidate_sha,
            "decision": self.decision,
            "notes": self.notes,
            "created_at": self.created_at or _now(),
            "commit": self.commit or git_commit(),
        }


class AuditLog:
    """Append-only audit trail (one JSON object per line)."""

    def __init__(self, path: str = "learning/audit.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def append(self, entry: AuditEntry) -> dict:
        row = entry.to_dict()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def entries(self) -> list[dict]:
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    def count(self, event: str | None = None) -> int:
        recs = self.entries()
        if event is None:
            return len(recs)
        return sum(1 for r in recs if r["event"] == event)

    def last(self, event: str | None = None) -> dict | None:
        recs = self.entries()
        if event is not None:
            recs = [r for r in recs if r["event"] == event]
        return recs[-1] if recs else None
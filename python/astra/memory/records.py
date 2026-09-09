"""Memory record schema (docs/MEMORY.md § 4).

A record is JSON-serializable (with the optional ``embedding`` stored as a
base64 blob) and immutable once written: corrections create new revisions that
``deprecates`` a superseded record.
"""

from __future__ import annotations

import base64
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

MEMORY_KINDS = ("fact", "episode", "semantic", "session")
VERIFICATION_STATUSES = ("unverified", "verified", "disputed")
SOURCES = ("user_said", "verified_correction", "auto_extract", "model_generated")

# Records tagged with this string are treated as benchmark-evaluation material
# and are excluded from normal retrieval unless explicitly requested
# (docs/MEMORY.md § 10, docs/BENCHMARKS.md § 6).
QUARANTINE_TAG = "eval-quarantine"


def new_id() -> str:
    return f"mem_{uuid.uuid4().hex[:12]}"


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _b64_encode(arr: np.ndarray | None) -> str | None:
    if arr is None:
        return None
    return base64.b64encode(np.asarray(arr, dtype=np.float32).tobytes()).decode("ascii")


def _b64_decode(blob: str | list[float] | None) -> np.ndarray | None:
    if blob is None:
        return None
    if isinstance(blob, list):
        return np.asarray(blob, dtype=np.float32)
    raw = base64.b64decode(blob.encode("ascii"))
    if len(raw) % 4:
        raise ValueError("corrupt embedding blob (not a multiple of 4 bytes)")
    return np.frombuffer(raw, dtype=np.float32).copy()


@dataclass
class MemoryRecord:
    """One immutable memory record (v1 schema)."""

    content: str
    kind: str = "fact"
    id: str = field(default_factory=new_id)
    source: str = "auto_extract"
    confidence: float = 0.5
    verification_status: str = "unverified"
    created_at: str = field(default_factory=now_iso)
    expires_at: str | None = None
    revision: int = 1
    deprecates: str | None = None
    tags: list[str] = field(default_factory=list)
    attribution: list[str] = field(default_factory=list)
    embedding: np.ndarray | None = field(default=None, repr=False)

    def validate(self) -> None:
        if not self.content.strip():
            raise ValueError("memory record content must be non-empty")
        if self.kind not in MEMORY_KINDS:
            raise ValueError(f"invalid kind {self.kind!r}; expected one of {MEMORY_KINDS}")
        if self.source not in SOURCES:
            raise ValueError(f"invalid source {self.source!r}; expected one of {SOURCES}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.verification_status not in VERIFICATION_STATUSES:
            raise ValueError(f"invalid verification_status {self.verification_status!r}")
        if self.revision < 1:
            raise ValueError("revision must be >= 1")
        for stamp in (self.created_at, self.expires_at):
            if stamp is not None:
                datetime.fromisoformat(stamp)
        if self.expires_at is not None and self.expires_at < self.created_at:
            raise ValueError("expires_at must be >= created_at")

    @property
    def quarantined(self) -> bool:
        return QUARANTINE_TAG in self.tags

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "content": self.content,
            "source": self.source,
            "confidence": self.confidence,
            "verification_status": self.verification_status,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "revision": self.revision,
            "deprecates": self.deprecates,
            "tags": list(self.tags),
            "attribution": list(self.attribution),
        }
        if self.embedding is not None:
            d["embedding"] = _b64_encode(self.embedding)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryRecord:
        rec = cls(
            id=d["id"],
            kind=d["kind"],
            content=d["content"],
            source=d.get("source", "auto_extract"),
            confidence=float(d.get("confidence", 0.5)),
            verification_status=d.get("verification_status", "unverified"),
            created_at=d.get("created_at", now_iso()),
            expires_at=d.get("expires_at"),
            revision=int(d.get("revision", 1)),
            deprecates=d.get("deprecates"),
            tags=list(d.get("tags", [])),
            attribution=list(d.get("attribution", [])),
        )
        rec.embedding = _b64_decode(d.get("embedding"))
        return rec


def corrected_record(original: MemoryRecord, content: str | None = None, **overrides: Any) -> MemoryRecord:
    """New immutable revision superseding ``original`` (docs/MEMORY.md § 8.2)."""
    new = MemoryRecord(
        content=content if content is not None else original.content,
        kind=original.kind,
        source=overrides.get("source", "verified_correction"),
        confidence=float(overrides.get("confidence", original.confidence)),
        verification_status=overrides.get("verification_status", original.verification_status),
        tags=list(overrides.get("tags", original.tags)),
        attribution=list(overrides.get("attribution", original.attribution)),
        revision=original.revision + 1,
        deprecates=original.id,
    )
    new.expires_at = overrides.get("expires_at", original.expires_at)
    new.embedding = None  # re-embedded lazily by the store / embedder
    return new
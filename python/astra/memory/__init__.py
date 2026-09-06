"""Memory engine (docs/MEMORY.md).

Phase 0 provides the interface contract only — implementation arrives in
Phase 4 (Astra 0.5). Nothing here is executable storage yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MemoryRecord:
    id: str
    kind: str          # fact | episode | semantic | session
    content: str
    source: str        # user_said | verified_correction | auto_extract | model_generated
    confidence: float
    verification_status: str = "unverified"   # unverified | verified | disputed
    revision: int = 1
    deprecates: str | None = None
    tags: list[str] = field(default_factory=list)
    expires_at: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "content": self.content,
            "source": self.source,
            "confidence": self.confidence,
            "verification_status": self.verification_status,
            "revision": self.revision,
            "deprecates": self.deprecates,
            "tags": self.tags,
            "expires_at": self.expires_at,
        }


# Correctness contract for the Phase 4 implementation (tests must enforce):
#  - correction creates a new revision; revisions are immutable
#  - soft-delete always (audit); purge requires explicit review flag
#  - conflicts (same semantic content, competing facts) -> 'disputed', resolution queue
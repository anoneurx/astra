"""Short-term memory: session-scoped working buffer (docs/MEMORY.md § 2, § 8.5).

Long-term facts live in a ``MemoryStore`` as durable kinds
(``fact``/``episode``/``semantic``) and are recalled by ``store.search()``.
Short-term memory is a *working buffer during an interaction*: ``session``
records scoped to one session id, auto-expiring with the session's TTL and
removed in bulk on ``close()``. ``store.search()`` excludes ``session`` kinds
by default, so ephemeral conversation never pollutes durable recall.

Promotion turns a session record into a durable record via an audited copy:
``promote(session, rid, kind="fact")`` writes a new long-term record whose
attribution links back to the source session record, leaving the session
record itself untouched.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any

from astra.memory.records import MemoryRecord, now_iso
from astra.memory.retrieval import RetrievalHit, retrieve

SESSION_TAG_PREFIX = "session:"
DEFAULT_SESSION_TTL = timedelta(hours=24)


class SessionMemory:
    """Session-scoped short-term memory over a shared ``MemoryStore``.

    Records are ``kind="session"`` and carry a ``session:<id>`` tag; recall
    is scoped to this session so one session's working buffer never leaks
    into another's (or into long-term recall).
    """

    def __init__(self, store, session_id: str, ttl: timedelta = DEFAULT_SESSION_TTL):
        if not session_id or not session_id.strip():
            raise ValueError("session id must be non-empty")
        self.store = store
        self.session_id = session_id
        self.ttl = ttl
        self.tag = f"{SESSION_TAG_PREFIX}{session_id}"

    # ------------------------------------------------------------------ writes

    def add_turn(self, user: str | None = None, assistant: str | None = None,
                 **kwargs: Any) -> list[MemoryRecord]:
        """Record one (or two) conversational turns as session records.

        Returns the created records (user turn first, then assistant reply).
        """
        recs: list[MemoryRecord] = []
        if user is not None and user.strip():
            recs.append(self._record(user, source=kwargs.pop("user_source", "user_said")))
        if assistant is not None and assistant.strip():
            recs.append(self._record(assistant, source=kwargs.pop("assistant_source", "model_generated")))
        if not recs:
            raise ValueError("provide at least a non-empty user or assistant turn")
        return [self.store.add(r) for r in recs]

    def _record(self, content: str, source: str) -> MemoryRecord:
        return MemoryRecord(
            content=content,
            kind="session",
            source=source,
            tags=[self.tag],
            confidence=1.0,
            verification_status="verified",
            expires_at=(datetime.now(UTC) + self.ttl).isoformat(),
        )

    def add(self, content: str, source: str = "user_said") -> MemoryRecord:
        """Append a bare session record (returns it)."""
        return self.store.add(self._record(content, source))

    # ------------------------------------------------------------------ reads

    def recall(
        self,
        query: str,
        k: int = 5,
        budget_tokens: int | None = None,
        tokenizer: Any = None,
        weights: dict[str, float] | None = None,
        now: datetime | None = None,
    ) -> list[RetrievalHit]:
        """Retrieve this session's short-term records nearest to ``query``."""
        self.purge_expired(now=now)
        now = now or datetime.now(UTC)
        candidates = [
            r for r in self.store.records(kinds=["session"])
            if self.tag in r.tags and r.verification_status != "disputed"
        ]
        query_emb = self.store.embedder.embed([query])[0]
        hits = retrieve(candidates, query_emb, k=k, budget_tokens=budget_tokens,
                        tokenizer=tokenizer, weights=weights, now=now)
        with open(self.store.audit_path, "a") as f:
            f.write(json.dumps({"ts": now_iso(), "action": "session_query",
                                "store": self.store.name, "session": self.session_id,
                                "query_hash": hashlib.sha256(query.encode()).hexdigest()[:12],
                                "hits": len(hits)}, sort_keys=True) + "\n")
        return hits

    # --------------------------------------------------------------- lifecycle

    def records(self) -> list[MemoryRecord]:
        """Active session records (excluding deleted/expired)."""
        self.purge_expired()
        return [
            r for r in self.store.records(kinds=["session"])
            if self.tag in r.tags
        ]

    def purge_expired(self, now: datetime | None = None) -> int:
        """Soft-delete this session's records past their TTL (audited)."""
        now = now or datetime.now(UTC)
        purged = 0
        for r in self.records_unchecked():
            if r.expires_at is None:
                continue
            try:
                when = datetime.fromisoformat(r.expires_at)
                if when.tzinfo is None:
                    when = when.replace(tzinfo=UTC)
            except ValueError:
                continue
            if when <= now:
                self.store.soft_delete(r.id)
                purged += 1
        return purged

    def records_unchecked(self) -> list[MemoryRecord]:
        return [r for r in self.store.records(kinds=["session"]) if self.tag in r.tags]

    def close(self) -> int:
        """End the session: soft-delete all of its records (audited)."""
        n = 0
        for r in self.records():
            self.store.soft_delete(r.id)
            n += 1
        return n


def promote(
    session: SessionMemory,
    rid: str,
    kind: str = "fact",
    confidence: float | None = None,
    tags: list[str] | None = None,
) -> MemoryRecord:
    """Make a session record durable: an audited copy into long-term kinds.

    The new record keeps the source content, becomes a durable ``kind``,
    and its attribution cites the source session record id. The source
    session record is left untouched (it still expires with the session).
    """
    src = session.store.get(rid)
    if src is None:
        raise KeyError(f"no such active record {rid}")
    if session.tag not in src.tags:
        raise ValueError(f"record {rid} does not belong to session {session.session_id}")
    attribution = list(src.attribution) + [f"promoted:from_session:{src.id}"]
    new = MemoryRecord(
        content=src.content,
        kind=kind,
        source="model_generated" if src.source == "model_generated" else "verified_correction",
        confidence=confidence if confidence is not None else src.confidence,
        verification_status="verified" if confidence is None or confidence >= 0.7 else "unverified",
        tags=list(tags or []) + ["promoted:session"],
        attribution=attribution,
    )
    return session.store.add(new)
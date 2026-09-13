"""Versioned, audited, transactional memory store (docs/MEMORY.md §§ 4–8, 10).

Persistence is a single JSON file (records + states + revision heads + meta)
written atomically (temp file + os.replace). Every mutation also appends an
immutable line to a JSONL audit log. Record state is ``active``,
``deprecated``, ``deleted``, or (post-purge) absent.

Storage path default: ``memory/store/<name>.json`` (git-ignored).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from astra.memory.embedder import Embedder, HashEmbedder
from astra.memory.records import (
    MEMORY_KINDS,
    MemoryRecord,
    corrected_record,
    now_iso,
)
from astra.memory.retrieval import RetrievalHit, retrieve
from astra.utils import git_commit, read_json

SCHEMA_VERSION = 1
DEFAULT_DIR = "memory/store"          # git-ignored
DEFAULT_NAME = "longterm"
ACTIVE, DEPRECATED, DELETED = "active", "deprecated", "deleted"


class MemoryStore:
    """Append-only memory records with lifecycle + flat retrieval."""

    def __init__(
        self,
        name: str = DEFAULT_NAME,
        directory: str | os.PathLike[str] = DEFAULT_DIR,
        embedder: Embedder | None = None,
        meta: dict[str, Any] | None = None,
    ):
        self.name = name
        self.directory = Path(directory)
        self.embedder = embedder or HashEmbedder()
        self.path = self.directory / f"{name}.json"
        self.audit_path = self.directory / f"{name}.audit.jsonl"
        self._entries: dict[str, dict[str, Any]] = {}
        self._heads: dict[str, str] = {}
        self.meta: dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "name": name,
            "commit": git_commit(),
            "created_at": now_iso(),
            "embedding_config": self.embedder.config,
            **(meta or {}),
        }
        self._dirty = False

    # ------------------------------------------------------------------ IO

    @classmethod
    def open(
        cls,
        name: str = DEFAULT_NAME,
        directory: str | os.PathLike[str] = DEFAULT_DIR,
        embedder: Embedder | None = None,
    ) -> MemoryStore:
        store = cls(name, directory, embedder=embedder)
        if store.path.exists():
            store._load()
        return store

    def _load(self) -> None:
        data = read_json(str(self.path))
        if data.get("schema") != SCHEMA_VERSION:
            raise ValueError(f"store {self.name} schema {data.get('schema')} != {SCHEMA_VERSION}")
        self.meta = dict(data["meta"])
        self.meta["commit"] = data["meta"].get("commit")
        if data["meta"].get("embedding_config") != self.embedder.config:
            raise ValueError(
                f"store {self.name} embedding mismatch: stored "
                f"{data['meta'].get('embedding_config')} != current {self.embedder.config}"
            )
        self._entries = {
            rid: {"record": MemoryRecord.from_dict(e["record"]), "state": e["state"]}
            for rid, e in data["entries"].items()
        }
        self._heads = dict(data.get("heads", {}))

    def _save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.meta["embedding_config"] = self.embedder.config
        blob = {
            "schema": SCHEMA_VERSION,
            "meta": self.meta,
            "heads": self._heads,
            "entries": {
                rid: {"record": e["record"].to_dict(), "state": e["state"]}
                for rid, e in self._entries.items()
            },
        }
        fd, tmp = tempfile.mkstemp(dir=str(self.directory), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(blob, fh, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def _commit(self, action: str, detail: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        line = {"ts": now_iso(), "action": action, "store": self.name, **detail}
        with open(self.audit_path, "a") as f:
            f.write(json.dumps(line, sort_keys=True) + "\n")
        self._save()

    @property
    def version_key(self) -> str:
        """Store version identity (embedding + schema) for cross-process audit."""
        return f"{self.meta['schema']}:{json.dumps(self.meta['embedding_config'], sort_keys=True)}"

    def flush(self) -> None:
        if self._dirty:
            self._save()
            self._dirty = False

    # ----------------------------------------------------------- internals

    def _get_entry(self, rid: str) -> dict[str, Any] | None:
        return self._entries.get(rid)

    def _active_candidates(self) -> list[MemoryRecord]:
        return [
            e["record"] for e in self._entries.values() if e["state"] == ACTIVE
        ]

    def _ensure_embeddings(self) -> int:
        """Lazily embed any active records missing an embedding (persists)."""
        n = 0
        for e in self._entries.values():
            if e["state"] == ACTIVE and e["record"].embedding is None:
                embs = self.embedder.embed([e["record"].content])
                e["record"].embedding = embs[0]
                n += 1
        if n:
            self._dirty = True
            self._commit("reindex", {"count": n})
        return n

    # -------------------------------------------------------------- writes

    def add(self, record: MemoryRecord | None = None, content: str = "", **kwargs: Any) -> MemoryRecord:
        """Insert a record; embeds it via the store embedder."""
        rec = record or MemoryRecord(content=content, **kwargs)
        rec.validate()
        if rec.id in self._entries:
            raise ValueError(f"record {rec.id} already exists (records are immutable)")
        if rec.embedding is None:
            embs = self.embedder.embed([rec.content])
            rec.embedding = embs[0]
        self._entries[rec.id] = {"record": rec, "state": ACTIVE}
        self._heads[rec.deprecates or rec.id] = rec.id
        self._dirty = True
        self._commit("add", {"id": rec.id, "kind": rec.kind})
        return rec

    def correct(self, rid: str, content: str | None = None, **overrides: Any) -> MemoryRecord:
        """New immutable revision superseding ``rid`` (docs/MEMORY.md § 8.2)."""
        entry = self._get_entry(rid)
        if entry is None:
            raise KeyError(f"no such record {rid}")
        if entry["state"] == DEPRECATED:
            rid = self._follow_head(rid)
            entry = self._get_entry(rid)
            if entry is None:
                raise KeyError(f"no such record {rid}")
        original: MemoryRecord = entry["record"]
        new = corrected_record(original, content, **overrides)
        new.validate()
        if new.embedding is None:
            embs = self.embedder.embed([new.content])
            new.embedding = embs[0]
        entry["state"] = DEPRECATED
        self._entries[new.id] = {"record": new, "state": ACTIVE}
        self._heads[original.id] = new.id
        self._dirty = True
        self._commit("correct", {"id": original.id, "new_revision": new.id, "revision": new.revision})
        return new

    def _follow_head(self, rid: str) -> str:
        seen = set()
        while rid in self._heads and rid not in seen:
            seen.add(rid)
            rid = self._heads[rid]
        return rid

    def get_latest(self, rid: str) -> MemoryRecord | None:
        """Return the newest active revision for ``rid`` (follows heads)."""
        head = self._follow_head(rid)
        entry = self._get_entry(head)
        if entry is None or entry["state"] != ACTIVE:
            return None
        return entry["record"]

    def get(self, rid: str, include_all: bool = False) -> MemoryRecord | None:
        """Return a record (active by default; include_all also fetches deleted/deprecated)."""
        entry = self._get_entry(rid)
        if entry is None:
            return None
        if entry["state"] != ACTIVE and not include_all:
            return None
        return entry["record"]

    def soft_delete(self, rid: str, purge: bool = False) -> None:
        """Soft-delete always (audit); hard purge only with ``purge=True``."""
        entry = self._get_entry(rid)
        if entry is None:
            raise KeyError(f"no such record {rid}")
        if entry["state"] == DELETED and not purge:
            return
        if purge:
            del self._entries[rid]
            self._dirty = True
            self._commit("purge", {"id": rid, "purge": True})
            return
        self._heads.pop(rid, None)
        entry["state"] = DELETED
        self._dirty = True
        self._commit("delete", {"id": rid})

    def mark_disputed(self, rids: Iterable[str], reason: str = "") -> None:
        """Flag records as conflicting; all surface in the resolution queue (docs/MEMORY.md § 8.4)."""
        rids = list(rids)
        for rid in rids:
            entry = self._get_entry(rid)
            if entry is None or entry["state"] != ACTIVE:
                raise ValueError(f"cannot dispute inactive/missing record {rid}")
            old = entry["record"].verification_status
            entry["record"].verification_status = "disputed"
            self._dirty = True
            self._commit("dispute", {"id": rid, "from": old, "reason": reason})
        if rids:
            self._commit("dispute_group", {"group": sorted(rids), "reason": reason})

    def resolve(self, rid: str, by: str = "human", verification: str = "verified") -> MemoryRecord:
        """Resolve a conflict: chosen record is verified (audit-trailed, no silent auto-win)."""
        entry = self._get_entry(rid)
        if entry is None or entry["state"] != ACTIVE:
            raise ValueError(f"cannot resolve inactive/missing record {rid}")
        old = entry["record"].verification_status
        entry["record"].verification_status = verification
        self._dirty = True
        self._commit("resolve", {"id": rid, "from": old, "to": verification, "by": by})
        return entry["record"]

    def conflicts(self) -> list[MemoryRecord]:
        return [
            e["record"] for e in self._entries.values()
            if e["state"] == ACTIVE and e["record"].verification_status == "disputed"
        ]

    def expire(self, now: datetime | None = None) -> int:
        """Soft-delete records whose ``expires_at`` has passed (TTL audit, § 8.1)."""
        now = now or datetime.now(UTC)
        expired = []
        for e in self._entries.values():
            rec, st = e["record"], e["state"]
            if st == ACTIVE and rec.expires_at is not None:
                try:
                    when = datetime.fromisoformat(rec.expires_at)
                    if when.tzinfo is None:
                        when = when.replace(tzinfo=UTC)
                except ValueError:
                    continue
                if when <= now:
                    expired.append(rec.id)
        for rid in expired:
            entry = self._get_entry(rid)
            if entry is not None:
                entry["state"] = DELETED
            self._heads.pop(rid, None)
            self._dirty = True
            self._commit("expire", {"id": rid})
        return len(expired)

    def records(
        self,
        kinds: Iterable[str] | None = None,
        include_deleted: bool = False,
    ) -> list[MemoryRecord]:
        kinds = set(kinds or MEMORY_KINDS)
        out = []
        for e in self._entries.values():
            rec, st = e["record"], e["state"]
            if rec.kind not in kinds:
                continue
            if st != ACTIVE and not include_deleted:
                continue
            out.append(rec)
        return out

    # --------------------------------------------------------------- reads

    def compact(
        self,
        prune_deleted: bool = True,
        dedupe: bool = True,
        kinds: Iterable[str] | None = None,
    ) -> dict[str, int]:
        """Durable-memory growth maintenance (docs/ROADMAP.md § Phase 8).

        Shrinks the active memory surface without rewriting history:
        - ``prune_deleted`` physically removes soft-deleted entries (already
          inaccessible; audit rows preserve the reason).
        - ``dedupe`` marks older ACTIVE records with identical content + kind
          as DEPRECATED, keeping the newest of each clone group (the revision
          chain is untouched; DEPRECATED rows remain in the file for audit).
        Every removal is audit-trailed (``compact`` rows) and the store is
        saved once at the end. Returns counts: pruned_deleted / deduped /
        removed / active (after).
        """
        kinds = set(kinds) if kinds is not None else set(MEMORY_KINDS)
        stats: dict[str, int] = {"pruned_deleted": 0, "deduped": 0, "removed": 0}

        if prune_deleted:
            for rid in [r for r, e in self._entries.items() if e["state"] == DELETED]:
                del self._entries[rid]
                self._heads.pop(rid, None)
                stats["pruned_deleted"] += 1
                stats["removed"] += 1
                self._dirty = True
                self._commit("compact", {"id": rid, "subaction": "prune_deleted",
                                         "reason": "soft-deleted at compaction"})

        if dedupe:
            groups: dict[tuple[str, str | None], list[str]] = {}
            for e in self._entries.values():
                rec = e["record"]
                if e["state"] != ACTIVE or rec.kind not in kinds:
                    continue
                key = (rec.kind, stable_hash(rec.content))
                groups.setdefault(key, []).append(rec.id)
            for rids in groups.values():
                if len(rids) < 2:
                    continue
                rids_sorted = sorted(
                    rids,
                    key=lambda r: (self._entries[r]["record"].created_at,
                                   self._entries[r]["record"].revision),
                )
                keep = rids_sorted[-1]
                for rid in rids_sorted[:-1]:
                    self._entries[rid]["state"] = DEPRECATED
                    stats["deduped"] += 1
                    stats["removed"] += 1
                    self._dirty = True
                    self._commit("compact", {"id": rid, "subaction": "dedupe",
                                             "keep": keep, "kind": self._entries[keep]["record"].kind})

        stats["active"] = sum(1 for e in self._entries.values() if e["state"] == ACTIVE)
        if self._dirty:
            self.meta["last_compact"] = now_iso()
            self._save()
        return stats

    def search(
        self,
        query: str | None = None,
        query_emb: Any = None,
        k: int = 8,
        budget_tokens: int | None = None,
        tokenizer: Any = None,
        weights: dict[str, float] | None = None,
        include_disputed: bool = False,
        allow_quarantine: bool = False,
        kinds: Iterable[str] | None = None,
        now: datetime | None = None,
    ) -> list[RetrievalHit]:
        """Retrieve top-k active records by hybrid score (audited read).

        ``kinds`` scopes the candidate set. Defaults to long-term kinds
        (everything except ``session``); short-term recall passes
        ``kinds=["session"]`` via SessionMemory (docs/MEMORY.md § 2 § 8.5).
        """
        now = now or datetime.now(UTC)
        self._ensure_embeddings()
        if query_emb is None:
            if query is None:
                raise ValueError("provide either query or query_emb")
            query_emb = self.embedder.embed([query])[0]
        if kinds is None:
            kinds = {"fact", "episode", "semantic"}
        else:
            kinds = set(kinds)
        candidates = [
            r for r in self._active_candidates()
            if r.kind in kinds
            and (include_disputed or r.verification_status != "disputed")
            and (allow_quarantine or not r.quarantined)
        ]
        hits = retrieve(candidates, query_emb, k=k, budget_tokens=budget_tokens,
                        tokenizer=tokenizer, weights=weights, now=now)
        self._commit("query", {"k": k, "query_hash": stable_hash(query) if query else None,
                               "hits": len(hits), "kinds": sorted(kinds)})
        self.meta["read_count"] = self.meta.get("read_count", 0) + 1
        self._dirty = True
        return hits


MEMORY_KINDS_ALL = ("fact", "episode", "semantic", "session")


def stable_hash(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.sha256(text.encode()).hexdigest()[:12]
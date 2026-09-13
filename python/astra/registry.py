"""Model registry v1 (docs/VERSIONING.md § 5, docs/ARCHITECTURE.md).

Registered artifacts are immutable: each entry records the artifact sha256 plus
the identifiers needed to reproduce or audit it (semver, commit, config id,
data manifest id, eval report id). Re-registering an artifact that already
exists with a different attached state is an error; storing the same sha256
twice is a no-op that returns the existing entry.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from astra.utils import sha256_file


@dataclass
class ArtifactRecord:
    """One immutable registered model artifact (docs/VERSIONING.md § 5)."""

    name: str
    semver: str
    sha256: str
    path: str
    config_id: str
    commit: str
    data_manifest_id: str
    checklist_id: str
    eval_report_id: str
    created_at: str
    params: int
    step: int = 0
    size_bytes: int = 0
    notes: str = ""
    extra: dict = field(default_factory=dict)


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[2],
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


class ModelRegistry:
    """Append-only registry persisted as JSON.

    The default location is ``checkpoints/registry.json`` (git-ignored content,
    matching docs/ARCHITECTURE.md). Entries are keyed by sha256; lookups by
    name return the most recent registered entry for that name.

    Promotion state (Phase 6): the registry tracks an *active* entry per name
    (``active`` map). ``promote`` registers/points the active entry at a
    candidate artifact; ``rollback`` restores the previous registered entry.
    The promotion history is the append-only audit log (astra/learning/audit.py);
    the registry only stores the current active pointer per name.
    """

    SCHEMA_VERSION = 1

    def __init__(self, path: str | None = None):
        self.path = str(path) if path else "checkpoints/registry.json"
        self._entries: dict[str, ArtifactRecord] = {}
        self._active: dict[str, str] = {}
        self.load()

    @property
    def entries(self) -> dict[str, ArtifactRecord]:
        return dict(self._entries)

    @property
    def active(self) -> dict[str, str]:
        """Map of name -> sha256 currently promoted as active."""
        return dict(self._active)

    def load(self) -> None:
        if not Path(self.path).exists():
            self._entries = {}
            self._active = {}
            return
        with open(self.path) as f:
            raw = json.load(f)
        if raw.get("schema") != self.SCHEMA_VERSION:
            raise ValueError(f"unsupported registry schema {raw.get('schema')!r}")
        self._entries = {
            sha: ArtifactRecord(**rec) for sha, rec in raw.get("entries", {}).items()
        }
        self._active = dict(raw.get("active", {}))
        # validate: active pointers must reference a registered sha
        for name, sha in list(self._active.items()):
            if sha not in self._entries:
                self._active.pop(name)

    def save(self) -> str:
        payload = {
            "schema": self.SCHEMA_VERSION,
            "active": dict(sorted(self._active.items())),
            "entries": {
                sha: asdict(rec)
                for sha, rec in sorted(self._entries.items(), key=lambda kv: kv[1].created_at)
            },
        }
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        tmp = f"{self.path}.tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        Path(tmp).replace(self.path)
        return self.path

    def register(
        self,
        name: str,
        semver: str,
        path: str,
        config_id: str,
        data_manifest_id: str,
        checklist_id: str = "",
        eval_report_id: str = "",
        commit: str | None = None,
        created_at: str | None = None,
        params: int = 0,
        step: int = 0,
        notes: str = "",
        size_bytes: int = 0,
        force: bool = False,
    ) -> ArtifactRecord:
        """Register an artifact; refuses to silently replace an existing sha.

        Registering the same sha twice returns the existing entry unchanged.
        Registering a *different* artifact under an already-registered sha is an
        immutable-state error unless ``force`` is set (never use in release).
        """
        sha = sha256_file(path)
        if sha in self._entries:
            existing = self._entries[sha]
            if existing.name == name and existing.semver == semver and not force:
                return existing
            if not force:
                raise ValueError(
                    f"sha256 {sha[:12]} already registered as {existing.name} "
                    f"{existing.semver}; refusing to overwrite (immutable registry). "
                    "Pass force=True only to override as an explicit audit action."
                )
        if not config_id:
            raise ValueError("config_id is required for a registered artifact")
        if commit is None:
            commit = _git_commit()
        if created_at is None:
            created_at = datetime.now(UTC).isoformat()
        if params == 0 or step == 0:
            mpath = str(path).rsplit(".npz", 1)[0] + ".manifest.json"
            if Path(mpath).exists():
                try:
                    meta = json.loads(Path(mpath).read_text())
                    params = params or meta.get("params", 0)
                    step = step or meta.get("step", 0)
                except (OSError, ValueError):
                    pass
        rec = ArtifactRecord(
            name=name,
            semver=semver,
            sha256=sha,
            path=path,
            config_id=config_id,
            commit=commit,
            data_manifest_id=data_manifest_id,
            checklist_id=checklist_id,
            eval_report_id=eval_report_id,
            created_at=created_at,
            params=params,
            step=step,
            size_bytes=size_bytes or Path(path).stat().st_size,
            notes=notes,
        )
        self._entries[sha] = rec
        return rec

    def get(self, name: str) -> ArtifactRecord | None:
        """Latest entry registered under ``name``."""
        matches = [r for r in self._entries.values() if r.name == name]
        if not matches:
            return None
        return max(matches, key=lambda r: r.created_at)

    def by_sha(self, sha: str) -> ArtifactRecord | None:
        return self._entries.get(sha)

    def verify(self, name: str) -> tuple[ArtifactRecord | None, bool]:
        rec = self.get(name)
        if rec is None:
            return None, False
        actual = sha256_file(rec.path)
        return rec, actual == rec.sha256

    def list_names(self) -> list[str]:
        names = {r.name for r in self._entries.values()}
        return sorted(names, key=lambda n: (n,))

    # -- Phase 6: promotion / rollback --------------------------------------

    def history(self, name: str) -> list[ArtifactRecord]:
        """All registered entries for ``name``, oldest first."""
        matches = [r for r in self._entries.values() if r.name == name]
        return sorted(matches, key=lambda r: r.created_at)

    def current(self, name: str) -> ArtifactRecord | None:
        """The active entry for ``name``, or the latest if none promoted."""
        sha = self._active.get(name)
        if sha and sha in self._entries:
            return self._entries[sha]
        return self.get(name)

    def current_sha(self, name: str) -> str:
        rec = self.current(name)
        return rec.sha256 if rec else ""

    def promote(
        self,
        name: str,
        sha: str,
        *,
        created_at: str | None = None,
        notes: str = "",
    ) -> ArtifactRecord:
        """Point the active entry for ``name`` at a registered artifact.

        ``sha`` must already be registered (use ``register`` first). The
        previous active entry is *not* deleted — it stays in the registry for
        rollback. Returns the newly active record.
        """
        rec = self._entries.get(sha)
        if rec is None:
            raise ValueError(f"sha256 {sha[:12]} is not registered; register it before promotion")
        prev = self._active.get(name)
        self._active[name] = sha
        self.save()
        if prev is not None and prev != sha:
            old = self._entries.get(prev)
            if old is not None and "promoted_from" not in old.extra:
                old.extra["superseded_by"] = sha
                if created_at:
                    old.extra["superseded_at"] = created_at
                self.save()
        return rec

    def rollback(self, name: str, *, notes: str = "", created_at: str | None = None) -> ArtifactRecord | None:
        """Revert to the previous registered entry for ``name``.

        Returns the restored active record, or ``None`` when there is no
        previous entry to roll back to (already at baseline).
        """
        current_sha = self._active.get(name)
        if not current_sha:
            return None
        hist = [r.sha256 for r in self.history(name)]
        if current_sha not in hist:
            return None
        idx = hist.index(current_sha)
        if idx <= 0:
            # at baseline: nothing to roll back to, but keep current as active
            return self._entries.get(current_sha)
        prev_sha = hist[idx - 1]
        self._active[name] = prev_sha
        old = self._entries.get(prev_sha)
        if old is not None:
            old.extra["rolled_back_to"] = created_at
        self.save()
        return self._entries.get(prev_sha)


__all__ = ["ArtifactRecord", "ModelRegistry"]
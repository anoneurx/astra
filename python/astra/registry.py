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
    """

    SCHEMA_VERSION = 1

    def __init__(self, path: str | None = None):
        self.path = str(path) if path else "checkpoints/registry.json"
        self._entries: dict[str, ArtifactRecord] = {}
        self.load()

    @property
    def entries(self) -> dict[str, ArtifactRecord]:
        return dict(self._entries)

    def load(self) -> None:
        if not Path(self.path).exists():
            self._entries = {}
            return
        with open(self.path) as f:
            raw = json.load(f)
        if raw.get("schema") != self.SCHEMA_VERSION:
            raise ValueError(f"unsupported registry schema {raw.get('schema')!r}")
        self._entries = {
            sha: ArtifactRecord(**rec) for sha, rec in raw.get("entries", {}).items()
        }

    def save(self) -> str:
        payload = {
            "schema": self.SCHEMA_VERSION,
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
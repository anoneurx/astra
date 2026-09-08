"""Regression tests for the model registry v1 (docs/VERSIONING.md § 5)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astra.registry import ArtifactRecord, ModelRegistry
from astra.utils import sha256_file


@pytest.fixture()
def ckpt(tmp_path: Path) -> str:
    p = tmp_path / "mini.npz"
    np.savez_compressed(p, w=1.0)
    (tmp_path / "mini.manifest.json").write_text(json.dumps({"params": 8, "step": 42}))
    return str(p)


@pytest.fixture()
def registry(tmp_path: Path) -> ModelRegistry:
    return ModelRegistry(str(tmp_path / "registry.json"))


def _record(tmp_path: Path, name: str = "test-model") -> dict:
    return {
        "name": name,
        "semver": "0.4.0",
        "path": str(tmp_path / "mini.npz"),
        "config_id": "toy-v64",
        "data_manifest_id": "datasets/toy/manifest.json",
        "commit": "deadbeef",
        "eval_report_id": "docs/releases/v0.4.0.md",
    }


def test_round_trip_persists(ckpt, registry, tmp_path):
    registry.register(**_record(tmp_path))
    registry.save()
    reg2 = ModelRegistry(str(tmp_path / "registry.json"))
    rec = reg2.get("test-model")
    assert rec is not None
    assert rec.sha256 == sha256_file(ckpt)
    assert rec.semver == "0.4.0"
    assert rec.commit == "deadbeef"


def test_register_same_sha_is_idempotent(ckpt, registry, tmp_path):
    kw = _record(tmp_path)
    r1 = registry.register(**kw)
    r2 = registry.register(**kw)
    assert r1.sha256 == r2.sha256
    assert len(registry.entries) == 1


def test_register_conflicting_metadata_is_error(ckpt, registry, tmp_path):
    kw = _record(tmp_path)
    registry.register(**kw)
    kw["semver"] = "0.4.1"
    with pytest.raises(ValueError, match="immutable"):
        registry.register(**kw)
    # explicit override is an auditable action
    kw["force"] = True
    r = registry.register(**kw)
    assert registry.get("test-model").semver == "0.4.1"
    assert registry.get("test-model").created_at == r.created_at


def test_force_refreshes_metadata(ckpt, registry, tmp_path):
    kw = _record(tmp_path)
    registry.register(**kw)
    kw.update(force=True, notes="audit override")
    r2 = registry.register(**kw)
    assert r2.notes == "audit override"
    assert len(registry.entries) == 1


def test_verify_detects_tampering(ckpt, registry, tmp_path):
    registry.register(**_record(tmp_path))
    assert registry.verify("test-model")[1]
    with open(ckpt, "r+b") as f:  # corrupt the glued npz payload
        f.seek(16)
        f.write(b"\x00")
    assert registry.verify("test-model")[1] is False


def test_get_returns_latest_for_name(ckpt, registry, tmp_path):
    big = tmp_path / "big.npz"
    np.savez_compressed(big, w=np.arange(4))
    kw = _record(tmp_path)
    r1 = registry.register(**kw)
    kw.update(path=str(big), semver="0.4.0", name="test-model")
    r2 = registry.register(**kw)
    assert registry.get("test-model").sha256 == r2.sha256
    assert r1.sha256 != r2.sha256
    assert registry.get("missing") is None


def test_missing_manifest_defaults(ckpt, registry, tmp_path):
    (tmp_path / "mini.manifest.json").unlink()
    kw = _record(tmp_path)
    rec = registry.register(**kw)
    assert rec.params == 0 and rec.step == 0


def test_pulls_params_from_manifest(ckpt, registry, tmp_path):
    rec = registry.register(**_record(tmp_path))
    assert rec.params == 8 and rec.step == 42
    assert rec.size_bytes == Path(ckpt).stat().st_size


def test_schema_version_detected(ckpt, registry, tmp_path):
    registry.register(**_record(tmp_path))
    registry.save()
    p = tmp_path / "registry.json"
    raw = json.loads(p.read_text())
    raw["schema"] = 999
    p.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="unsupported registry schema"):
        ModelRegistry(str(p))


def test_created_at_iso_and_commit_auto(tmp_path):
    reg = ModelRegistry(str(tmp_path / "r.json"))
    np.savez_compressed(tmp_path / "mini.npz", w=2.0)
    kw = _record(tmp_path)
    rec = reg.register(**kw)
    assert "T" in rec.created_at
    assert rec.commit  # auto-detected from git or "unknown" fallback


def test_required_config_id(ckpt, registry, tmp_path):
    kw = _record(tmp_path)
    kw["config_id"] = ""
    with pytest.raises(ValueError, match="config_id"):
        registry.register(**kw)


def test_record_dataclass():
    rec = ArtifactRecord(
        name="x", semver="0.4.0", sha256="aa", path="p.npz", config_id="c",
        commit="c", data_manifest_id="d", checklist_id="", eval_report_id="",
        created_at="2026-09-08T00:00:00+00:00", params=10,
    )
    assert rec.params == 10
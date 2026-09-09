"""Tests for the Phase-6 promotion gate: GateEngine, AuditLog, registry promote/rollback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from astra.learning.audit import AuditEntry, AuditLog
from astra.learning.gates import GateEngine, GateItem
from astra.registry import ModelRegistry


def make_metrics(target=4.4, base_target=5.2, regress=2.84, base_regress=2.83):
    return {
        "target_nova_loss": {"base": base_target, "candidate": target},
        "regression_astra_loss": {"base": base_regress, "candidate": regress},
    }


GATES = [
    GateItem(id="target", type="gain", metric="target_nova_loss", min_gain=0.1),
    GateItem(id="regress", type="no_regress", metric="regression_astra_loss", max_regress=0.1),
]


def test_gate_accepts_when_gain_and_no_regression():
    eng = GateEngine(GATES, policy="auto")
    dec = eng.evaluate(make_metrics())
    assert dec.accepted is True
    assert dec.summary == "ACCEPT"
    assert all(r.passed for r in dec.results)
    # target delta ~ -0.8 gain -> passes; regression +0.01 -> passes


def test_gate_rejects_on_target_miss():
    eng = GateEngine(GATES, policy="auto")
    # candidate essentially unchanged on target (no gain) but ok on regression
    dec = eng.evaluate(make_metrics(target=5.15, base_target=5.2))
    assert dec.accepted is False
    assert dec.summary == "REJECT"
    target = next(r for r in dec.results if r.gate.id == "target")
    assert target.passed is False


def test_gate_rejects_on_regression():
    eng = GateEngine(GATES, policy="auto")
    # big regression on astra-loss
    dec = eng.evaluate(make_metrics(regress=3.5, base_regress=2.8))
    assert dec.accepted is False
    regr = next(r for r in dec.results if r.gate.id == "regress")
    assert regr.passed is False


def test_gate_manual_policy_requires_approval():
    eng = GateEngine(GATES, policy="manual")
    dec = eng.evaluate(make_metrics(), human_approval=False)
    assert dec.policy == "manual"
    assert dec.accepted is False
    assert dec.summary == "PENDING_HUMAN"
    # all gate checks passed but not approved
    assert all(r.passed for r in dec.results)
    # with approval -> accepted
    dec2 = eng.evaluate(make_metrics(), human_approval=True)
    assert dec2.accepted is True


def test_gate_absolute_type():
    eng = GateEngine([GateItem(id="ppl", type="absolute", metric="val_ppl", op="<", value=5.0)])
    dec = eng.evaluate({"val_ppl": {"base": 4.4, "candidate": 4.3}})
    assert dec.accepted is True
    dec = eng.evaluate({"val_ppl": {"base": 4.4, "candidate": 5.1}})
    assert dec.accepted is False


def test_gate_missing_metric_raises():
    eng = GateEngine(GATES, policy="auto")
    with pytest.raises(ValueError):
        eng.evaluate({"target_nova_loss": {"base": 5.0, "candidate": 4.0}})


# --- audit log -------------------------------------------------------------

def test_audit_append_only(tmp_path):
    log = AuditLog(path=str(tmp_path / "audit.jsonl"))
    log.append(AuditEntry(event="promote", model_name="m", base_sha="a", candidate_sha="b",
                          decision={"accepted": True}))
    log.append(AuditEntry(event="rollback", model_name="m", candidate_sha="b", notes="regress"))
    assert log.count() == 2
    assert log.count("promote") == 1
    assert log.count("rollback") == 1
    last = log.last("rollback")
    assert last["event"] == "rollback"
    assert last["model_name"] == "m"
    # rows are independent JSON objects on separate lines
    lines = Path(str(tmp_path / "audit.jsonl")).read_text().strip().splitlines()
    assert len(lines) == 2
    for ln in lines:
        json.loads(ln)


# --- registry promotion / rollback ----------------------------------------

@pytest.fixture()
def reg(tmp_path):
    import numpy as np

    r = ModelRegistry(path=str(tmp_path / "registry.json"))

    def make_ckpt(p, seed=0):
        p = str(p)
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, w=np.asarray([seed]))

    c1, c2 = str(tmp_path / "v1.npz"), str(tmp_path / "v2.npz")
    make_ckpt(c1, seed=1)
    make_ckpt(c2, seed=2)
    r.register(name="m", semver="0.1.0", path=c1, config_id="cfg", data_manifest_id="d1",
               created_at="2026-01-01T00:00:00Z")
    return {"reg": r, "v1": c1, "v2": c2}


def test_registry_promote_and_rollback(reg):
    r = reg["reg"]
    # baseline: latest registered is v1
    assert r.current("m").path == reg["v1"]
    # register v2 then promote
    r.register(name="m", semver="0.2.0", path=reg["v2"], config_id="cfg", data_manifest_id="d2",
               created_at="2026-01-02T00:00:00Z")
    sha2 = r.current("m").sha256
    r.promote("m", sha2, notes="test promote")
    assert r.current("m").path == reg["v2"]
    assert r.active["m"] == sha2
    assert r.current_sha("m") == sha2
    # rollback returns to v1
    restored = r.rollback("m", notes="regress detected")
    assert restored is not None
    assert restored.path == reg["v1"]
    assert r.current("m").path == reg["v1"]


def test_registry_promote_unregistered_raises(reg):
    with pytest.raises(ValueError):
        reg["reg"].promote("m", "not-a-real-sha" * 4)


def test_registry_rollback_at_baseline(reg):
    r = reg["reg"]
    sha = r.current("m").sha256
    r.promote("m", sha, notes="rebase")  # no-op active already
    assert r.rollback("m") is None or r.current("m") is not None
    assert r.current("m").path == reg["v1"]


def test_registry_history_sorted(reg):
    r = reg["reg"]
    r.register(name="m", semver="0.2.0", path=reg["v2"], config_id="cfg", data_manifest_id="d2",
               created_at="2026-01-02T00:00:00Z")
    hist = r.history("m")
    assert [h.semver for h in hist] == ["0.1.0", "0.2.0"]
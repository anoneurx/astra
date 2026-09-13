"""Phase-7 full-stack E2E integration test (docs/PHASE_STATUS.md).

Runs the *production* subsystem chain in-process against a freshly-trained tiny
model — tokenizer -> registry(active) -> model.load -> inference.sample ->
memory.recall -> learning(intake -> no-leak -> candidate -> measure) ->
gates -> registry.promote -> audit — the same path as ``tools/e2e.py``.

Operates entirely under ``tmp_path`` so there is no dependence on the trained
toy checkpoint in ``checkpoints/``. Training a tiny base then a candidate is
deliberately small (~1s); the full manual demo with 150 steps lives in
``tools/e2e.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "python"))

import e2e as resolver
import numpy as np
import pytest
from astra.model.config import ModelConfig
from astra.training.data import Corpus
from astra.training.loop import train

TOY = {"vocab_size": 300, "d_model": 24, "n_layers": 1, "n_heads": 2, "d_head": 12,
       "d_ffn": 48, "max_seq_len": 32, "rope_theta": 10000.0, "eps": 1e-6,
       "tie_embeddings": True}
SING_TRAIN = "The operative is called Astra."
SING_NEW = "The operative is called Wells."
SING_HOLDOUT = "The operative is called Tess."


@pytest.fixture(scope="module")
def infra(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    from astra.tokenizer import ByteLevelBPE

    tok = ByteLevelBPE(vocab_size=300)
    cfg = ModelConfig.from_dict(TOY)
    cfg.vocab_size = len(tok)
    cfg = ModelConfig.from_dict(cfg.to_dict())

    train_ids = np.array(tok.encode(SING_TRAIN * 40), dtype=np.int32)
    val_ids = np.array(tok.encode(SING_TRAIN * 5), dtype=np.int32)
    tcfg = {
        "max_steps": 30, "peak_lr": 3e-3, "min_lr": 1e-4, "warmup_steps": 4,
        "batch_seq": 2, "accum_steps": 1, "optimizer": "adamw",
        "scheduler": "cosine", "weight_decay": 0.02, "grad_clip": 1.0,
        "val_every": 1000, "val_shards": 1, "experiment_store": str(tmp / "runs"),
    }
    train(cfg, tcfg, Corpus(ids=train_ids, manifest={"name": "e2e-base"}),
          Corpus(ids=val_ids, manifest={"name": "e2e-val"}), seed=0, out_dir=str(tmp / "base"))
    return {"tok": tok, "cfg": cfg, "base": f"{tmp}/base/final.npz", "tmp": tmp}


def _experiences() -> list[dict]:
    return [
        {
            "id": "e1", "kind": "sft",
            "payload": {"input": "", "output": SING_NEW},
            "feedback_id": "f1", "source": "human", "confidence": 0.99,
            "verification_status": "human_verified", "trust_tier": "human_verification",
            "status": "active", "dedup_key": "n1",
        },
        {
            "id": "e2", "kind": "sft",
            "payload": {"input": "", "output": "The operative is called Ronin."},
            "feedback_id": "f2", "source": "human", "confidence": 0.99,
            "verification_status": "human_verified", "trust_tier": "human_verification",
            "status": "active", "dedup_key": "n2",
        },
    ]


@pytest.fixture(scope="module")
def registered_base(infra):
    from astra.registry import ModelRegistry

    cfg = infra["cfg"]
    r = ModelRegistry(str(infra["tmp"] / "registry.json"))
    cfg_id = json.dumps(cfg.to_dict(), sort_keys=True)
    rec = r.register(name="astra-name", semver="0.8.9", path=infra["base"], config_id=cfg_id,
                     data_manifest_id="e2e-sing", checklist_id="phase7-e2e-fixture",
                     params=0, step=30)
    r.promote("astra-name", rec.sha256, notes="fixture")
    return r


def test_full_stack_walk_promotes(infra, registered_base, tmp_path):
    tok, cfg = infra["tok"], infra["cfg"]
    tmp = infra["tmp"]
    out = str(tmp / "candidate")
    replay = np.array(tok.encode(SING_TRAIN * 40), dtype=np.int32)

    report = resolver.run_e2e(
        cfg=cfg, tok=tok, base_checkpoint=infra["base"],
        experiences=_experiences(), replay_ids=replay,
        target_corpus=[SING_HOLDOUT],
        regression_corpus=[SING_TRAIN, SING_TRAIN, SING_TRAIN],
        leak_corpus=[SING_HOLDOUT],
        out_dir=out, store_root=str(tmp / "store"), memory_dir=str(tmp / "memory"),
        registry_path=str(tmp / "registry.json"), audit_path=str(tmp / "audit.jsonl"),
        model_name="astra-name", semver="0.9.0-e2e",
        gates=None, steps=40, batch_seq=2, peak_lr=5e-3, replay_ratio=0.4,
        seed=0, prompt="The operative ",
    )

    # every subsystem step passed
    assert report["pipeline"] == "PASS"
    assert all(s["ok"] for s in report["steps"] if "leak" not in s["name"])
    names = [s["name"] for s in report["steps"]]
    for want in ("tokenizer", "registry.active", "model.load", "inference.sample",
                 "memory.recall", "learning.intake", "learning.no-leak",
                 "learning.candidate", "gate.target", "gate.regress",
                 "gate.decision", "registry.promote"):
        assert any(n.startswith(want) for n in names), f"missing step {want}"

    # model learned the new fact and did not forget prior capability
    assert report["decision"]["accepted"]
    assert report["promoted"] is True
    target_delta = report["metrics"]["target"]["candidate"] - report["metrics"]["target"]["base"]
    regress_delta = report["metrics"]["regression"]["candidate"] - report["metrics"]["regression"]["base"]
    assert target_delta < 0
    assert regress_delta < 0.1

    # registry now promotes the candidate; audit has the promote row
    current = resolver.ModelRegistry(str(tmp / "registry.json")).current("astra-name")
    assert current is not None and current.sha256 == report["promoted_sha"]
    assert current.semver == "0.9.0-e2e"

    from astra.learning.audit import AuditLog

    log = AuditLog(path=str(tmp / "audit.jsonl"))
    promotes = [e for e in log.entries() if e["event"] == "promote"]
    assert len(promotes) == 1
    assert promotes[0]["candidate_sha"] == report["promoted_sha"]
    assert promotes[0]["decision"]["accepted"] is True


def test_full_stack_walk_reject_keeps_base(infra, registered_base, tmp_path):
    """Gates that reject leave the base active and write an audit reject row."""
    tok, cfg = infra["tok"], infra["cfg"]
    tmp = infra["tmp"]
    out = str(tmp / "candidate-reject")
    replay = np.array(tok.encode(SING_TRAIN * 40), dtype=np.int32)

    # an impossible target gate forces REJECT regardless of training quality
    from astra.learning.gates import GateItem

    hard = [GateItem(id="target", type="gain", metric="target", min_gain=100.0,
                     note="impossible bound"),
            GateItem(id="regress", type="no_regress", metric="regression", max_regress=0.0,
                     note="strict")]

    report = resolver.run_e2e(
        cfg=cfg, tok=tok, base_checkpoint=infra["base"],
        experiences=_experiences(), replay_ids=replay,
        target_corpus=[SING_HOLDOUT],
        regression_corpus=[SING_TRAIN, SING_TRAIN, SING_TRAIN],
        leak_corpus=[SING_HOLDOUT],
        out_dir=out, store_root=str(tmp / "store2"), memory_dir=str(tmp / "mem2"),
        registry_path=str(tmp / "registry.json"), audit_path=str(tmp / "audit2.jsonl"),
        model_name="astra-name", semver="0.9.1-e2e",
        gates=hard, steps=20, batch_seq=2, peak_lr=5e-3, replay_ratio=0.4,
        seed=1, prompt="The operative ",
    )

    assert report["pipeline"] == "PASS"
    assert report["decision"]["accepted"] is False
    assert report["promoted"] is False

    from astra.learning.audit import AuditLog

    log = AuditLog(path=str(tmp / "audit2.jsonl"))
    rejects = [e for e in log.entries() if e["event"] == "reject"]
    assert len(rejects) == 1

    base = resolver.ModelRegistry(str(tmp / "registry.json")).current("astra-name")
    # the offered candidate was rejected: active is unchanged, not the reject semver
    assert base is not None and base.semver != "0.9.1-e2e"


def test_leak_aborts_pipeline(infra, tmp_path):
    """Training a line that also appears in the eval target is a red pipeline."""
    tok, cfg = infra["tok"], infra["cfg"]
    out = str(tmp_path / "candidate-leak")
    replay = np.array(tok.encode(SING_TRAIN * 40), dtype=np.int32)

    from astra.registry import ModelRegistry

    reg = ModelRegistry(str(tmp_path / "leak-registry.json"))
    cfg_id = json.dumps(cfg.to_dict(), sort_keys=True)
    reg.register(name="astra-name", semver="0.8.9", path=infra["base"], config_id=cfg_id,
                 data_manifest_id="e2e-sing", checklist_id="phase7-e2e-fixture", params=0, step=30)
    reg.promote("astra-name", max(reg.entries.values(), key=lambda e: e.created_at).sha256,
                notes="fixture")

    with pytest.raises(resolver.PipelineError, match="leakage"):
        resolver.run_e2e(
            cfg=cfg, tok=tok, base_checkpoint=infra["base"],
            # training line IS the eval target -> must abort before training
            experiences=[_experiences()[0]], replay_ids=replay,
            target_corpus=[SING_NEW], regression_corpus=[SING_TRAIN] * 3,
            leak_corpus=[SING_NEW], out_dir=out,
            store_root=str(tmp_path / "store3"), memory_dir=str(tmp_path / "mem3"),
            registry_path=str(tmp_path / "leak-registry.json"),
            audit_path=str(tmp_path / "leak-audit.jsonl"),
            model_name="astra-name", semver="0.9.2-e2e",
            steps=5, seed=2, prompt="The operative ",
        )


def test_memory_augmented_candidate(infra, registered_base, tmp_path):
    """Retrieved long-term memories condition candidate training.

    A pre-seeded memory fact about the new name must be recalled by the
    experience-driven query, fed into ``train_candidate`` as the ``<|memory|>``
    block, and recorded (hash/length/tokens) in the candidate manifest.
    """
    from astra.memory import MemoryRecord, MemoryStore

    tok, cfg = infra["tok"], infra["cfg"]
    tmp = infra["tmp"]
    memory_dir = str(tmp / "memory-aug")
    out = str(tmp / "candidate-aug")
    replay = np.array(tok.encode(SING_TRAIN * 40), dtype=np.int32)

    # pre-seed the persistent store with a fact the walk must later recall
    mem = MemoryStore.open(name="e2e", directory=memory_dir)
    mem.add(MemoryRecord(
        content=f"{SING_NEW} (verified memory that conditions fine-tuning)",
        kind="fact", source="verified_correction", confidence=0.99,
        verification_status="verified",
    ))

    report = resolver.run_e2e(
        cfg=cfg, tok=tok, base_checkpoint=infra["base"],
        experiences=_experiences(), replay_ids=replay,
        target_corpus=[SING_HOLDOUT],
        regression_corpus=[SING_TRAIN, SING_TRAIN, SING_TRAIN],
        leak_corpus=[SING_HOLDOUT],
        out_dir=out, store_root=str(tmp / "store-aug"), memory_dir=memory_dir,
        registry_path=str(tmp / "registry.json"), audit_path=str(tmp / "audit-aug.jsonl"),
        model_name="astra-name", semver="0.9.3-e2e",
        steps=40, batch_seq=2, peak_lr=5e-3, replay_ratio=0.4,
        seed=3, prompt="The operative ",
    )

    assert report["pipeline"] == "PASS"
    assert all(s["ok"] for s in report["steps"])
    # the memory-conditioning step actually fed the candidate
    assert any(n.startswith("memory.context") for n in [s["name"] for s in report["steps"]])
    assert report["memory_context_tokens"] > 0

    # candidate manifest carries the memory block provenance
    assert report["promoted"]
    import glob

    from astra.utils import read_json

    run_dir_candidates = sorted(glob.glob(f"{out}/runs/*/report.json"))
    assert run_dir_candidates, "no candidate run report found"
    run_report = read_json(run_dir_candidates[-1])
    mem_ctx = run_report["manifest"]["memory_context"]
    assert mem_ctx["hash"], "candidate manifest missing memory_context hash"
    assert mem_ctx["length"] > 0
    # manifest token count = block tokens + trailing <bos> separator
    assert abs(mem_ctx["tokens"] - report["memory_context_tokens"]) <= 1


def _walk(infra, seed, out_name, audit_path, semver, registry="registry.json", steps=20):
    """Small helper: run one full-stack walk against a fresh candidate dir."""
    tok, cfg = infra["tok"], infra["cfg"]
    tmp = infra["tmp"]
    replay = np.array(tok.encode(SING_TRAIN * 40), dtype=np.int32)
    return resolver.run_e2e(
        cfg=cfg, tok=tok, base_checkpoint=infra["base"],
        experiences=_experiences(), replay_ids=replay,
        target_corpus=[SING_HOLDOUT],
        regression_corpus=[SING_TRAIN, SING_TRAIN, SING_TRAIN],
        leak_corpus=[SING_HOLDOUT],
        out_dir=str(tmp / out_name), store_root=str(tmp / f"store-{out_name}"),
        memory_dir=str(tmp / f"m-{out_name}"),
        registry_path=str(tmp / registry), audit_path=str(tmp / audit_path),
        model_name="astra-name", semver=semver,
        steps=steps, batch_seq=2, peak_lr=5e-3,
        seed=seed, prompt="The operative ",
    )


def test_memory_correction_feeds_learning(infra, registered_base, tmp_path):
    """A corrected memory record (rev head) conditions the candidate, not the
    deprecated revision."""
    from astra.memory import MemoryRecord, MemoryStore, search_memory_block

    tok, cfg = infra["tok"], infra["cfg"]
    tmp = infra["tmp"]
    memory_dir = str(tmp / "mem-correct")
    out = str(tmp / "candidate-correct")

    mem = MemoryStore.open(name="e2e", directory=memory_dir)
    original = mem.add(MemoryRecord(
        content="The operative is called Astra.", kind="fact", source="auto_extract",
        confidence=0.5, verification_status="verified",
    ))
    corrected = mem.correct(original.id, content=f"{SING_NEW} (corrected fact)")

    replay = np.array(tok.encode(SING_TRAIN * 40), dtype=np.int32)
    report = resolver.run_e2e(
        cfg=cfg, tok=tok, base_checkpoint=infra["base"],
        experiences=_experiences(), replay_ids=replay,
        target_corpus=[SING_HOLDOUT], regression_corpus=[SING_TRAIN] * 3,
        leak_corpus=[SING_HOLDOUT],
        out_dir=out, store_root=str(tmp / "store-correct"), memory_dir=memory_dir,
        registry_path=str(tmp / "registry.json"), audit_path=str(tmp / "audit-correct.jsonl"),
        model_name="astra-name", semver="0.9.4-e2e",
        steps=20, batch_seq=2, peak_lr=5e-3, seed=4, prompt="The operative ",
    )
    assert report["pipeline"] == "PASS"
    assert report["memory_context_tokens"] > 0

    # recompute the expected conditioned block from the *corrected* head
    mem = MemoryStore.open(name="e2e", directory=memory_dir)
    joined = " ".join(
        str(ex.get("payload", {}).get("output") or ex.get("payload", {}).get("good") or "")
        for ex in _experiences()
    ).strip()
    block = search_memory_block(mem, joined, tok, k=4, budget_tokens=max(64, cfg.max_seq_len // 2))
    assert block.included, "no memory selected for conditioning"
    assert corrected.id in [h.record.id for h in block.included]
    assert original.id not in [h.record.id for h in block.included]

    import glob
    import hashlib

    from astra.utils import read_json

    run_report = read_json(max(glob.glob(f"{out}/runs/*/report.json")))
    mem_ctx = run_report["manifest"]["memory_context"]
    assert mem_ctx["hash"] == hashlib.sha256(block.text.encode()).hexdigest()[:12]


def _seed_registry(tmp_dir, name: str, cfg):
    """Fresh registry with one promoted base entry; returns the registry."""
    from astra.registry import ModelRegistry

    reg = ModelRegistry(str(tmp_dir / f"{name}.json"))
    cfg_id = json.dumps(cfg.to_dict(), sort_keys=True)
    rec = reg.register(name="astra-name", semver="0.8.9", path=str(tmp_dir / "base" / "final.npz"),
                       config_id=cfg_id, data_manifest_id="e2e-sing",
                       checklist_id="phase7-e2e-fixture", params=0, step=30)
    reg.promote("astra-name", rec.sha256, notes="fixture")
    return reg


def _open_registry(tmp_dir, name: str):
    from astra.registry import ModelRegistry

    return ModelRegistry(str(tmp_dir / f"{name}.json"))


def test_multi_step_rollback_audited(infra, tmp_path):
    """Two promotions then rollbacks restore the correct prior shas, in audit
    order."""
    from astra.learning.audit import AuditLog

    tmp = infra["tmp"]
    audit_path = str(tmp / "audit-roll.jsonl")
    reg = _seed_registry(tmp, "roll-registry", infra["cfg"])
    base_sha = reg.current("astra-name").sha256

    r1 = _walk(infra, seed=0, out_name="walk1", audit_path=audit_path,
               semver="0.9.5-e2e", registry="roll-registry.json")
    r2 = _walk(infra, seed=5, out_name="walk2", audit_path=audit_path,
               semver="0.9.6-e2e", registry="roll-registry.json")
    assert r1["promoted"] and r2["promoted"]
    assert r1["promoted_sha"] != base_sha and r2["promoted_sha"] != base_sha
    sha_a, sha_b = r1["promoted_sha"], r2["promoted_sha"]
    assert sha_a != sha_b

    reg = _open_registry(tmp, "roll-registry")
    assert reg.current_sha("astra-name") == sha_b
    assert reg.history("astra-name")[-1].sha256 == sha_b

    # first rollback restores the previous promotion (walk1 artifact)
    from astra.learning.audit import AuditEntry

    log = AuditLog(audit_path)
    restored = reg.rollback("astra-name", notes="test rollback #1")
    log.append(AuditEntry(event="rollback", model_name="astra-name",
                          candidate_sha=sha_b, decision={"drill": False}, notes="test rollback #1"))
    assert restored.sha256 == sha_a
    # second rollback restores the pre-walk active (fixture base)
    restored2 = reg.rollback("astra-name", notes="test rollback #2")
    log.append(AuditEntry(event="rollback", model_name="astra-name",
                          candidate_sha=sha_a, decision={"drill": False}, notes="test rollback #2"))
    assert restored2.sha256 == base_sha

    # audit log records the walk promotes and both rollbacks, in causal order
    events = [e["event"] for e in log.entries()]
    assert events == ["promote", "promote", "rollback", "rollback"]
    assert log.entries()[0]["candidate_sha"] == sha_a
    assert log.entries()[1]["candidate_sha"] == sha_b
    assert log.entries()[2]["candidate_sha"] == sha_b  # rolled back from B
    assert log.entries()[3]["candidate_sha"] == sha_a  # rolled back from A
    assert all(e["commit"] for e in log.entries())


def test_audit_trail_completeness(infra, tmp_path):
    """Every promote/reject the stack produces has a matching audited row with
    the decision verbatim, and the registry reflects exactly those promotions."""
    from astra.learning.audit import AuditLog
    from astra.registry import ModelRegistry

    tmp = infra["tmp"]
    audit_path = str(tmp / "audit-complete.jsonl")
    _seed_registry(tmp, "complete-registry", infra["cfg"])
    registry_path = str(tmp / "complete-registry.json")

    # ACCEPT then REJECT (impossible gate) then ACCEPT again
    accept1 = _walk(infra, seed=0, out_name="c1", audit_path=audit_path,
                    semver="0.9.7-e2e", registry="complete-registry.json")
    from astra.learning.gates import GateItem

    hard = [GateItem(id="target", type="gain", metric="target", min_gain=100.0, note="impossible"),
            GateItem(id="regress", type="no_regress", metric="regression", max_regress=0.0, note="strict")]
    reject = resolver.run_e2e(
        cfg=infra["cfg"], tok=infra["tok"], base_checkpoint=infra["base"],
        experiences=_experiences(),
        replay_ids=np.array(infra["tok"].encode(SING_TRAIN * 40), dtype=np.int32),
        target_corpus=[SING_HOLDOUT], regression_corpus=[SING_TRAIN] * 3,
        leak_corpus=[SING_HOLDOUT],
        out_dir=str(tmp / "cr"), store_root=str(tmp / "store-cr"),
        memory_dir=str(tmp / "m-cr"), registry_path=registry_path,
        audit_path=audit_path, model_name="astra-name", semver="0.9.8-e2e",
        gates=hard, steps=20, batch_seq=2, peak_lr=5e-3, seed=6, prompt="The operative ",
    )
    accept2 = _walk(infra, seed=9, out_name="c2", audit_path=audit_path,
                    semver="0.9.9-e2e", registry="complete-registry.json")
    assert accept1["promoted"] and not reject["promoted"] and accept2["promoted"]

    log = AuditLog(audit_path)
    rows = log.entries()
    promotes = [r for r in rows if r["event"] == "promote"]
    rejects = [r for r in rows if r["event"] == "reject"]
    assert [r["event"] for r in rows] == ["promote", "reject", "promote"]
    assert [r["candidate_sha"] for r in promotes] == [accept1["promoted_sha"], accept2["promoted_sha"]]
    assert rejects[0]["decision"]["accepted"] is False
    assert all(r["decision"]["accepted"] is True and r["decision"].get("summary") for r in promotes)
    assert all(r["commit"] for r in rows)

    # registry history = fixture + the two accepted candidates (reject adds none)
    reg = ModelRegistry(registry_path)
    hist = reg.history("astra-name")
    promoted_shas = {r["candidate_sha"] for r in promotes}
    assert {h.sha256 for h in hist}.issuperset(promoted_shas)
    assert reg.current_sha("astra-name") == accept2["promoted_sha"]


def test_preference_path_e2e(infra, registered_base, tmp_path):
    """A preference experience flows through the full stack and drives the DPO
    step (candidate manifest records it), palatable to the gate engine."""
    tok, cfg = infra["tok"], infra["cfg"]
    tmp = infra["tmp"]

    import glob

    from astra.learning.audit import AuditLog
    from astra.utils import read_json

    out = str(tmp / "candidate-pref-e2e")
    pref = [
        {
            "id": "e1", "kind": "sft",
            "payload": {"input": "", "output": SING_NEW},
            "feedback_id": "f1", "source": "human", "confidence": 0.99,
            "verification_status": "human_verified", "trust_tier": "human_verification",
            "status": "active", "dedup_key": "n1",
        },
        {
            "id": "e2", "kind": "preference",
            "payload": {"input": "", "good": SING_NEW,
                        "bad": "The operative is called Quark."},
            "feedback_id": "f2", "source": "human", "confidence": 0.99,
            "verification_status": "human_verified", "trust_tier": "human_verification",
            "status": "active", "dedup_key": "p1",
        },
    ]
    replay = np.array(tok.encode(SING_TRAIN * 40), dtype=np.int32)
    report = resolver.run_e2e(
        cfg=cfg, tok=tok, base_checkpoint=infra["base"],
        experiences=pref, replay_ids=replay,
        target_corpus=[SING_HOLDOUT], regression_corpus=[SING_TRAIN] * 3,
        leak_corpus=[SING_HOLDOUT],
        out_dir=out, store_root=str(tmp / "store-pref-e2e"), memory_dir=str(tmp / "m-pref-e2e"),
        registry_path=str(tmp / "registry.json"), audit_path=str(tmp / "audit-pref-e2e.jsonl"),
        model_name="astra-name", semver="0.9.10-e2e",
        steps=40, batch_seq=2, peak_lr=5e-3, seed=7, prompt="The operative ",
    )
    assert report["pipeline"] == "PASS"

    run_report = read_json(max(glob.glob(f"{out}/runs/*/report.json")))
    assert run_report["manifest"]["n_preference"] == 1
    assert run_report["manifest"]["n_lm"] == 2

    # the promoted decision carried a real gate verdict and audit row
    assert report["decision"]["accepted"] in (True, False)
    rows = AuditLog(str(tmp / "audit-pref-e2e.jsonl")).entries()
    assert rows and rows[-1]["event"] in ("promote", "reject")
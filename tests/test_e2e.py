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
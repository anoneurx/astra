"""Benchmark harness v1 tests (docs/ROADMAP.md Phase 3)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from astra.evaluation.bench import _satisfies, run_suite, run_val_ppl
from astra.model import LiteLM, ModelConfig
from astra.training.checkpoint import save_checkpoint
from astra.training.data import Corpus


def _tiny_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        d_model=16,
        n_layers=1,
        n_heads=4,
        d_head=4,
        d_ffn=32,
        max_seq_len=6,
    )


@pytest.mark.parametrize(
    "op,value,metric,expect",
    [
        ("<=", 5.0, 4.9, True),
        ("<=", 5.0, 5.0, True),
        ("<=", 5.0, 5.1, False),
        (">=", 50.0, 51.0, True),
        (">", 50.0, 50.0, False),
    ],
)
def test_satisfies_threshold(op, value, metric, expect):
    assert _satisfies(metric, {"op": op, "value": value}) is expect


def test_run_val_ppl_is_deterministic_and_finite():
    cfg = _tiny_cfg()
    m = LiteLM(cfg, seed=0)
    corpus = Corpus(
        ids=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.int32),
        manifest={"name": "tiny-val", "kind": "validation"},
    )
    l1 = run_val_ppl(m, corpus)
    l2 = run_val_ppl(m, corpus)
    assert np.isfinite(l1)
    assert l1 == l2


def test_run_suite_gates(tmp_path):
    """A suite with a passing threshold passes; a failing one fails."""
    cfg = _tiny_cfg()
    m = LiteLM(cfg, seed=0)
    ckpt = str(tmp_path / "tiny.npz")
    save_checkpoint(ckpt, m, opt=None, schedule=None, step=5, loss_hist=[], meta={})
    corpus = Corpus(
        ids=np.arange(cfg.vocab_size, dtype=np.int32),
        manifest={"name": "tiny-val", "kind": "validation"},
    )

    passing = tmp_path / "pass.json"
    passing.write_text(json.dumps({
        "suite": "tiny-pass", "version": 1, "items": [
            {"id": "v", "scorer": "val_ppl", "args": {"seed": 0},
             "threshold": {"op": "<", "value": 100.0}},
        ],
    }))
    report, ok = run_suite(cfg, ckpt, corpus, str(passing), str(tmp_path / "pass-result.json"))
    assert ok is True
    assert report["passed"] is True
    r = json.loads((tmp_path / "pass-result.json").read_text())
    assert r["checkpoint_sha256"]

    failing = tmp_path / "fail.json"
    failing.write_text(json.dumps({
        "suite": "tiny-fail", "version": 1, "items": [
            {"id": "v", "scorer": "val_ppl", "args": {"seed": 0},
             "threshold": {"op": "<", "value": 1e-9}},
        ],
    }))
    report, ok = run_suite(cfg, ckpt, corpus, str(failing), str(tmp_path / "fail-result.json"))
    assert ok is False
    assert report["passed"] is False
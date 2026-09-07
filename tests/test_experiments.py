"""Experiment store contracts (docs/ROADMAP.md Phase 2: queryable store)."""

from __future__ import annotations

import json

import pytest

from astra.experiments import ExperimentStore


@pytest.fixture()
def store(tmp_path):
    return ExperimentStore(root=str(tmp_path))


def test_log_get_roundtrip(store):
    run_id = store.new_run_id(seed=42)
    assert store.log(run_id, {"seed": 42, "final_val": {"loss": 2.5, "ppl": 12.2}}) == run_id
    assert store.has(run_id)
    rec = store.get(run_id)
    assert rec["run_id"] == run_id
    assert rec["final_val"]["loss"] == 2.5


def test_list_and_length(store):
    store.log(store.new_run_id(seed=1), {"seed": 1})
    store.log(store.new_run_id(seed=2), {"seed": 2})
    assert len(store) == 2
    assert len(store.list_ids()) == 2


def test_query_exact_match_top_level(store):
    store.log(store.new_run_id(seed=1), {"seed": 1, "final_val": {"loss": 2.5}})
    store.log(store.new_run_id(seed=2), {"seed": 2, "final_val": {"loss": 3.5}})
    hits = store.query(seed=2)
    assert len(hits) == 1
    assert hits[0]["seed"] == 2


def test_query_dotpath_comparison(store):
    store.log(store.new_run_id(seed=1), {"seed": 1, "final_val": {"loss": 2.5}})
    store.log(store.new_run_id(seed=2), {"seed": 2, "final_val": {"loss": 3.5}})
    under_3 = store.query(**{"final_val.loss__lt": 3.0})
    assert len(under_3) == 1
    assert under_3[0]["seed"] == 1
    over_3 = store.query(**{"final_val.loss__ge": 3.0})
    assert len(over_3) == 1
    assert over_3[0]["seed"] == 2


def test_query_persists_json(store):
    run_id = store.new_run_id(seed=7)
    store.log(run_id, {"seed": 7, "git_commit": "abc123"})
    raw = json.loads((store.root / f"{run_id}.json").read_text())
    assert raw["git_commit"] == "abc123"
    assert raw["run_id"] == run_id


def test_unknown_run_not_found(store):
    assert not store.has("does-not-exist")
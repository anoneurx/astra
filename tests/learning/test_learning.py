"""Tests for the learning engine data layer: experience store + feedback intake."""

from __future__ import annotations

import pytest
from astra.learning import TRUST_TIERS
from astra.learning.experience import (
    ExperienceStore,
    dedup_key,
    make_example,
)
from astra.learning.feedback import (
    build_preference_payload,
    build_sft_payload,
    make_feedback,
    route_to_examples,
    validate_feedback,
)


@pytest.fixture()
def store(tmp_path):
    return ExperienceStore(root=str(tmp_path / "store"))


def test_trush_tiers_exist():
    assert TRUST_TIERS[0] == "model_self_score"
    assert TRUST_TIERS[-1] == "human_verification"


def test_make_example_roundtrip(store):
    ex = make_example(
        "sft",
        {"input": "What is your name?", "output": "My name is Astra."},
        feedback_id="abc",
        source="user",
        confidence=0.9,
        verification_status="human_verified",
        trust_tier="human_verification",
    )
    eid, added = store.add(ex)
    assert added is True
    assert store.get(eid)["kind"] == "sft"
    assert store.active() == [ex]


def test_dedup_key():
    # identical payload -> identical key regardless of id
    a = dedup_key("sft", {"input": "x", "output": "y"})
    b = dedup_key("sft", {"input": "x", "output": "y"})
    assert a == b
    # different kind -> different key
    assert a != dedup_key("preference", {"input": "x", "good": "y", "bad": "z"})


def test_add_dedup_is_noop(store):
    ex1 = make_example(
        "sft",
        {"input": "q", "output": "a"},
        feedback_id="f1",
        source="user",
        confidence=0.7,
        verification_status="verified",
        trust_tier="verified_correction",
    )
    ex2 = make_example(
        "sft",
        {"input": "q", "output": "a"},
        feedback_id="f2",  # different provenance, same content
        source="user",
        confidence=0.7,
        verification_status="verified",
        trust_tier="verified_correction",
    )
    a, added1 = store.add(ex1)
    b, added2 = store.add(ex2)
    assert added1 is True and added2 is False
    assert a == b  # same example returned as the existing one
    assert len(store.active()) == 1


def test_withdraw_and_quarantine(store):
    ex = make_example(
        "sft",
        {"input": "q", "output": "a"},
        feedback_id="f1",
        source="user",
        confidence=0.7,
        verification_status="verified",
        trust_tier="verified_correction",
    )
    eid, _ = store.add(ex)
    assert store.quarantine(eid, "flagged")
    assert store.get(eid)["status"] == "quarantined"
    assert store.active() == []
    assert store.withdraw(eid)
    assert store.get(eid)["status"] == "withdrawn"


def test_validate_feedback_accepts_verified(store):
    fb = make_feedback(
        "human",
        "human_verification",
        "The capital of France is Paris.",
        confidence=0.95,
        verification_status="human_verified",
        context="Where is the capital of France?",
    )
    res = validate_feedback(fb, store=store)
    assert res["result"] == "accepted"
    r = route_to_examples(fb, "sft", build_sft_payload(fb, "Paris"), store)
    assert r["added"] is True
    assert store.active()[0]["feedback_id"] == fb["id"]


def test_validate_feedback_quarantines_low_confidence(store):
    fb = make_feedback(
        "user",
        "user_signal",
        "The sky is green.",
        confidence=0.2,  # below user_signal min 0.6
        verification_status="verified",
        context="What color is the sky?",
    )
    res = validate_feedback(fb, store=store)
    assert res["result"] == "quarantined"
    assert any("low_confidence" in e for e in res["errors"])


def test_validate_feedback_quarantines_contradiction(store):
    def check(text):
        return "contradiction" in text

    fb = make_feedback(
        "user",
        "automated_signal",
        "contradiction text",
        confidence=0.95,
        verification_status="automated",
    )
    res = validate_feedback(fb, contradiction_check=check)
    assert res["result"] == "quarantined"
    assert "contradicts_high_confidence_memory" in res["reasons"]


def test_validate_feedback_duplicate_already_an_example(store):
    fb = make_feedback(
        "human",
        "human_verification",
        "The name of this system is Astra.",
        confidence=0.99,
        verification_status="human_verified",
    )
    assert validate_feedback(fb, store=store)["result"] == "accepted"
    route_to_examples(fb, "sft", build_sft_payload(fb, "Astra"), store)
    assert validate_feedback(fb, store=store)["result"] == "duplicate"


def test_route_preference_payload():
    fb = make_feedback(
        "user", "user_signal", "good one",
        confidence=0.8, verification_status="verified",
    )
    p = build_preference_payload(fb, "good", "bad")
    assert p["good"] == "good" and p["bad"] == "bad"
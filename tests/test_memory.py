"""Memory engine v1 tests (docs/MEMORY.md, Phase 4).

Covers: the immutable record schema, both embedders, the versioned store +
lifecycle (correct/delete/expire/dispute/resolve), flat retrieval with hybrid
ranking + token budget, quarantine/leakage behaviour, working-context
injection, the short/long-term session split (SessionMemory + promotion), and
the append-only audit log.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from astra.memory import (
    HashEmbedder,
    MemoryRecord,
    MemoryStore,
    SessionMemory,
    build_memory_block,
    corrected_record,
    promote,
)
from astra.model import LiteLM, ModelConfig
from astra.tokenizer import ByteLevelBPE

DIM = 32


@pytest.fixture
def store(tmp_path):
    return MemoryStore(name="test", directory=tmp_path, embedder=HashEmbedder(DIM))


def _rec(content: str, **kw) -> MemoryRecord:
    return MemoryRecord(content=content, **kw)


# ------------------------------------------------------------------ records

def test_record_roundtrip_with_embedding():
    rec = MemoryRecord(content="Paris is the capital of France.", kind="fact", confidence=0.9,
                       tags=["geo"], embedding=np.arange(8, dtype="float32"))
    d = rec.to_dict()
    assert "embedding" in d
    back = MemoryRecord.from_dict(d)
    assert back.content == rec.content and back.id == rec.id
    assert (back.embedding == rec.embedding).all()


def test_record_validate_rejects_bad_fields():
    with pytest.raises(ValueError):
        _rec("  ").validate()
    with pytest.raises(ValueError):
        _rec("x", kind="nope").validate()
    with pytest.raises(ValueError):
        _rec("x", confidence=1.5).validate()
    with pytest.raises(ValueError):
        _rec("x", verification_status="maybe").validate()
    with pytest.raises(ValueError):
        _rec("x", revision=0).validate()
    with pytest.raises(ValueError):
        _rec("x", created_at="2026-01-01T00:00:00", expires_at="2025-01-01T00:00:00").validate()


def test_corrected_record_bumps_revision():
    r = _rec("old")
    new = corrected_record(r, "new")
    assert new.revision == r.revision + 1
    assert new.deprecates == r.id
    assert new.embedding is None  # re-embedded lazily by the store


# ---------------------------------------------------------------- embedders

def test_hash_embedder_deterministic_normalized():
    e = HashEmbedder(DIM)
    v1 = e.embed(["Paris is the capital of France."])
    v2 = e.embed(["Paris is the capital of France."])
    assert v1.shape == (1, DIM)
    assert (v1 == v2).all()
    assert abs(float((v1[0] ** 2).sum()) - 1.0) < 1e-5
    assert e.config == e.config


def test_litelm_extractor_shapes_and_determinism():
    cfg = ModelConfig(vocab_size=128, d_model=16, n_layers=2, n_heads=4, d_head=4,
                      d_ffn=32, max_seq_len=8)
    model = LiteLM(cfg, seed=0)
    from astra.memory import LiteLMExtractor

    tok = ByteLevelBPE(vocab_size=128)
    ex = LiteLMExtractor(model, tok)
    embs = ex.embed(["hello world", "a b"])
    assert embs.shape[0] == 2 and embs.shape[1] == cfg.d_model
    assert (ex.embed(["hello world"]) == ex.embed(["hello world"])).all()
    assert ex.dim == cfg.d_model
    assert ex.embed([]).shape == (0, cfg.d_model)


# -------------------------------------------------------------------- store

def test_store_add_and_get(store):
    r = store.add(_rec("hello world"))
    got = store.get(r.id)
    assert got is not None and got.content == "hello world"
    assert got.embedding is not None


def test_store_persist_roundtrip(tmp_path):
    s1 = MemoryStore(name="rt", directory=tmp_path, embedder=HashEmbedder(DIM))
    r = s1.add(_rec("persist me", tags=["a"]))
    s2 = MemoryStore.open("rt", tmp_path, embedder=HashEmbedder(DIM))
    got = s2.get(r.id)
    assert got is not None and got.content == "persist me"
    assert (got.embedding == r.embedding).all()
    assert s2.version_key == s1.version_key


def test_store_rejects_embedding_drift(tmp_path):
    MemoryStore(name="drift", directory=tmp_path, embedder=HashEmbedder(DIM)).add(_rec("x"))
    with pytest.raises(ValueError, match="embedding mismatch"):
        MemoryStore.open("drift", tmp_path, embedder=HashEmbedder(DIM + 8))


def test_correct_creates_revision_and_head(store):
    a = store.add(_rec("paris is france capital", kind="fact"))
    store.add(_rec("paris is france capital", kind="fact"))
    new = store.correct(a.id, content="Paris is the capital of France.")
    assert new.revision == 2 and new.deprecates == a.id
    assert new.id != a.id
    assert store.get(a.id) is None  # deprecated revision hidden by default
    assert store.get(a.id, include_all=True).revision == 1
    assert store.get_latest(a.id).id == new.id
    with pytest.raises(ValueError):
        store.add(MemoryRecord(content="dup", id=a.id))


def test_soft_delete_and_purge(store):
    r = store.add(_rec("delete me"))
    store.soft_delete(r.id)
    assert store.get(r.id) is None
    assert store.get(r.id, include_all=True) is not None
    assert [x.id for x in store.records(include_deleted=True)] == [r.id]
    store.soft_delete(r.id, purge=True)
    assert store.get(r.id, include_all=True) is None


def test_expire_removes_expired(store):
    store.add(_rec("volatile news", created_at="2020-01-01T00:00:00+00:00",
                   expires_at="2020-02-01T00:00:00+00:00"))
    store.add(_rec("stable fact"))
    assert store.expire() == 1
    assert len(store.records()) == 1


def test_dispute_conflicts_resolve(store):
    a = store.add(_rec("claim one", kind="fact"))
    b = store.add(_rec("claim two", kind="fact"))
    store.mark_disputed([a.id, b.id], reason="same topic, conflicting facts")
    assert {x.id for x in store.conflicts()} == {a.id, b.id}
    assert all(x.verification_status == "disputed" for x in store.conflicts())
    rec = store.resolve(a.id, by="human-crosscheck")
    assert rec.verification_status == "verified"
    assert [x.id for x in store.conflicts()] == [b.id]


# ---------------------------------------------------------------- retrieval

def test_search_ranks_relevant_first(store):
    store.add(_rec("Paris is the capital of France.", tags=["geo"]))
    store.add(_rec("photosynthesis converts light into chemical energy.", tags=["bio"]))
    hits = store.search("capital of france")
    assert hits[0].record.content.startswith("Paris")
    assert hits[0].cosine > hits[1].cosine


def test_retrieval_excludes_deleted_and_disputed(store):
    other = store.add(_rec("spain capital", tags=["geo"]))
    target = store.add(_rec("france capital", tags=["geo"]))
    del_rec = store.add(_rec("france capital paris", tags=["geo"]))
    store.soft_delete(del_rec.id)
    store.mark_disputed([target.id])
    ids = [h.id for h in store.search("france capital paris")]
    assert target.id not in ids and del_rec.id not in ids
    assert other.id in ids
    ids_inc = [h.id for h in store.search("france", include_disputed=True)]
    assert target.id in ids_inc


def test_retrieval_excludes_quarantine(store):
    store.add(_rec("SECRET BENCHMARK ANSWER 99", tags=["eval-quarantine"]))
    store.add(_rec("ordinary fact about numbers", tags=[]))
    hits = store.search("SECRET BENCHMARK ANSWER 99")
    assert not any("SECRET" in h.record.content for h in hits)
    hits = store.search("SECRET BENCHMARK ANSWER 99", allow_quarantine=True)
    assert any("SECRET" in h.record.content for h in hits)


def test_token_budget_enforced(store):
    a = store.add(_rec("the quick brown fox jumps over the lazy dog"))
    store.add(_rec("another completely unrelated topic sentence"))
    tok = ByteLevelBPE(vocab_size=64)
    hits_nobudget = store.search("quick brown fox lazy dog")
    budget = len(tok.encode(a.content))
    hits = store.search("quick brown fox lazy dog", k=8, budget_tokens=budget, tokenizer=tok)
    assert hits[0].id == a.id
    assert len(hits) == 1
    assert hits_nobudget[0].id == a.id


def test_k_limits_number_of_hits(store):
    for i in range(5):
        store.add(_rec(f"shared topic record number {i}"))
    hits = store.search("shared topic", k=3)
    assert len(hits) == 3


def test_recency_breaks_identical_cosine(store):
    old = store.add(_rec("fact A words here", created_at="2020-01-01T00:00:00+00:00"))
    fresh = store.add(_rec("fact A words here", created_at="2026-01-01T00:00:00+00:00"))
    hits = store.search("fact A words here", weights={"cosine": 0.0, "recency": 1.0,
                                                      "confidence": 0.0, "kind": 0.0})
    assert hits[0].id == fresh.id
    assert old.id != fresh.id


def test_lazy_reembed_after_stripped_embeddings(tmp_path, store):
    r = store.add(_rec("lazily re-embedded fact"))
    # simulate a legacy/failed write where embeddings were dropped on disk
    data = json.loads(store.path.read_text())
    del data["entries"][r.id]["record"]["embedding"]
    store.path.write_text(json.dumps(data))
    reopened = MemoryStore.open("test", tmp_path, embedder=HashEmbedder(DIM))
    hits = reopened.search("lazily re-embedded fact")
    assert hits and hits[0].record.content.startswith("lazily")


# ---------------------------------------------------------------- injection

def test_build_memory_block_format_and_budget(store):
    store.add(_rec("first thing to remember"))
    store.add(_rec("second thing to remember, longer"))
    tok = ByteLevelBPE(vocab_size=64)
    hits = store.search("remember", k=8)
    block = build_memory_block(hits, tok, budget_tokens=2)
    assert block.text.startswith("<|memory|>")
    assert block.text.endswith("<|/memory|>")
    assert len(block.included) >= 1 and block.tokens
    assert block.prepend("hi").startswith("<|memory|>")
    assert build_memory_block([], tok).text == ""


# -------------------------------------------------------------------- audit

def test_audit_log_tracks_writes_and_reads(store):
    r = store.add(_rec("audited fact"))
    store.soft_delete(r.id)
    store.search("audited")
    lines = store.audit_path.read_text().strip().splitlines()
    assert any('"action": "add"' in ln and r.id in ln for ln in lines)
    assert any('"action": "delete"' in ln for ln in lines)
    assert any('"action": "query"' in ln for ln in lines)


# ---------------------------------------------------------------- leakage

def test_memory_store_lives_outside_datasets(tmp_path):
    s = MemoryStore(name="data", directory=tmp_path)
    assert "datasets" not in str(s.path.resolve())
    s.add(_rec("fact never near training data"))
    assert not any(p.parts[0] == "datasets" for p in (s.path, s.audit_path))


def test_quarantine_tag_is_the_leakage_gate(store):
    assert "eval-quarantine" in MemoryRecord(content="x", tags=["eval-quarantine"]).to_dict()["tags"]
    r = store.add(_rec("LEAK MARKER CONTENT", source="auto_extract", tags=["eval-quarantine"]))
    assert r.quarantined
    assert not any("LEAK" in h.record.content for h in store.search("LEAK MARKER CONTENT"))


# ------------------------------------------------- short/long-term session split

def _session(store, sid="sess-1", ttl_hours=24.0):
    return SessionMemory(store, sid, ttl=timedelta(hours=ttl_hours))


def test_session_add_turn_scoped_and_durable_excluded(store):
    s = _session(store)
    recs = s.add_turn(user="user set the breaker", assistant="breaker reset to 5A")
    assert len(recs) == 2
    assert all(r.kind == "session" for r in recs)
    assert all(r.tags == ["session:sess-1"] for r in recs)
    assert any(r.source == "user_said" for r in recs)
    assert any(r.source == "model_generated" for r in recs)
    assert all(r.verification_status == "verified" for r in recs)
    # long-term search must NOT surface session records
    assert not any(h.record.kind == "session" for h in store.search("breaker"))


def test_session_recall_only_within_its_session(store):
    a = _session(store, "sess-a")
    b = _session(store, "sess-b")
    a.add_turn(user="Lunar module landing site was the Sea of Tranquility")
    b.add_turn(user="Solar panels angle toward the sun")
    ra = a.recall("where did the moon lander touch down")
    rb = b.recall("where did the moon lander touch down")
    assert len(ra) == 1 and "Tranquility" in ra[0].record.content
    assert len(rb) == 1 and "sun" in rb[0].record.content


def test_session_ttl_expiry(store):
    s = _session(store, ttl_hours=1.0)
    recs = s.add_turn(user="meeting in room 4B at noon")
    assert len(s.records()) == 1
    late = datetime.now(UTC) + timedelta(hours=2)
    n = s.purge_expired(now=late)
    assert n == 1
    assert s.records() == []
    assert s.recall("room", now=late) == []
    assert store.get(recs[0].id) is None


def test_session_close_deletes_its_records(store):
    s = _session(store)
    s.add_turn(user="todo: ship the manifest", assistant="logged")
    s.add_turn(user="also: prune tmp files")
    assert len(s.records()) == 3
    assert s.close() == 3
    assert s.records() == []
    assert s.recall("manifest") == []
    assert store.records(kinds=["session"]) == []


def test_long_term_search_requires_session_kind_explicit(store):
    s = _session(store, "sess-x")
    s.add_turn(user="THE UNIQUE FACT 928374")
    # default search excludes it
    assert not any(h.record.content == "THE UNIQUE FACT 928374" for h in store.search("928374"))
    # explicit kinds=["session"] reaches it
    hits = store.search("928374", kinds=["session"])
    assert any(h.record.content == "THE UNIQUE FACT 928374" for h in hits)


def test_promote_makes_durable_record_with_attribution(store):
    s = _session(store)
    turns = s.add_turn(user="Paris is the capital of France")
    src = turns[0]
    long = promote(s, src.id, kind="fact", confidence=0.95, tags=["geo"])
    assert long.id != src.id
    assert long.kind == "fact"
    assert long.verification_status == "verified"
    assert f"promoted:from_session:{src.id}" in long.attribution
    assert "promoted:session" in long.tags
    assert any(h.record.id == long.id for h in store.search("capital of France"))
    assert src.kind == "session"  # source untouched
    # promotion source must belong to the session
    other = store.add(_rec("foreign fact"))
    with pytest.raises(ValueError, match="does not belong"):
        promote(s, other.id)

# ------------------------------------------------------------- compaction

def test_compact_prunes_soft_deleted_and_dedups_duplicates(store):
    a = store.add(_rec("the same fact content", kind="fact"))
    b = store.add(_rec("the same fact content", kind="fact"))
    c = store.add(_rec("a unique fact", kind="fact"))
    store.soft_delete(c.id)

    assert len(store.records()) == 2  # a, b active; c deleted
    stats = store.compact()
    assert stats["pruned_deleted"] == 1
    assert stats["deduped"] == 1
    assert stats["removed"] == 2
    assert stats["active"] == 1

    live = store.records()
    assert len(live) == 1
    assert live[0].id == b.id  # newest of the identical pair is kept
    assert live[0].content == "the same fact content"
    # a is deprecated, not physically gone (history preserved)
    assert store.get(a.id, include_all=True) is not None
    assert store.get(a.id) is None
    # the soft-deleted record was physically pruned by compaction
    assert store.get(c.id, include_all=True) is None
    # every removal was audited
    actions = [line["action"] for line in map(json.loads,
                                              store.audit_path.read_text().splitlines())]
    assert "compact" in actions


def test_compact_scoped_to_kind(store):
    store.add(_rec("dup", kind="fact"))
    store.add(_rec("dup", kind="fact"))
    store.add(_rec("dup", kind="semantic"))
    stats = store.compact(kinds=["semantic"])
    assert stats["deduped"] == 0
    stats = store.compact(kinds=["fact"])
    assert stats["deduped"] == 1

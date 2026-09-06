"""Tokenizer tests (EX-01). Byte-level BPE correctness and round-trip guarantees."""

from __future__ import annotations

import random
import string

import pytest

from astra.tokenizer import ByteLevelBPE, PAD_ID, UNK_ID
from astra.utils import stable_seed


@pytest.fixture(scope="module")
def tok() -> ByteLevelBPE:
    return ByteLevelBPE(vocab_size=64, min_frequency=2)


UNICODE_STRINGS = [
    "hello world",
    "hello, 世界 — how are you?",
    "مرحبا بالعالم",
    "a\u00a0b\tc\n",
    "emoji: \U0001f600\U0001f680 done",
    "lone surrogate: \ud800 inside",
    "".join(random.Random(1).choices(string.printable, k=500)),
    "\x00\x01\x02 bytes",
    "a" * 10_000,
]


def test_roundtrip_exact(tok):
    for s in UNICODE_STRINGS:
        ids = tok.encode(s)
        assert tok.decode(ids) == s, s[:40]


def test_deterministic_encode(tok):
    s = "deterministic encoding test 42"
    assert tok.encode(s) == tok.encode(s)


def test_empty_string(tok):
    assert tok.decode([]) == ""
    assert tok.encode("") == []


def test_ids_within_range(tok):
    s = "range check \U0001f600"
    ids = tok.encode(s)
    assert all(0 <= i < len(tok) for i in ids)


def test_byte_level_no_unknown():
    t = ByteLevelBPE(vocab_size=32)
    full = "".join(chr(c) for c in range(256)) * 2 + "\U0001f600"
    ids = t.encode(full)
    assert UNK_ID not in ids


def test_special_tokens_reserved(tok):
    assert PAD_ID == 256
    assert tok.id_to_piece[256] == b"<pad>"


def test_merges_reduce_tokens():
    corpus = ("ab" * 5000 + "cd" * 5000) * 10
    t = ByteLevelBPE(vocab_size=300, min_frequency=5).train(corpus)
    assert t.stats["tokens_after"] <= t.stats["tokens_seen"]
    assert len(t) > 256


def test_train_on_str_and_bytes_identical():
    corpus = "The quick brown fox jumps over the lazy dog. " * 200
    a = ByteLevelBPE(vocab_size=64, min_frequency=2).train(corpus)
    b = ByteLevelBPE(vocab_size=64, min_frequency=2).train(corpus.encode("utf-8"))
    assert [m[:2] for m in a.merges] == [m[:2] for m in b.merges]


def test_save_load_roundtrip(tmp_path):
    corpus = (string.ascii_letters * 1000) * 20
    t = ByteLevelBPE(vocab_size=128).train(corpus)
    p = tmp_path / "t.json"
    t.save(p)
    t2 = ByteLevelBPE.load(p)
    s = "round trip persistence \U0001f600"
    assert t2.decode(t2.encode(s)) == s
    assert len(t2) == len(t)


def test_seed_hasher_deterministic():
    assert stable_seed("abc") == stable_seed("abc")
"""Safety gates + data hygiene tests (EX-05)."""

from __future__ import annotations

import numpy as np
import pytest

from astra.safety import (
    deduplicate_lines,
    fingerprint,
    generate_manifest,
    leak_check,
    min_length_filter,
    ngram_set,
    sanitize_control,
)


def test_sanitize_control_keeps_newline_tab():
    src = "a\x00b\x01c\nd\te\x7f"
    assert sanitize_control(src) == "abc\nd\te"


def test_dedup_preserves_order_and_counts_removal():
    lines = ["a", "b", "a", "c", "b"]
    kept, removed = deduplicate_lines(lines)
    assert kept == ["a", "b", "c"]
    assert removed == 2


def test_min_length_filter(tmp_path):
    kept, removed = min_length_filter(["short", "this is sufficiently long"], min_chars=10)
    assert kept == ["this is sufficiently long"]
    assert removed == 1


def test_leak_check_clean_sets():
    tr = list(range(1000))
    hd = list(range(2000, 3000))
    r = leak_check(tr, hd, n=4)
    assert r["leak_free"] is True
    assert r["overlap_fraction"] == 0.0


def test_leak_check_detects_overlap():
    tr = list(range(0, 100))
    hd = list(range(50, 150))  # shares 4-grams like (50,51,52,53)
    r = leak_check(tr, hd, n=4)
    assert r["overlap_ngrams"] > 0
    assert r["leak_free"] is False


def test_ngram_set():
    assert ngram_set([1, 2, 3, 4], n=3) == {(1, 2, 3), (2, 3, 4)}
    assert ngram_set([1, 2], n=4) == set()


def test_fingerprint_stable():
    assert fingerprint("same text") == fingerprint("same text")
    assert fingerprint("same text") != fingerprint("other")


def test_manifest_schema():
    m = generate_manifest(
        name="toy-train", kind="training", corpus_path="x.txt",
        cleaning_rules=["utf8-normalize-v1"], filters=["pii-v1"],
        sha256="abcd", docs=100, chars=5000,
    )
    assert m["hash"]["value"] == "abcd"
    assert m["contamination_checked"] is True
    assert m["kind"] == "training"
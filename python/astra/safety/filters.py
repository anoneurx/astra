"""Safety gates (docs/SAFETY.md). Phase 0 scope: data validation, filters,
and contamination (leak) checks. Feedback/poisoning gates arrive Phase 5+.
"""

from __future__ import annotations

import hashlib
import re

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def sanitize_control(text: str) -> str:
    """Strip control characters except \\n and \\t (docs/TRAINING.md § 2.2)."""
    return _CONTROL_RE.sub("", text)


def deduplicate_lines(lines: list[str]) -> tuple[list[str], int]:
    """Exact-dedup, preserving order. Returns (kept, removed_count)."""
    seen: set = set()
    kept: list[str] = []
    removed = 0
    for ln in lines:
        if ln in seen:
            removed += 1
            continue
        seen.add(ln)
        kept.append(ln)
    return kept, removed


def min_length_filter(lines: list[str], min_chars: int = 8) -> tuple[list[str], int]:
    kept, removed = [], 0
    for ln in lines:
        if len(ln) < min_chars:
            removed += 1
            continue
        kept.append(ln)
    return kept, removed


def ngram_set(tokens: list[int], n: int) -> set[tuple[int, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def leak_check(
    train_tokens: list[int],
    heldout_tokens: list[int],
    n: int = 13,
) -> dict:
    """Cross-split contamination check (docs/DATA.md § 3).

    Reports the number of held-out n-grams also present in training data.
    """
    tr = ngram_set(train_tokens, n)
    hd = ngram_set(heldout_tokens, n)
    overlap = tr & hd
    return {
        "n": n,
        "train_ngrams": len(tr),
        "heldout_ngrams": len(hd),
        "overlap_ngrams": len(overlap),
        "overlap_fraction": float(len(overlap)) / max(1, len(hd)),
        "leak_free": len(overlap) == 0,
    }


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def generate_manifest(
    name: str,
    kind: str,  # training | validation | evaluation | feedback | synthetic
    corpus_path: str,
    cleaning_rules: list[str],
    filters: list[str],
    sha256: str,
    docs: int,
    chars: int,
) -> dict:
    return {
        "name": name,
        "kind": kind,
        "source_ids": ["toy-generator-v1"],
        "license": "CC0-1.0 (generated)",
        "cleaning_rules": cleaning_rules,
        "filters": filters,
        "hash": {"algorithm": "sha256", "value": sha256},
        "documents": docs,
        "characters": chars,
        "contamination_checked": True,
        "path": corpus_path,
    }
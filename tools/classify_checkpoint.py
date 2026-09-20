#!/usr/bin/env python3
"""Classify a downloaded astra checkpoint: word-prose vs word-chat (or unknown).

The .npz from Colab/Kaggle carries only w:/m:/v: arrays (the manifest is a
separate sidecar). This compares the checkpoint against known word-prose and
word-chat bases and reports a verdict:

    what  ./tools/classify_checkpoint.py /path/to/final.npz

Verdict rules (median mean-abs-diff over 44 shared weight keys):
  <0.02 to chat-gpu   -> chat fine-tune (3800 steps)
  <0.02 to prose-*    -> prose base (5400 steps)
  0.02..0.35          -> different model (chat vs prose confusion)
  else                -> not comparable to known checkpoints
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def weight_keys(f) -> list[str]:
    return [k for k in f.files if k.startswith("w:")]


def median_diff(target, ref) -> tuple[float, int]:
    target = np.load(target, mmap_mode="r")
    ref = np.load(ref, mmap_mode="r")
    keys = [k for k in weight_keys(target) if k in ref.files]
    if not keys:
        return float("inf"), 0
    diffs = []
    for k in keys:
        a = target[k][...].astype(np.float64)
        b = ref[k][...].astype(np.float64)
        diffs.append(float(np.abs(a - b).mean()))
    return float(np.median(diffs)), len(keys)


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    target = Path(sys.argv[1])
    if not target.is_file():
        print(f"no such file: {target}")
        sys.exit(1)

    f = np.load(target, mmap_mode="r")
    n_w = len(weight_keys(f))
    print(f"file: {target} ({target.stat().st_size:,} bytes)")
    print(f"w-keys: {n_w} (word model = 44)")

    refs = {
        "prose-gpu": REPO / "checkpoints/astra5m_word_prose_gpu/resumed/final.npz",
        "prose-cpu": REPO / "checkpoints/astra5m_word_prose/resumed/final.npz",
        "chat-gpu": REPO / "checkpoints/astra5m_word_chat/resumed/final.npz",
    }
    results = {}
    for name, path in refs.items():
        if Path(path).is_file():
            d, n = median_diff(target, path)
            results[name] = d
            print(f"  vs {name:10s}: median-abs-diff={d:.4f} ({n} keys)")

    if n_w != 44:
        print("VERDICT: not a word-level model checkpoint (wrong key count)")
        return
    if results:
        best_name, best = min(results.items(), key=lambda kv: kv[1])
        if best < 0.02:
            if best_name.startswith("chat"):
                verdict = "CHAT fine-tune (3800 steps) - matches the known chat model"
            else:
                verdict = "prose base (5400 steps) - matches a known prose run"
        elif best < 0.35:
            verdict = "different from all known references (chat vs prose?)"
        else:
            verdict = "unknown - not comparable to the known checkpoints"
    else:
        verdict = "no reference checkpoints available"
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
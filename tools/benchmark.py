#!/usr/bin/env python3
"""Model benchmark runner (docs/ROADMAP.md Phase 3, docs/BENCHMARKS.md).

Runs all suites under benchmarks/suites against a checkpoint and prints a
pass/fail gate. Requires a trained checkpoint (e.g. checkpoints/phase0/final.npz).

Usage:
  python tools/benchmark.py --checkpoint checkpoints/phase0/final.npz \
      --config configs/toy_pretrain.json --out benchmarks/results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from astra.evaluation.bench import run_suite
from astra.model import ModelConfig
from astra.tokenizer import ByteLevelBPE
from astra.training import Corpus
from astra.utils import read_json, sha256_file


def _val_corpus(raw: dict) -> Corpus:
    import numpy as np

    path = Path(raw["data"]["val"])
    ids = ByteLevelBPE.load(raw["tokenizer"]).encode(path.read_text(encoding="utf-8"))
    return Corpus(
        ids=np.array(ids, dtype=np.int32),
        manifest={"name": "toy-val", "kind": "validation", "sha256": sha256_file(path)},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help=".npz checkpoint")
    ap.add_argument("--config", default="configs/toy_pretrain.json")
    ap.add_argument("--out", default="benchmarks/results", help="output dir")
    ap.add_argument("--suites", nargs="*", default=None, help="suite names (all if omitted)")
    args = ap.parse_args()

    raw = read_json(args.config)
    cfg = ModelConfig.from_dict(raw["model"])
    val_corpus = _val_corpus(raw)
    suite_dir = Path("benchmarks/suites")
    suites = args.suites or [p.stem for p in sorted(suite_dir.glob("*.json"))]

    all_ok = True
    for name in suites:
        suite_path = suite_dir / f"{name}.json"
        if not suite_path.exists():
            print(f"[benchmark] SKIP (no manifest) {name}")
            continue
        ckpt_dir = Path(args.out) / args.checkpoint.replace("/", "_") / name.replace("/", "_")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        out_json = str(ckpt_dir / "result.json")
        report, passed = run_suite(cfg, args.checkpoint, val_corpus, str(suite_path), out_json)
        summary = {i["id"]: (i["metric"], i["threshold_met"]) for i in report["items"]}
        print(f"[benchmark] {name}: {'PASS' if passed else 'FAIL'}  {json.dumps(summary)}")
        all_ok = all_ok and passed
    print(f"[benchmark] overall: {'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
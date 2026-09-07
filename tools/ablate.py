#!/usr/bin/env python3
"""Run ablation experiments over architecture variants (docs/ROADMAP.md Phase 3).

Each variant flips exactly one field of the baseline config (norm_type, ffn_type,
pos_type) and is trained under an identical train config, seed, and data, so the
only source of val-loss difference is the variant. Results are written (one
JSON file per variant plus an index) under ``--out`` and logged to the run store.

Usage:
  python tools/ablate.py --config configs/toy_pretrain.json \
      --variants rms-vs-ln gelu-vs-swiglu rope-vs-learned --steps 400
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from astra.model import LiteLM, ModelConfig
from astra.tokenizer import ByteLevelBPE
from astra.training import Corpus, train
from astra.utils import read_json, sha256_file

VARIANTS = {
    "rms-vs-ln": {
        "baseline": {"norm_type": "rmsnorm"},
        "variant": {"norm_type": "layernorm"},
        "axis": "norm_type",
    },
    "gelu-vs-swiglu": {
        "baseline": {"ffn_type": "swiglu"},
        "variant": {"ffn_type": "gelu"},
        "axis": "ffn_type",
    },
    "rope-vs-learned": {
        "baseline": {"pos_type": "rope"},
        "variant": {"pos_type": "learned"},
        "axis": "pos_type",
    },
}


def build_corpus(raw: dict, split: str) -> Corpus:
    path = Path(raw["data"][split])
    ids = ByteLevelBPE.load(raw["tokenizer"]).encode(path.read_text(encoding="utf-8"))
    return Corpus(
        ids=np.array(ids, dtype=np.int32),
        manifest={"name": f"toy-{split}", "kind": "validation" if split == "val" else "training",
                  "sha256": sha256_file(path)},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/toy_pretrain.json")
    ap.add_argument("--variants", nargs="*",
                    choices=list(VARIANTS),
                    default=list(VARIANTS))
    ap.add_argument("--steps", type=int, default=None, help="override max_steps")
    ap.add_argument("--seed", type=int, default=None, help="override data seed")
    ap.add_argument("--out", default="experiments/ablations")
    args = ap.parse_args()

    raw = read_json(args.config)
    tr = deepcopy(raw["training"])
    if args.steps:
        tr["max_steps"] = args.steps
    seed = args.seed if args.seed is not None else raw.get("seed", 0)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    base_cfg = ModelConfig.from_dict(raw["model"])

    tr_corpus = build_corpus(raw, "train")
    val_corpus = build_corpus(raw, "val")

    index: list[dict] = []
    for name in args.variants:
        spec = VARIANTS[name]
        variant_cfg = deepcopy(base_cfg)
        variant_cfg = ModelConfig.from_dict({**variant_cfg.to_dict(), **spec["variant"]})
        baseline_cfg = ModelConfig.from_dict({**base_cfg.to_dict(), **spec["baseline"]})

        results = {}
        for label, cfg in (("baseline", baseline_cfg), ("variant", variant_cfg)):
            print(f"[ablate] {name}/{label} {spec['axis']}={getattr(cfg, spec['axis'])} "
                  f"params={LiteLM(cfg, seed=seed).num_params}")
            report = train(
                cfg,
                train_config=tr,
                train_corpus=tr_corpus,
                val_corpus=val_corpus,
                seed=seed,
                out_dir=str(out_root / name / label),
            )
            
            results[label] = {
                "final_val": report.final_val,
                "params": report.manifest["params"],
                "elapsed_s": report.elapsed_s,
                "config": {**cfg.to_dict()},
            }
        entry = {"name": name, "axis": spec["axis"], "results": results}
        index.append(entry)
        (out_root / name).with_suffix(".json").parent.mkdir(parents=True, exist_ok=True)
        (out_root / name / "summary.json").write_text(
            json.dumps(entry, indent=2, sort_keys=True)
        )

    (out_root / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True))
    print(f"[ablate] summary -> {out_root}/index.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run (and optionally re-run) toy pretraining.

Usage: python -m astra.training.train --config configs/toy_pretrain.json
Run twice with identical args to verify EX-04 reproducibility.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import numpy as np

from astra.model import LiteLM, ModelConfig
from astra.safety import leak_check
from astra.tokenizer import ByteLevelBPE
from astra.training import Corpus, train
from astra.utils import read_json, sha256_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/toy_pretrain.json")
    ap.add_argument("--seed", type=int, default=None, help="override data seed in config")
    ap.add_argument("--steps", type=int, default=None, help="override max_steps")
    ap.add_argument("--out", default=None, help="override out_dir")
    ap.add_argument("--experiments", default=None, help="override experiment_store dir")
    ap.add_argument("--resume", default=None, help="resume from checkpoint .npz")
    args = ap.parse_args()

    raw = read_json(args.config)
    cfg = ModelConfig.from_dict(raw["model"])
    tr = raw["training"]
    seed = args.seed if args.seed is not None else raw.get("seed", 0)
    if args.steps:
        tr = {**tr, "max_steps": args.steps}
    out_dir = args.out or raw["out_dir"]
    if args.experiments:
        tr = {**tr, "experiment_store": args.experiments}

    tok = ByteLevelBPE.load(raw["tokenizer"])
    cfg.vocab_size = len(tok)
    cfg = ModelConfig.from_dict(cfg.to_dict())  # re-validate after vocab change

    train_path = Path(raw["data"]["train"])
    val_path = Path(raw["data"]["val"])

    train_ids = tok.encode(train_path.read_text(encoding="utf-8"))
    val_ids = tok.encode(val_path.read_text(encoding="utf-8"))

    # safety gate: contamination check between train and val (docs/DATA.md § 3)
    leak = leak_check(train_ids, val_ids, n=13)
    print(f"[safety] leak_check(train, val, n=13) = {leak}")
    if not leak["leak_free"]:
        raise SystemExit(f"contamination detected: {leak}")

    train_corpus = Corpus(
        ids=np.array(train_ids, dtype=np.int32),
        manifest={"name": "toy-train", "kind": "training", "sha256": sha256_file(train_path)},
    )
    val_corpus = Corpus(
        ids=np.array(val_ids, dtype=np.int32),
        manifest={"name": "toy-val", "kind": "validation", "sha256": sha256_file(val_path)},
    )

    model = LiteLM(cfg, seed=seed)
    print(f"[model] params={model.num_params} model={cfg.name} vocab={cfg.vocab_size}")
    print(f"[data ] train_tokens={len(train_ids)} val_tokens={len(val_ids)}")

    report = train(
        cfg,
        train_config=tr,
        train_corpus=train_corpus,
        val_corpus=val_corpus,
        seed=seed,
        out_dir=out_dir,
        resume_from=args.resume,
    )
    print("[final val]", json.dumps(report.final_val, indent=2))
    print(f"run manifest -> {out_dir}/report.json")


if __name__ == "__main__":
    main()
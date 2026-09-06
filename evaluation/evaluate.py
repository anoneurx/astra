#!/usr/bin/env python3
"""Evaluate a checkpoint against the frozen evaluation/validation corpus.

Usage: python -m astra.evaluation.evaluate
    --checkpoint checkpoints/phase0/final.npz
    --config configs/toy_pretrain.json
Returns a JSON eval report tied to the checkpoint checksum.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import numpy as np

from astra.evaluation import evaluate_checkpoint
from astra.model import ModelConfig
from astra.tokenizer import ByteLevelBPE
from astra.training import Corpus
from astra.utils import read_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/toy_pretrain.json")
    ap.add_argument("--val-corpus", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    raw = read_json(args.config)
    val_path = args.val_corpus or raw["data"]["val"]

    tok = ByteLevelBPE.load(raw["tokenizer"])
    val_ids = tok.encode(Path(val_path).read_text(encoding="utf-8"))
    val_corpus = Corpus(ids=np.array(val_ids, dtype=np.int32),
                        manifest={"name": "toy-val", "kind": "validation"})

    cfg = ModelConfig.from_dict({**raw["model"], "vocab_size": len(tok)})

    out = args.out or (str(Path(args.checkpoint).parent) + "/eval_report.json")
    report = evaluate_checkpoint(args.checkpoint, tok, val_corpus, cfg, out_json=out)
    print(json.dumps(report["metrics"], indent=2))
    print(f"eval report -> {out}")


if __name__ == "__main__":
    main()
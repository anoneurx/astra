#!/usr/bin/env python3
"""Print the parameter count for a model config (docs/MODEL.md § 5).

Usage: python tools/param_count.py configs/toy_pretrain.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from astra.model import LiteLM, ModelConfig
from astra.utils import read_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--vocab", type=int, default=None, help="override vocab_size")
    args = ap.parse_args()
    raw = read_json(args.config)["model"]
    if args.vocab:
        raw = {**raw, "vocab_size": args.vocab}
    cfg = ModelConfig.from_dict(raw)
    m = LiteLM(cfg, seed=0)
    print(json.dumps({
        "model": cfg.name,
        "num_params": m.num_params,
        "params_million": round(m.num_params / 1e6, 4),
        "config": cfg.to_dict(),
    }, indent=2))


if __name__ == "__main__":
    main()
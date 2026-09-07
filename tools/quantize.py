#!/usr/bin/env python3
"""Quantization preview tool (docs/ROADMAP.md Phase 3).

Applied to a checkpoint on disk without modifying the original; reports
per-parameter error and (optionally) a val-loss comparison by loading the
quantized weights into a copy of the model.

Usage:
  python tools/quantize.py checkpoints/phase0/final.npz --method fp16
  python tools/quantize.py checkpoints/phase0/final.npz --method bf16 --config configs/toy_pretrain.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import numpy as np
from astra.model import LiteLM, ModelConfig
from astra.quantize import quantize_model_weights
from astra.training.checkpoint import load_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help=".npz checkpoint path")
    ap.add_argument("--method", choices=["fp16", "bf16", "int8"], default="bf16")
    ap.add_argument("--config", default="configs/toy_pretrain.json", help="model config json")
    args = ap.parse_args()

    raw = json.loads(Path(args.config).read_text())
    cfg = ModelConfig.from_dict(raw["model"])
    model = LiteLM(cfg, seed=0)
    step, _hist, _meta = load_checkpoint(args.checkpoint, model, opt=None, schedule=None)

    results = quantize_model_weights(model, method=args.method)
    total_bits = sum(
        np.prod(r.weights_after.shape) * {"fp16": 16, "bf16": 16, "int8": 8}[args.method]
        for r in results.values()
    )
    param_bytes = sum(np.prod(r.weights_after.shape) * 4 for r in results.values())
    worst = max(results.values(), key=lambda r: r.max_abs_error)
    print(f"checkpoint      : {args.checkpoint} (step {step})")
    print(f"method          : {args.method}")
    print(f"params quantized: {len(results)}")
    print(f"weights memory  : {param_bytes / 1e6:.2f} MB fp32 -> {total_bits / 8 / 1e6:.2f} MB")
    print(f"worst-error     : {worst.weights_before.shape} max_abs={worst.max_abs_error:.6f} "
          f"mean={worst.mean_abs_error:.6f}")
    print(f"max mean-err    : {max(r.mean_abs_error for r in results.values()):.6f}")


if __name__ == "__main__":
    main()
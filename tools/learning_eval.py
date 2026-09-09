#!/usr/bin/env python3
"""Candidate-vs-active evaluator (docs/LEARNING.md § 1.10-1.12).

Compares two checkpoints on a versioned learning-eval set: a *target* set that
measures gain on the newly-learned concept and a *regression* set that measures
whether prior capability is preserved.

Decision rule (deterministic, same seed for both models):
- Target must improve by >= ``--min-gain`` points.
- Regression must not drop by more than ``--max-regress`` points.

Usage:
    python tools/learning_eval.py --base checkpoints/base/final.npz \\
        --candidate checkpoints/candidate/final.npz --config configs/toy_name.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import numpy as np
from astra.model import LiteLM, ModelConfig
from astra.tokenizer import ByteLevelBPE
from astra.training.checkpoint import load_checkpoint
from astra.utils import read_json, write_json


def score_checkpoint(
    ckpt_path: str,
    tok: ByteLevelBPE,
    cfg: ModelConfig,
    eval_sets: dict,
    max_new: int = 24,
    temperature: float = 0.05,
    seed: int = 0,
) -> dict:
    """Run a checkpoint against eval sets; return per-set hit rates."""
    model = LiteLM(cfg, seed=0)
    load_checkpoint(ckpt_path, model, opt=None, schedule=None)
    results: dict = {}
    total_correct: dict = {}
    for set_name, spec in eval_sets.items():
        hits = 0
        details = []
        for i, item in enumerate(spec["items"]):
            rng = np.random.default_rng(seed * 1000 + i)
            from astra.evaluation.metrics import generate

            gen = generate(model, tok, tok.encode(item["prompt"]), max_new=max_new,
                           temperature=temperature, rng=rng)
            text = tok.decode(gen)
            ok = all(sub in text for sub in item["contains"])
            hits += int(ok)
            details.append({"prompt": item["prompt"], "contains": item["contains"],
                            "generated": text, "hit": ok})
        results[set_name] = {
            "n": len(spec["items"]),
            "hits": hits,
            "score": hits / max(1, len(spec["items"])),
            "details": details,
        }
        total_correct[set_name] = hits
    return {"by_set": results, "total": total_correct}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--config", default="configs/toy_name.json")
    ap.add_argument("--eval-set", default="datasets/learning/name_nova_qa.json")
    ap.add_argument("--min-gain", type=float, default=0.2, help="min target score delta to accept")
    ap.add_argument("--max-regress", type=float, default=0.0, help="allowed regression on other sets")
    ap.add_argument("--out", default=None, help="path to write comparison JSON")
    args = ap.parse_args()

    raw = read_json(args.config)
    cfg = ModelConfig.from_dict(raw["model"])
    tok = ByteLevelBPE.load(raw["tokenizer"])
    cfg.vocab_size = len(tok)
    cfg = ModelConfig.from_dict(cfg.to_dict())

    eval_spec = read_json(args.eval_set)
    if "sets" not in eval_spec:
        raise SystemExit("eval set must have a top-level 'sets' key")

    base = score_checkpoint(args.base, tok, cfg, eval_spec["sets"])
    cand = score_checkpoint(args.candidate, tok, cfg, eval_spec["sets"])

    decision = {"accepted": True, "gates": {}, "summary": ""}
    for set_name in eval_spec["sets"]:
        bs, cs = base["by_set"][set_name]["score"], cand["by_set"][set_name]["score"]
        delta = cs - bs
        if set_name.startswith("target"):
            gate = delta >= args.min_gain
            decision["gates"][set_name] = {
                "base": bs, "candidate": cs, "delta": delta, "gate": gate,
                "rule": f"gain >= {args.min_gain}",
            }
        else:
            gate = delta >= -args.max_regress
            decision["gates"][set_name] = {
                "base": bs, "candidate": cs, "delta": delta, "gate": gate,
                "rule": f"no more than {args.max_regress} regression",
            }
        decision["accepted"] = decision["accepted"] and gate

    decision["summary"] = (
        "ACCEPT" if decision["accepted"] else "REJECT"
    )
    if args.out:
        write_json(
            args.out,
            {
                "base_checkpoint": args.base,
                "candidate_checkpoint": args.candidate,
                "decision": decision,
                "base": base,
                "candidate": cand,
            },
        )
        print(f"comparison written -> {args.out}")
    print(json.dumps(decision["gates"], indent=2))
    print(f"DECISION: {decision['summary']}")


if __name__ == "__main__":
    main()
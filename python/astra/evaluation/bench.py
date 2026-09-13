"""Benchmark harness v1 (docs/ROADMAP.md Phase 3, docs/BENCHMARKS.md).

Loads a suite manifest (JSON) describing benchmark items, runs each item's
scorer against a checkpoint, and compares against thresholds. Produces a
versioned JSON report under benchmarks/results/ and returns a pass/fail gate.

Item schema (docs/BENCHMARKS.md § 3):
    {"id": ..., "type": "val_ppl" | "decode" | "reason_exact", "prompt"/"target",
     "scorer": ..., "expected_metric": ..., "threshold": {...}}
"""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from astra.model import LiteLM
from astra.training.checkpoint import load_checkpoint
from astra.training.data import Corpus, SeqStream
from astra.utils import sha256_file, write_json


@dataclass
class ItemResult:
    id: str
    metric: float
    ok: bool
    detail: dict = field(default_factory=dict)


def run_val_ppl(model: LiteLM, val_corpus: Corpus, seed: int = 0) -> float:
    """Mean cross-entropy on the frozen val corpus (same as training eval)."""
    stream = SeqStream(val_corpus, batch_seq=4, seq_len=model.cfg.max_seq_len,
                       rng=np.random.default_rng(seed))
    total, n = 0.0, 0
    for x, y in stream:
        _logits, loss = model.forward_loss(x, y)
        total += loss * (x.shape[0] * x.shape[1])
        n += x.shape[0] * x.shape[1]
    return float(total / max(1, n))


def run_inference_tokens_per_sec(model: LiteLM, n_tokens: int = 200, seed: int = 0) -> float:
    """Autoregressive decode throughput (single-stream, reference sampler)."""
    from astra.evaluation.metrics import generate

    rng = np.random.default_rng(seed)
    seed_ids = [1, 2, 3]
    t0 = time.time()
    generate(model, None, seed_ids, max_new=n_tokens, rng=rng)
    dt = time.time() - t0
    return n_tokens / max(1e-6, dt)


def run_repetition_fraction(model: LiteLM, n_tokens: int = 200, seed: int = 0) -> float:
    from astra.evaluation.metrics import generate, repetition_fraction

    rng = np.random.default_rng(seed)
    toks = generate(model, None, [1, 2, 3], max_new=n_tokens, rng=rng)
    return float(repetition_fraction(toks, n=4))


SCORERS: dict[str, Callable[..., float]] = {
    "val_ppl": run_val_ppl,
    "decode_tokens_per_sec": run_inference_tokens_per_sec,
    "repetition_fraction": run_repetition_fraction,
}


def _scorer_deps(scorer) -> set[str]:
    import inspect

    return set(inspect.signature(scorer).parameters) - {"model"}


def _satisfies(metric: float, threshold: dict) -> bool:
    """Threshold schema: {"op": "<="|">="|"<"|">", "value": float}."""
    op = threshold["op"]
    value = threshold["value"]
    if op == "<=":
        return metric <= value
    if op == "<":
        return metric < value
    if op == ">=":
        return metric >= value
    if op == ">":
        return metric > value
    raise ValueError(f"unknown op {op!r}")


def run_suite(
    model_cfg,
    ckpt_path: str,
    val_corpus: Corpus | None,
    suite_path: str,
    out_json: str,
    extra_ctx: dict | None = None,
) -> tuple[dict, bool]:
    """Run one suite manifest; return (report dict, all_pass bool)."""
    suite = json.loads(Path(suite_path).read_text())
    model = LiteLM(model_cfg, seed=0)
    step, _hist, _meta = load_checkpoint(ckpt_path, model, opt=None, schedule=None)
    ctx = {
        "model": model,
        "val_corpus": val_corpus,
        "ckpt_step": step,
        **(extra_ctx or {}),
    }

    results: list[ItemResult] = []
    for item in suite["items"]:
        scorer = SCORERS[item["scorer"]]
        t0 = time.time()
        args = dict(item.get("args", {}) or {})
        if "val_corpus" in _scorer_deps(scorer):
            args["val_corpus"] = ctx["val_corpus"]
        metric = scorer(ctx["model"], **args)
        dt = time.time() - t0
        threshold = item.get("threshold")
        ok = _satisfies(metric, threshold) if threshold else False
        results.append(ItemResult(id=item["id"], metric=metric, ok=ok,
                                  detail={"elapsed_s": round(dt, 3)}))

    report = {
        "suite": suite["suite"],
        "suite_version": suite.get("version", 1),
        "suite_status": suite.get("status", "ADVISORY"),
        "checkpoint": ckpt_path,
        "checkpoint_sha256": sha256_file(ckpt_path),
        "model_step": step,
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "items": [
            {
                "id": r.id,
                "metric": r.metric,
                "threshold_met": r.ok,
                "threshold": next((i.get("threshold") for i in suite["items"] if i["id"] == r.id), None),
                "detail": r.detail,
            }
            for r in results
        ],
        "passed": all(r.ok for r in results),
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    write_json(out_json, report)
    return report, report["passed"]


def main() -> None:  # pragma: no cover - thin CLI
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))

    from astra.model.config import ModelConfig
    from astra.utils import read_json

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/toy_pretrain.json")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw = read_json(args.config)
    cfg = ModelConfig.from_dict(raw["model"])
    report, passed = run_suite(cfg, args.checkpoint, None, args.suite, args.out)
    print(json.dumps({i["id"]: (i["metric"], i["threshold_met"]) for i in report["items"]}, indent=2))
    print("PASS" if passed else "FAIL")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
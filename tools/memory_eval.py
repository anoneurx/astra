#!/usr/bin/env python3
"""Memory engine retrieval-quality evaluation (docs/BENCHMARKS.md § 2.5).

Populates an in-memory store with a synthetic fact corpus, runs retrieval on
gold queries, and scores hit@k, MRR, and nDCG@5. Null thresholds until the
Astra 0.5 gate adopts them (docs/MEMORY.md § 7 status is research).

By default the corpus comes from datasets/memory/qa_v1.json; the embedded
FACTS/QUERIES are a fallback when the manifest is absent.

Usage:
    python tools/memory_eval.py
    python tools/memory_eval.py --config configs/toy_name.json --checkpoint checkpoints/name/resumed/final.npz
    python tools/memory_eval.py --out benchmarks/results/memory/eval1.json --seed 3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from astra.memory import HashEmbedder, MemoryRecord, MemoryStore
from astra.utils import environment, git_commit

# Adopted thresholds (benchmarks/suites/core-retrieval.json v2, GAP-5).
THRESHOLDS = {"hit_at_1": 0.8, "hit_at_5": 1.0, "mrr": 0.9, "ndcg_at_5": 0.9}

FACTS = [
    ("capital_france", "Paris is the capital of France."),
    ("capital_spain", "Madrid is the capital of Spain."),
    ("einstein", "Albert Einstein developed the theory of relativity."),
    ("h2o", "Water is composed of hydrogen and oxygen."),
    ("moon", "The Moon is Earth's only natural satellite."),
    ("marseille", "Marseille is a port city in southern France."),
    ("quark", "Quarks are fundamental constituents of matter."),
    ("gaucho", "A gaucho is a skilled cowboy of the South American pampas."),
    ("tardis", "The TARDIS is a fictional time machine in Doctor Who."),
    ("photosynthesis", "Photosynthesis converts light into chemical energy."),
]

QUERIES = [
    ("what is the capital city of France", ["capital_france"]),
    ("capital of spain", ["capital_spain"]),
    ("who developed relativity", ["einstein"]),
    ("what is water made of", ["h2o"]),
    ("moon satellite of earth", ["moon"]),
    ("southern french port city", ["marseille"]),
    ("what are quarks", ["quark"]),
    ("south american cowboy", ["gaucho"]),
    ("doctor who time machine", ["tardis"]),
    ("photosynthesis energy from light", ["photosynthesis"]),
    ("capital of france seaside", ["capital_france", "marseille"]),
]


def load_corpus(manifest: str | None) -> tuple[list[tuple[str, str]], list[tuple[str, list[str]]]]:
    """Load (facts, queries) from the manifest if present, else the fallback."""
    if manifest and Path(manifest).exists():
        data = json.loads(Path(manifest).read_text())
        facts = [(f["id"], f["content"]) for f in data["facts"]]
        queries = [(it["question"], it["gold_fact_ids"]) for it in data["items"]]
        return facts, queries
    return FACTS, QUERIES


def hit_at_k(ranked_ids: list[str], gold: set[str], k: int) -> float:
    return 1.0 if any(g in gold for g in ranked_ids[:k]) else 0.0


def reciprocal_rank(ranked_ids: list[str], gold: set[str]) -> float:
    for i, rid in enumerate(ranked_ids, start=1):
        if rid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_ids: list[str], gold: set[str], k: int) -> float:
    rel = [1.0 if rid in gold else 0.0 for rid in ranked_ids[:k]]
    if not rel:
        return 0.0
    dcg = sum(r / math.log2(i + 1) for i, r in enumerate(rel, start=1))
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--checkpoint")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--budget-tokens", type=int, default=128)
    ap.add_argument("--manifest", default="datasets/memory/qa_v1.json")
    ap.add_argument("--enforce", action="store_true",
                    help="apply adopted thresholds; exit 1 on any failure")
    args = ap.parse_args()

    facts, queries = load_corpus(args.manifest)

    if args.checkpoint:
        from astra.memory import LiteLMExtractor

        print(f"embedding with LiteLMExtractor: {args.config} / {args.checkpoint}")
        embedder = LiteLMExtractor.from_config(args.config, args.checkpoint)
    else:
        embedder = HashEmbedder(seed=args.seed)

    store = MemoryStore(name=f"eval-{args.seed}", directory=tempfile.mkdtemp(prefix="astra-mem-eval-"),
                        embedder=embedder)
    store.meta["purpose"] = "retrieval-quality evaluation"
    topics: dict[str, str] = {}
    for topic, text in facts:
        rec = store.add(MemoryRecord(content=text, kind="fact", source="auto_extract",
                                     tags=[topic], verification_status="verified", confidence=0.95))
        topics[topic] = rec.id

    print(f"indexed {len(facts)} facts")
    overall = {"hit_at_1": 0.0, "hit_at_5": 0.0, "mrr": 0.0, "ndcg_at_5": 0.0, "n": 0}
    per_query = []
    for text, gold_topics in queries:
        gold = {topics[t] for t in gold_topics}
        hits = store.search(query=text, k=args.k, budget_tokens=args.budget_tokens)
        ranked = [h.record.id for h in hits]
        ndcg = ndcg_at_k(ranked, gold, args.k)
        overall["hit_at_1"] += hit_at_k(ranked, gold, 1)
        overall["hit_at_5"] += hit_at_k(ranked, gold, args.k)
        overall["mrr"] += reciprocal_rank(ranked, gold)
        overall["ndcg_at_5"] += ndcg
        overall["n"] += 1
        per_query.append({
            "query": text, "gold": sorted(gold),
            "hit_at_1": hit_at_k(ranked, gold, 1),
            "hit_at_5": hit_at_k(ranked, gold, args.k),
            "mrr": reciprocal_rank(ranked, gold),
            "ndcg_at_5": ndcg,
            "retrieved": ranked[:3],
        })

    denom = max(overall["n"], 1)
    metrics = {k: round(v / denom, 4) for k, v in overall.items() if k != "n"}

    # Verdict vs adopted thresholds (HashEmbedder only is gated; the
    # LiteLMExtractor-on-toy encoder is a documented non-gated baseline).
    gated = embedder.config.get("name") == "HashEmbedder"
    verdict = {m: {"metric": metrics[m], "threshold": THRESHOLDS[m],
                   "pass": metrics[m] >= THRESHOLDS[m]} for m in THRESHOLDS}
    overall_pass = all(v["pass"] for v in verdict.values())
    report = {
        "task": "core-retrieval (engine-level, ADVISORY)",
        "commit": git_commit(),
        "environment": environment(),
        "embedder": embedder.config,
        "params": {"k": args.k, "budget_tokens": args.budget_tokens, "seed": args.seed},
        "corpus": {"facts": len(facts), "queries": len(queries), "manifest": str(manifest) if (manifest := args.manifest) else None},
        "records_indexed": len(topics),
        "thresholds": {"adopted": THRESHOLDS if gated else None,
                       "gated": gated,
                       "pass": overall_pass},
        "verdict": verdict,
        "metrics": metrics,
        "per_query": per_query,
    }
    print(json.dumps(metrics, indent=2))
    if gated:
        print(f"thresholds {'PASS' if overall_pass else 'FAIL'}: "
              + ", ".join(f"{m} {v['metric']} >={v['threshold']}" for m, v in verdict.items()))

    out = args.out
    if out is None:
        out = "benchmarks/results/memory/core-retrieval.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"report -> {out}")

    if args.enforce and gated and not overall_pass:
        print("FAILED bench thresholds (see verdict)")
        sys.exit(1)


if __name__ == "__main__":
    main()
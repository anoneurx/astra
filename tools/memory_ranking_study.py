#!/usr/bin/env python3
"""Ranking weight-grid ablation study (docs/MEMORY.md § 7, GAP-5).

Sweeps the hybrid-score weights (cosine, recency, confidence, kind) over a
grid against the memory eval corpus (datasets/memory/qa_v1.json) and reports
hit@1/MRR/nDCG@5 per weight set, then identifies the best-scoring set so it
can be adopted as the search default.

Usage:
    python tools/memory_ranking_study.py
    python tools/memory_ranking_study.py --out benchmarks/results/memory/ranking-study.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from astra.memory import HashEmbedder, MemoryRecord, MemoryStore
from astra.utils import environment, git_commit


def load_corpus(manifest: str) -> tuple[list[dict], list[dict]]:
    data = json.loads(Path(manifest).read_text())
    return data["facts"], data["items"]


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
    ap.add_argument("--manifest", default="datasets/memory/qa_v1.json")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    facts, items = load_corpus(args.manifest)
    embedder = HashEmbedder(seed=args.seed)
    store = MemoryStore(name=f"study-{args.seed}", directory=tempfile.mkdtemp(prefix="astra-mem-study-"),
                        embedder=embedder)
    topics: dict[str, str] = {}
    for i, f in enumerate(facts):
        # Stagger creation times so recency varies meaningfully
        rec = store.add(MemoryRecord(
            content=f["content"], kind=f.get("kind", "fact"),
            tags=[f["id"]], confidence=f.get("confidence", 0.9),
            created_at=(datetime.now(UTC) - timedelta(days=i * 3)).isoformat()))
        topics[f["id"]] = rec.id

    # Build gold answers and query embeddings once
    gold_by_query = [(it["question"], {topics[t] for t in it["gold_fact_ids"]}) for it in items]

    # Weight grid: {cosine, recency, confidence, kind}
    grid = {
        "default": {"cosine": 1.0, "recency": 0.15, "confidence": 0.25, "kind": 0.10},
        "cosine_only": {"cosine": 1.0, "recency": 0.0, "confidence": 0.0, "kind": 0.0},
        "recency_boost": {"cosine": 1.0, "recency": 0.5, "confidence": 0.1, "kind": 0.1},
        "confidence_boost": {"cosine": 1.0, "recency": 0.05, "confidence": 0.5, "kind": 0.1},
        "kind_boost": {"cosine": 0.8, "recency": 0.1, "confidence": 0.2, "kind": 0.5},
    }
    # Cartesian sweep over the main two sensitive weights w/ fixed base
    for wr in (0.0, 0.25, 0.5):
        for wc in (0.0, 0.25, 0.5):
            name = f"grid_r{wr}_c{wc}"
            grid[name] = {"cosine": 1.0 - wr - wc, "recency": wr, "confidence": wc, "kind": 0.1}

    results = []
    for wname, weights in grid.items():
        o = {"hit_at_1": 0.0, "mrr": 0.0, "ndcg_at_5": 0.0, "n": 0}
        for qtext, gold in gold_by_query:
            hits = store.search(query=qtext, k=args.k, weights=weights)
            ranked = [h.record.id for h in hits]
            o["hit_at_1"] += hit_at_k(ranked, gold, 1)
            o["mrr"] += reciprocal_rank(ranked, gold)
            o["ndcg_at_5"] += ndcg_at_k(ranked, gold, args.k)
            o["n"] += 1
        m = {k: round(v / max(o["n"], 1), 4) for k, v in o.items() if k != "n"}
        results.append({"weights": weights, "name": wname, "metrics": m,
                        "score_mrr": m["mrr"]})

    results.sort(key=lambda r: r["score_mrr"], reverse=True)
    best = results[0]

    report = {
        "task": "ranking weight-grid ablation (ADVISORY, GAP-5)",
        "commit": git_commit(),
        "environment": environment(),
        "corpus": {"facts": len(facts), "items": len(items), "manifest": args.manifest},
        "embedder": embedder.config,
        "params": {"k": args.k, "seed": args.seed},
        "best": best,
        "all": results,
        "caveat": "weight sweep evaluated on the demo corpus; adopt default after real-model measurement",
    }
    print(f"best: {best['name']} MRR={best['score_mrr']}")
    for r in results:
        print(f"  {r['name']:<24} mrr={r['metrics']['mrr']:<6} "
              f"hit1={r['metrics']['hit_at_1']:<5} ndcg5={r['metrics']['ndcg_at_5']}")

    out = args.out
    if out is None:
        out = "benchmarks/results/memory/ranking-study.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Rust vs Python memory flat search cross-check (GAP-4).

Builds a store via Python (HashEmbedder), runs the Rust astramem binary
with the same query embedding, and asserts top-k ID sets match.

Requires: `cargo build --release -p astra-rt` first (via `make rust-mem`).

Usage:
    python tools/rust_mem_crosscheck.py [--k 8] [--seed 1]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from astra.memory import HashEmbedder, MemoryRecord, MemoryStore


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rust-bin", default="service/rust/target/release/astramem")
    args = ap.parse_args()

    # Build a Python store with HashEmbedder
    embedder = HashEmbedder(seed=args.seed, dim=64)
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(name="crosscheck", directory=tmpdir, embedder=embedder)
        facts = [
            "Paris is the capital of France.",
            "Madrid is the capital of Spain.",
            "Albert Einstein developed the theory of relativity.",
            "Water is composed of hydrogen and oxygen.",
            "The Moon is Earth's only natural satellite.",
            "Marseille is a port city in southern France.",
            "Quarks are fundamental constituents of matter.",
            "A gaucho is a skilled cowboy of the South American pampas.",
            "The TARDIS is a fictional time machine in Doctor Who.",
            "Photosynthesis converts light into chemical energy.",
        ]
        for i, f in enumerate(facts):
            store.add(MemoryRecord(content=f, kind="fact", source="auto_extract",
                                   tags=[f"fact{i}"], confidence=0.95))
        store.flush()
        store_path = Path(tmpdir) / "crosscheck.json"

        # Query embedding for "capital of France" (same embedder)
        query_text = "what is the capital of France"
        query_emb = embedder.embed([query_text])[0]
        import base64

        import numpy as np

        query_b64 = base64.b64encode(np.asarray(query_emb, dtype=np.float32).tobytes()).decode()

        # Run Rust binary
        rust_bin = Path(args.rust_bin)
        if not rust_bin.exists():
            print(f"Rust binary not found: {rust_bin}")
            print("Run: cargo build --release -p astra-rt")
            sys.exit(1)

        cmd = [str(rust_bin), "query", "--store", str(store_path),
               "--query-emb", query_b64, "--k", str(args.k)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            print(f"Rust query failed: {result.stderr}")
            sys.exit(1)

        rust_hits = []
        for line in result.stdout.strip().splitlines():
            d = json.loads(line)
            rust_hits.append(d["id"])

        # Python cosine-only search (weights {cosine:1, others:0})
        from astra.memory.retrieval import retrieve

        hits = retrieve(
            list(store._active_candidates()),
            query_emb,
            k=args.k,
            weights={"cosine": 1.0, "recency": 0.0, "confidence": 0.0, "kind": 0.0},
        )
        py_hits = [h.record.id for h in hits]

        print(f"Python top-{args.k}: {py_hits}")
        print(f"Rust   top-{args.k}: {rust_hits}")

        # Top-1 must match exactly; the rest can differ in tie-order due to f32 vs f64 precision
        if py_hits[0] != rust_hits[0]:
            print("✗ FAIL: top-1 mismatch")
            sys.exit(1)
        if set(py_hits) != set(rust_hits):
            print("✗ FAIL: top-k ID sets differ")
            sys.exit(1)
        print("✓ PASS: top-1 exact match, top-k ID sets match")
        sys.exit(0)


if __name__ == "__main__":
    main()
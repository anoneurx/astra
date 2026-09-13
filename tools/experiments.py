#!/usr/bin/env python3
"""Query the experiment store (docs/ROADMAP.md Phase 2).

Usage: python tools/experiments.py [--root experiments/runs] \
         list | show <run_id> | query [field=value ...]

Filters accept dot-paths and comparison suffixes:
  python tools/experiments.py query seed=42
  python tools/experiments.py query final_val.loss__lt=3.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from astra.experiments import ExperimentStore


def _parse_filter(spec: str) -> tuple[str, object]:
    key, _, value = spec.partition("=")
    if not value:
        raise SystemExit(f"invalid filter (want field=value): {spec}")
    try:
        parsed: object = json.loads(value)
    except json.JSONDecodeError:
        parsed = value
    return key, parsed


def main() -> None:
    ap = argparse.ArgumentParser(description="Query the Astra experiment store")
    ap.add_argument("--root", default="experiments/runs")
    ap.add_argument("command", choices=["list", "show", "query"])
    ap.add_argument("args", nargs="*")
    args = ap.parse_args()

    store = ExperimentStore(root=args.root)

    if args.command == "list":
        runs = store.list_runs()
        if not runs:
            print("(no runs recorded)")
            return
        for r in runs:
            fv = r.get("final_val", {})
            print(
                f"{r['run_id']}  "
                f"seed={r.get('seed')}  "
                f"val_loss={fv.get('loss', float('nan')):.4f}  "
                f"ppl={fv.get('ppl', float('nan')):.2f}  "
                f"{r.get('git_commit', '?')}"
            )
        return

    if args.command == "show":
        if not args.args:
            raise SystemExit("usage: experiments.py show <run_id>")
        run_id = args.args[0]
        if not store.has(run_id):
            raise SystemExit(f"run not found: {run_id}")
        print(json.dumps(store.get(run_id), indent=2, sort_keys=True))
        return

    if args.command == "query":
        filters = dict(_parse_filter(spec) for spec in args.args)
        hits = store.query(**filters)
        print(
            json.dumps(
                [{"run_id": r["run_id"], "seed": r.get("seed"),
                  "final_val": r.get("final_val"), "git_commit": r.get("git_commit")}
                 for r in hits],
                indent=2, sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
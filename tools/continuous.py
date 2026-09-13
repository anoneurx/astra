#!/usr/bin/env python3
"""Phase-8 continuous-evolution operator (docs/ROADMAP.md § Phase 8).

Drives the supervised Phase-6/7 loop on a schedule, records each accept's
metrics into the long-term ``MetricsStore``, then runs the ``DriftDetector``
over the promotion series and acts on the result:

  - ACCEPT + improving / stable      -> continue (monotonic non-regression)
  - ACCEPT + regressing or drift     -> alert; ``--auto-rollback`` rolls back
       to the previous active (rollback always available) and audits it, or
       stops for operator review (supervised default)
  - ``--check``                      -> drift check only, no iteration

Usage:
  python tools/continuous.py --config configs/toy_name.json
  python tools/continuous.py --check
  python tools/continuous.py --watch 300 --max-runs 4
  python tools/continuous.py --check --auto-rollback
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from astra.learning.audit import AuditEntry, AuditLog
from astra.learning.evolution import DriftDetector, MetricsStore, PromotionSnapshot
from astra.registry import ModelRegistry


def _iteration_args(args) -> list[str]:
    return [
        "--config", args.config,
        "--base", args.base,
        "--out-dir", args.out_dir,
        "--store", args.store,
        "--memory-dir", args.memory_dir,
        "--audit", args.audit,
        "--registry", args.registry,
        "--semver", args.semver,
        "--policy", args.policy,
        "--steps", str(args.steps),
        "--seed", str(args.seed),
        "--replay-ratio", str(args.replay_ratio),
        "--peak-lr", str(args.peak_lr),
    ]


def _run_iteration(args) -> dict:
    """Run one supervised loop iteration and read its report."""
    run = subprocess.run(
        [sys.executable, "tools/self_improve.py"] + _iteration_args(args),
        check=False,
    )
    if run.returncode != 0:
        raise SystemExit(f"iteration failed (exit {run.returncode}); not recorded")
    report_path = Path(args.out_dir) / "self_improve_report.json"
    if not report_path.exists():
        raise SystemExit(f"no report produced at {report_path}")
    import json

    return json.loads(report_path.read_text(encoding="utf-8"))


def _record(report: dict, args) -> None:
    if not report.get("promoted"):
        return
    store = MetricsStore(root=args.metrics_dir)
    store.record(PromotionSnapshot(
        name=args.name,
        sha256=report["promoted_sha"],
        semver=args.semver,
        metrics=report.get("metrics", {}),
        report_id=f"{args.out_dir}/self_improve_report.json",
    ))


def check_drift(args, store: MetricsStore | None = None) -> list:
    store = store or MetricsStore(root=args.metrics_dir)
    detector = DriftDetector(
        regression_threshold=args.regression_threshold,
        drift_tolerance=args.drift_tolerance,
    )
    report = {"name": args.name, "metrics": {}, "drift": False, "regressing": False}
    reports = []
    for metric in store.metric_names(args.name):
        dr = detector.check(store.series(args.name), metric)
        reports.append(dr)
        report["metrics"][metric] = dr.to_dict()
        report["drift"] = report["drift"] or dr.drift
        report["regressing"] = report["regressing"] or dr.trend == "regressing"
        print(f"[drift  ] {metric:28s} {dr.trend:14s} drift={dr.drift}  {dr.message}")
    return reports, report


def maybe_rollback(report: dict, args, reason: str) -> bool:
    """Roll back to the previous active when the gate series has drifted."""
    registry = ModelRegistry(path=args.registry)
    current = registry.current(args.name)
    if current is None:
        return False
    restored = registry.rollback(args.name, notes=f"continuous-evolution: {reason}")
    if restored is None:
        return False
    AuditLog(path=args.audit).append(AuditEntry(
        event="rollback",
        model_name=args.name,
        candidate_sha=current.sha256,
        decision={"drift_detector": True, "reason": reason},
        notes=f"continuous evolution rolled back to {restored.sha256[:12]}",
    ))
    print(f"[rollback] restored {restored.sha256[:12]} ({reason})")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/toy_name.json")
    ap.add_argument("--name", default="astra-name")
    ap.add_argument("--base", default="checkpoints/name/resumed/final.npz")
    ap.add_argument("--out-dir", default="checkpoints/candidate")
    ap.add_argument("--store", default="learning/store")
    ap.add_argument("--memory-dir", default="memory/store")
    ap.add_argument("--audit", default="learning/audit.jsonl")
    ap.add_argument("--registry", default="checkpoints/registry.json")
    ap.add_argument("--metrics-dir", default="learning/metrics")
    ap.add_argument("--semver", default="0.9.0")
    ap.add_argument("--policy", default="auto", choices=["auto", "manual"])
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replay-ratio", type=float, default=0.5)
    ap.add_argument("--peak-lr", type=float, default=1e-4)
    ap.add_argument("--watch", type=int, default=0, metavar="SECONDS",
                    help="repeat iterations, sleeping SECONDS in between (0 = once)")
    ap.add_argument("--max-runs", type=int, default=0, help="cap iterations in --watch mode")
    ap.add_argument("--regression-threshold", type=float, default=0.1)
    ap.add_argument("--drift-tolerance", type=float, default=0.2)
    ap.add_argument("--check", action="store_true", help="drift check only (no iteration)")
    ap.add_argument("--auto-rollback", action="store_true",
                    help="roll back automatically on regressing/drift (supervised default: alert only)")
    ap.add_argument("--compact-memory", action="store_true",
                    help="run MemoryStore.compact() when drift is detected (durable memory growth)")
    args = ap.parse_args()

    if args.check:
        _, report = check_drift(args)
        if report["drift"]:
            print(f"[continuous] DRIFT detected under {args.name}: operator review needed "
                  f"(regressing={report['regressing']})")
        else:
            print(f"[continuous] no drift under {args.name}: promotion series is clean")
        return

    runs = 0
    while True:
        runs += 1
        print(f"[continuous] iteration {runs}")
        report = _run_iteration(args)
        print(f"[continuous] {'promoted ' + report['promoted_sha'][:12] if report.get('promoted') else 'rejected'}")
        _record(report, args)
        _, drift = check_drift(args)
        if drift["drift"]:
            print(f"[continuous] drift under {args.name}: {drift['regressing'] and 'regressing' or 'silent drift'}")
            if args.compact_memory:
                from astra.memory import MemoryStore

                stats = MemoryStore(name="longterm", directory=args.memory_dir).compact()
                print(f"[continuous] memory compaction: {stats}")
            if args.auto_rollback and report.get("promoted"):
                maybe_rollback(drift, args, "post-promotion drift detected by continuous evolution")
            else:
                print("[continuous] supervised: stopping for operator review")
                return
        if args.watch <= 0:
            return
        if args.max_runs and runs >= args.max_runs:
            print(f"[continuous] max-runs ({args.max_runs}) reached")
            return
        print(f"[continuous] sleeping {args.watch}s")
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
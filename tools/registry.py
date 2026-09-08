#!/usr/bin/env python3
"""Model registry CLI (docs/VERSIONING.md § 5).

Usage:
    python tools/registry.py register --name astra-name --semver 0.4.0 \
        --checkpoint checkpoints/name/resumed/final.npz --config configs/toy_name.json \
        --data datasets/name/manifest.json --eval-report docs/releases/v0.4.0.md
    python tools/registry.py list
    python tools/registry.py show astra-name
    python tools/registry.py verify astra-name
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from astra.registry import ModelRegistry
from astra.utils import sha256_file


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="Astra model registry v1")
    sub = ap.add_subparsers(dest="cmd", required=True)

    reg = sub.add_parser("register", help="register a checkpoint as an artifact")
    reg.add_argument("--name", required=True)
    reg.add_argument("--semver", required=True)
    reg.add_argument("--checkpoint", required=True)
    reg.add_argument("--config", required=True)
    reg.add_argument("--data-manifest", default="")
    reg.add_argument("--eval-report", default="")
    reg.add_argument("--checklist", default="")
    reg.add_argument("--notes", default="")
    reg.add_argument("--force", action="store_true", help="override an existing sha (audit action)")
    reg.add_argument("--registry", default="checkpoints/registry.json")

    lst = sub.add_parser("list", help="list registered names")
    lst.add_argument("--registry", default="checkpoints/registry.json")

    show = sub.add_parser("show", help="show the latest entry for a name")
    show.add_argument("name")
    show.add_argument("--registry", default="checkpoints/registry.json")

    ver = sub.add_parser("verify", help="recompute a checkpoint sha and compare")
    ver.add_argument("name")
    ver.add_argument("--registry", default="checkpoints/registry.json")

    args = ap.parse_args()
    reg = ModelRegistry(args.registry)

    if args.cmd == "register":
        cfg = _load(args.config)
        cid = cfg.get("config_id") or f"{args.config}:{sha256_file(args.config)[:12]}"
        ck = args.checkpoint
        manifest = ck.rsplit(".npz", 1)[0] + ".manifest.json"
        params = step = 0
        if Path(manifest).exists():
            m = _load(manifest)
            params, step = m.get("params", 0), m.get("step", 0)
        rec = reg.register(
            name=args.name,
            semver=args.semver,
            path=args.checkpoint,
            config_id=cid,
            data_manifest_id=args.data_manifest,
            checklist_id=args.checklist,
            eval_report_id=args.eval_report,
            params=params,
            step=step,
            notes=args.notes,
            force=args.force,
        )
        reg.save()
        print(f"registered {rec.sha256[:12]} {args.name} {args.semver} @ {args.checkpoint}")
    elif args.cmd == "list":
        if not reg.entries:
            print("(empty registry)")
            return
        for name in reg.list_names():
            rec = reg.get(name)
            _rec, ok = reg.verify(name)
            print(f"{'OK' if ok else 'MISMATCH'}  {name:12} {rec.semver:8} {rec.sha256[:12]}  {rec.path}")
    elif args.cmd == "show":
        rec = reg.get(args.name)
        if rec is None:
            print(f"no registered artifact named {args.name!r}")
            sys.exit(1)
        print(json.dumps(vars(rec), indent=2, sort_keys=True))
    elif args.cmd == "verify":
        rec, ok = reg.verify(args.name)
        if rec is None:
            print(f"no registered artifact named {args.name!r}")
            sys.exit(1)
        print(f"{'MATCH' if ok else 'MISMATCH'}  sha256 {rec.sha256}")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
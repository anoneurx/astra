#!/usr/bin/env python3
"""Memory engine CLI (docs/MEMORY.md).

Subcommands: init, add, get, query, list, correct, delete, expire, conflicts,
resolve, audit.

Examples:
    python tools/memory.py init --name longterm
    python tools/memory.py add --content "Paris is the capital of France." --kind fact
    python tools/memory.py query --text "what is the capital of France" -k 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))


def _embedder(args: argparse.Namespace):
    from astra.memory.embedder import HashEmbedder, LiteLMExtractor

    if getattr(args, "checkpoint", None):
        return LiteLMExtractor.from_config(args.config, args.checkpoint)
    return HashEmbedder(seed=getattr(args, "seed", 1))


def _store(args: argparse.Namespace):
    from astra.memory import MemoryStore

    return MemoryStore.open(name=args.name, embedder=_embedder(args))


def main() -> None:
    ap = argparse.ArgumentParser(description="Astra memory engine")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init")
    p.add_argument("--name", default="longterm")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--config")
    p.add_argument("--checkpoint")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("add")
    p.add_argument("--name", default="longterm")
    p.add_argument("--content", required=True)
    p.add_argument("--kind", default="fact", choices=["fact", "episode", "semantic", "session"])
    p.add_argument("--source", default="auto_extract", choices=["user_said", "verified_correction", "auto_extract", "model_generated"])
    p.add_argument("--confidence", type=float, default=0.5)
    p.add_argument("--tags", nargs="*", default=[])
    p.add_argument("--id")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--config")
    p.add_argument("--checkpoint")

    p = sub.add_parser("get")
    p.add_argument("--name", default="longterm")
    p.add_argument("--id", required=True)
    p.add_argument("--all", action="store_true")

    p = sub.add_parser("list")
    p.add_argument("--name", default="longterm")
    p.add_argument("--kind", nargs="*", default=[])
    p.add_argument("--deleted", action="store_true")

    p = sub.add_parser("query")
    p.add_argument("--name", default="longterm")
    p.add_argument("--text", required=True)
    p.add_argument("-k", type=int, default=8)
    p.add_argument("--budget-tokens", type=int)
    p.add_argument("--disputed", action="store_true")
    p.add_argument("--quarantine", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--config")
    p.add_argument("--checkpoint")

    p = sub.add_parser("correct")
    p.add_argument("--name", default="longterm")
    p.add_argument("--id", required=True)
    p.add_argument("--content")
    p.add_argument("--confidence", type=float)

    p = sub.add_parser("delete")
    p.add_argument("--name", default="longterm")
    p.add_argument("--id", required=True)
    p.add_argument("--purge", action="store_true")

    p = sub.add_parser("expire")
    p.add_argument("--name", default="longterm")

    p = sub.add_parser("conflicts")
    p.add_argument("--name", default="longterm")

    p = sub.add_parser("resolve")
    p.add_argument("--name", default="longterm")
    p.add_argument("--id", required=True)
    p.add_argument("--by", default="human")

    p = sub.add_parser("audit")
    p.add_argument("--name", default="longterm")
    p.add_argument("--tail", type=int, default=20)

    p = sub.add_parser("remember")
    p.add_argument("--name", default="longterm")
    p.add_argument("--session", required=True)
    p.add_argument("--content", required=True)
    p.add_argument("--reply")
    p.add_argument("--ttl-hours", type=float, default=24.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--config")
    p.add_argument("--checkpoint")

    p = sub.add_parser("session-recall")
    p.add_argument("--name", default="longterm")
    p.add_argument("--session", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--ttl-hours", type=float, default=24.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--config")
    p.add_argument("--checkpoint")

    p = sub.add_parser("session-close")
    p.add_argument("--name", default="longterm")
    p.add_argument("--session", required=True)
    p.add_argument("--ttl-hours", type=float, default=24.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--config")
    p.add_argument("--checkpoint")

    p = sub.add_parser("promote")
    p.add_argument("--name", default="longterm")
    p.add_argument("--session", required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--kind", default="fact", choices=["fact", "episode", "semantic"])
    p.add_argument("--confidence", type=float)
    p.add_argument("--tags", nargs="*", default=[])
    p.add_argument("--ttl-hours", type=float, default=24.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--config")
    p.add_argument("--checkpoint")

    args = ap.parse_args()
    _run(args.cmd, args)


def _run(cmd: str, args: argparse.Namespace) -> None:
    from astra.memory import MemoryStore
    from astra.memory.records import MemoryRecord

    if cmd == "init":
        store = MemoryStore.open(name=args.name, embedder=_embedder(args))
        print(f"initialized {store.path} (schema v{store.meta['schema']}, {store.meta['embedding_config']})")
        return

    store = _store(args)

    if cmd == "add":
        rec = MemoryRecord(content=args.content, kind=args.kind, source=args.source,
                           confidence=args.confidence, tags=list(args.tags))
        if args.id:
            rec.id = args.id
        store.add(rec)
        print(f"added {rec.id} ({rec.kind}, {rec.content!r})")
    elif cmd == "get":
        rec = store.get(args.id, include_all=args.all)
        print(json.dumps(rec.to_dict() if rec else None, indent=2))
    elif cmd == "list":
        recs = store.records(kinds=args.kind or None, include_deleted=args.deleted)
        for r in sorted(recs, key=lambda r: r.created_at):
            print(f"{r.id}  {r.kind:<8} rev{r.revision}  {r.content!r}")
    elif cmd == "query":
        hits = store.search(query=args.text, k=args.k, budget_tokens=args.budget_tokens,
                            include_disputed=args.disputed, allow_quarantine=args.quarantine)
        for h in hits:
            print(f"{h.score:.3f}  {h.record.id}  {h.components}")
            print(f"    {h.record.content!r}")
    elif cmd == "correct":
        kw = {}
        if args.content:
            kw["content"] = args.content
        if args.confidence is not None:
            kw["confidence"] = args.confidence
        new = store.correct(args.id, **kw)
        print(f"corrected {args.id} -> {new.id} (revision {new.revision})")
    elif cmd == "delete":
        store.soft_delete(args.id, purge=args.purge)
        print(f"deleted {args.id} (purge={args.purge})")
    elif cmd == "expire":
        n = store.expire()
        print(f"expired {n} records")
    elif cmd == "conflicts":
        for r in store.conflicts():
            print(f"{r.id}  {r.content!r}")
    elif cmd == "resolve":
        rec = store.resolve(args.id, by=args.by)
        print(f"resolved {args.id} -> {rec.verification_status}")
    elif cmd == "audit":
        if not store.audit_path.exists():
            print("(no audit lines)")
            return
        lines = store.audit_path.read_text().strip().splitlines()
        for line in lines[-args.tail:]:
            d = json.loads(line)
            print(f"{d['ts']}  {d['action']:<14} {d.get('id', '')}")
    elif cmd in ("remember", "session-recall", "session-close", "promote"):
        from datetime import timedelta

        from astra.memory import SessionMemory

        session = SessionMemory(store, args.session, ttl=timedelta(hours=args.ttl_hours))
        if cmd == "remember":
            if args.reply:
                recs = session.add_turn(user=args.content, assistant=args.reply)
            else:
                recs = [session.add(args.content)]
            for r in recs:
                print(f"session {args.session}: {r.id} ({r.source}, ttl {args.ttl_hours}h)")
        elif cmd == "session-recall":
            hits = session.recall(query=args.text, k=args.k)
            for h in hits:
                print(f"{h.score:.3f}  {h.record.id}  {h.components}")
                print(f"    {h.record.content!r}")
        elif cmd == "session-close":
            n = session.close()
            print(f"closed session {args.session}: {n} records deleted")
        elif cmd == "promote":
            from astra.memory import promote as promote_record

            rec = promote_record(session, args.id, kind=args.kind,
                                 confidence=args.confidence, tags=list(args.tags))
            print(f"promoted {args.id} -> long-term {rec.id} ({rec.kind}, "
                  f"verification={rec.verification_status})")


if __name__ == "__main__":
    main()
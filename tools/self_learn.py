#!/usr/bin/env python3
"""Continuous self-learning daemon (Phase 6, feedback-driven).

Watches ``learning/inbox/`` for new feedback (human corrections), validates it
through the trust cascade (feedback.py), routes accepted feedback into the
ExperienceStore, and — whenever new experiences arrive — runs one
improvement cycle:

  1. snapshot: active ``astra-prose`` checkpoint (registry) or the finished
     prose model, whichever is current;
  2. candidate: train off that base on all active experiences + a replay of
     the prose pretraining corpus (prevents forgetting) with retrieved long-
     term memory conditioning;
  3. measure: base-vs-candidate CE on a held-out target partition (must gain)
     and a regression partition (prose val slice; must not regress);
  4. gate + promote/reject via the GateEngine; every decision is audited and
     written to ``checkpoints/selflearn/reports/``.

The daemon idles between cycles and only trains when genuinely new feedback
appears, so it never churns the same lessons.

Inbox format (any mix under ``learning/inbox/``):
  - ``.json`` / ``.jsonl``: a feedback dict or a list of feedback dicts —
    ``{source, tier, text, confidence, verification_status, context}``
  - ``.txt``: one verified correction per line, e.g. ``Astra is from Nova.``

Usage:
    python tools/self_learn.py                # watch forever (systemd)
    python tools/self_learn.py --once         # single pass, then exit
    python tools/self_learn.py --once --steps 60   # quick QA pass
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import numpy as np
from astra.learning.audit import AuditEntry, AuditLog
from astra.learning.candidate import CandidateConfig, train_candidate
from astra.learning.evaluate import partition_metrics
from astra.learning.experience import ExperienceStore, make_example
from astra.learning.feedback import make_feedback, validate_feedback
from astra.learning.gates import GateEngine, GateItem
from astra.memory import search_memory_block
from astra.memory.store import MemoryStore
from astra.model import ModelConfig
from astra.registry import ModelRegistry
from astra.tokenizer import ByteLevelBPE
from astra.utils import write_json

PROSE_CFG = "configs/astra5m_prose.json"
PROSE_FINAL = "checkpoints/astra5m_prose/resumed/final.npz"
PROSE_TRAIN = "datasets/prose/train.txt"
PROSE_VAL = "datasets/prose/val.txt"
NAME_MODEL = "astra-prose"

_DEFAULT_INBOX = "learning/inbox"
_DEFAULT_STORE = "learning/store"
_DEFAULT_QUARANTINE = "learning/quarantine.jsonl"
_DEFAULT_CURSOR = "learning/selflearn_cursor.json"
_DEFAULT_STATUS = "learning/selflearn_status.json"
_DEFAULT_REPORTS = "checkpoints/selflearn/reports"
_DEFAULT_CANDIDATE = "checkpoints/selflearn/candidate"
_DEFAULT_REGISTRY = "checkpoints/registry.json"
_DEFAULT_AUDIT = "learning/audit.jsonl"
_DEFAULT_MEMORY = "memory/store"
_DEFAULT_TARGET = "datasets/chat/holdout.txt"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------- ingestion


def _read_feedback_file(path: Path) -> list[dict]:
    """Parse one inbox file into a list of feedback dicts."""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    if path.suffix in (".json", ".jsonl"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            fallback: list[dict] = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    fallback.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            data = fallback
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return []
        return [d for d in data if isinstance(d, dict) and (d.get("text") or "").strip()]
    # plain txt: one verified correction per line
    return [
        {"text": ln.strip()}
        for ln in text.splitlines()
        if ln.strip()
    ]


def _to_feedback(raw: dict) -> dict:
    return make_feedback(
        source=raw.get("source", "user"),
        tier=raw.get("tier", "human_verification"),
        text=raw.get("text", ""),
        confidence=float(raw.get("confidence", 0.95)),
        verification_status=raw.get("verification_status", "human_verified"),
        context=raw.get("context", ""),
    )


def ingest_files(
    inbox: Path,
    store: ExperienceStore,
    quarantine_path: Path,
    processed: set[str],
) -> dict[str, int]:
    """Validate inbox files -> quarantine failures, store accepted examples."""
    stats = {"files": 0, "lines": 0, "accepted": 0, "duplicate": 0, "quarantined": 0}
    for path in sorted(inbox.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        key = f"{path.name}:{path.stat().st_size}"
        if key in processed:
            continue
        stats["files"] += 1
        for raw in _read_feedback_file(path):
            if not raw.get("text"):
                continue
            stats["lines"] += 1
            fb = _to_feedback(raw)
            res = validate_feedback(fb, store=store)
            if res["result"] == "accepted":
                payload = {"input": fb.get("context", ""), "output": fb["text"]}
                ex = make_example(
                    "sft", payload,
                    feedback_id=fb["id"], source=fb["source"],
                    confidence=fb["confidence"], verification_status=fb["verification_status"],
                    trust_tier=fb["tier"],
                )
                _ex_id, added = store.add(ex)
                if added:
                    stats["accepted"] += 1
            elif res["result"] == "duplicate":
                stats["duplicate"] += 1
            else:
                stats["quarantined"] += 1
                with open(quarantine_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(res, ensure_ascii=False) + "\n")
        processed.add(key)
    return stats


# ------------------------------------------------------------ cycle


def _load_base(args) -> tuple[str, ModelConfig, ByteLevelBPE, np.ndarray, np.ndarray]:
    raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
    cfg = ModelConfig.from_dict(raw["model"])
    tok = ByteLevelBPE.load(raw["tokenizer"])
    cfg.vocab_size = len(tok)
    cfg = ModelConfig.from_dict(cfg.to_dict())

    registry = ModelRegistry(path=args.registry)
    current = registry.current(NAME_MODEL)
    base = args.base
    if current is not None and Path(current.path).exists():
        base = current.path
    elif args.base and Path(args.base).exists():
        base = args.base
    if not Path(base).exists():
        raise FileNotFoundError(f"no usable base checkpoint: {base}")

    replay = np.array(tok.encode(Path(PROSE_TRAIN).read_text(encoding="utf-8")), dtype=np.int32)
    regress = _bounded_lines(tok, PROSE_VAL, max_lines=20)
    return base, cfg, tok, replay, np.asarray(regress, dtype=object)


def _target_lines(args) -> list[str]:
    target = Path(args.target)
    if target.exists():
        return [ln.strip() for ln in target.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [ln.strip() for ln in Path("datasets/learning/nova_heldout.txt")
            .read_text(encoding="utf-8").splitlines() if ln.strip()]


def _bounded_lines(tok: ByteLevelBPE, path: str, max_lines: int = 20) -> list[str]:
    """Slice a corpus into window-sized-ish chunks so CE eval stays bounded.

    ``datasets/prose/val.txt`` is one paragraph per line and can be far longer
    than the model window; encoding a whole line and running it through
    forward_loss would allocate a huge attention matrix. Split on whitespace
    into <= max_seq_len windows instead.
    """
    text = Path(path).read_text(encoding="utf-8")
    words = text.split()
    lines: list[str] = []
    cur: list[str] = []
    budget = 96  # token-ish approximation: ~4.6 B/token, keep well under 128
    for w in words:
        cur.append(w)
        if len(cur) >= budget:
            lines.append(" ".join(cur))
            cur = []
        if len(lines) >= max_lines:
            break
    if cur and len(lines) < max_lines:
        lines.append(" ".join(cur))
    return lines[:max_lines]


def _build_gates(min_gain: float, max_regress: float) -> list[GateItem]:
    return [
        GateItem(id="target", type="gain", metric="target_loss", min_gain=min_gain,
                 note="candidate must improve the held-out target"),
        GateItem(id="regress", type="no_regress", metric="regress_loss", max_regress=max_regress,
                 note="must not regress prose capability"),
    ]


def run_cycle(args, experiences: list[dict]) -> dict:
    base, cfg, tok, replay, regress_lines = _load_base(args)
    target_lines = _target_lines(args)

    mem = MemoryStore.open(name="longterm", directory=args.memory)
    mem_query = " ".join(str(e["payload"].get("output") or "") for e in experiences).strip() or "Astra"
    budget = max(64, cfg.max_seq_len // 2)
    mem_block = search_memory_block(mem, mem_query, tok, k=4, budget_tokens=budget)
    memory_context = mem_block.text if mem_block.included else None

    ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    out_dir = f"{args.out_dir}/{ts}"

    cc = CandidateConfig(
        max_steps=args.steps,
        peak_lr=args.peak_lr,
        warmup_steps=max(5, args.steps // 15),
        batch_seq=4,
        replay_ratio=args.replay_ratio,
    )
    if args.dry_run:
        res = _DryRunResult(base, len(experiences), cc)
    else:
        res = train_candidate(
            base_checkpoint=base,
            tokenizer=tok,
            model_config=cfg,
            experiences=experiences,
            replay_ids=replay,
            config=cc,
            out_dir=out_dir,
            seed=args.seed,
            memory_context=memory_context,
        )

    partitions = {
        "target_loss": target_lines,
        "regress_loss": list(regress_lines),
        "experience_loss": [e["payload"].get("output", "") for e in experiences][:10],
    }
    metrics = partition_metrics(base, res.checkpoint, tok, cfg, partitions)

    eng = GateEngine(_build_gates(args.min_gain, args.max_regress), policy=args.policy)
    decision = eng.evaluate(metrics, human_approval=args.approve)

    report = {
        "cycle": ts,
        "base": base,
        "candidate": res.checkpoint,
        "steps": res.steps,
        "final_loss": res.final_loss if hasattr(res, "final_loss") else None,
        "decision": decision.to_dict(),
        "metrics": metrics,
        "experiences": [{"id": e["id"], "kind": e["kind"]} for e in experiences],
        "num_experiences": len(experiences),
        "memory_context_tokens": len(mem_block.tokens) if mem_block.included else 0,
        "dry_run": args.dry_run,
    }

    audit = AuditLog(path=args.audit)
    registry = ModelRegistry(path=args.registry)

    if decision.accepted and not args.dry_run:
        cfg_id = json.dumps(cfg.to_dict(), sort_keys=True)
        rec = registry.register(
            name=NAME_MODEL,
            semver=f"{args.semver}-{ts}",
            path=res.checkpoint,
            config_id=cfg_id,
            data_manifest_id=f"selflearn-{ts}",
            checklist_id="selflearn-cycle",
            eval_report_id=f"{args.reports}/{ts}.json",
            params=res.manifest["params"],
            step=res.steps,
            notes=f"accepted self-learn cycle (gates passed): {decision.summary}",
        )
        registry.promote(NAME_MODEL, rec.sha256, notes="gates passed (self-learn)")
        audit.append(AuditEntry(event="promote", model_name=NAME_MODEL,
                                base_sha=report["base"], candidate_sha=rec.sha256,
                                decision=decision.to_dict(), notes="self-learn cycle accepted"))
        report["promoted"] = True
        report["promoted_sha"] = rec.sha256
        report["candidate"] = res.checkpoint
    else:
        audit.append(AuditEntry(event="reject", model_name=NAME_MODEL,
                                decision=decision.to_dict(), notes="self-learn cycle rejected"))
        report["promoted"] = False

    Path(args.reports).mkdir(parents=True, exist_ok=True)
    write_json(f"{args.reports}/{ts}.json", report)
    return report


class _DryRunResult:
    def __init__(self, checkpoint: str, n_ex: int, cfg: CandidateConfig):
        self.checkpoint = checkpoint
        self.manifest = {"params": 0, "steps": 0}
        self.steps = 0
        self.elapsed_s = 0.0
        self.final_loss = float("nan")

    @property
    def base(self) -> str:
        return self.checkpoint


# -------------------------------------------------------------------- main


def _load_cursor(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return set(data.get("processed", []))
    except (OSError, ValueError):
        return set()


def _save_cursor(path: Path, processed: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(str(path), {"processed": sorted(processed), "updated_at": _now_iso()})


def _write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(str(path), payload)


def serve(args) -> int:
    inbox = Path(args.inbox)
    inbox.mkdir(parents=True, exist_ok=True)
    quarantine = Path(args.quarantine)
    quarantine.parent.mkdir(parents=True, exist_ok=True)
    store = ExperienceStore(root=args.store)
    cursor_path = Path(args.cursor)
    processed = _load_cursor(cursor_path)
    last_report: dict | None = None

    while True:
        try:
            stats = ingest_files(inbox, store, quarantine, processed)
            _save_cursor(cursor_path, processed)

            new_examples = store.active(kinds={"sft", "preference"})
            if last_report is not None:
                # only retrain when genuinely new experiences exist
                known = {e["id"] for e in last_report.get("experiences", [])}
                pending = [e for e in new_examples if e["id"] not in known]
            else:
                pending = new_examples

            if not pending and stats["accepted"] == 0:
                status = {"ts": _now_iso(), "idle": True, "inbox": stats,
                          "active_examples": len(store.active()), "last_cycle": None}
                _write_status(Path(args.status), status)
                if args.once:
                    return 0
                time.sleep(args.interval)
                continue

            report = run_cycle(args, new_examples)
            last_report = report
            _write_status(Path(args.status), {
                "ts": _now_iso(), "idle": False, "inbox": stats,
                "active_examples": len(store.active()),
                "last_cycle": report.get("cycle"),
                "decision": report.get("decision", {}).get("summary", ""),
                "promoted": report.get("promoted", False),
            })
        except FileNotFoundError as e:
            print(f"[self-learn] base issue: {e}", flush=True)
            if args.once:
                return 2
        except Exception as e:  # noqa: BLE001 — keep daemon alive across transient failures
            import traceback
            traceback.print_exc()
            status = {"ts": _now_iso(), "idle": False, "error": str(e), "last_cycle": None}
            _write_status(Path(args.status), status)
            if args.once:
                return 3
        if args.once:
            return 0
        time.sleep(args.interval)


def main() -> None:
    ap = argparse.ArgumentParser(description="Astra continuous self-learning daemon")
    ap.add_argument("--inbox", default=_DEFAULT_INBOX)
    ap.add_argument("--store", default=_DEFAULT_STORE)
    ap.add_argument("--quarantine", default=_DEFAULT_QUARANTINE)
    ap.add_argument("--cursor", default=_DEFAULT_CURSOR)
    ap.add_argument("--status", default=_DEFAULT_STATUS)
    ap.add_argument("--reports", default=_DEFAULT_REPORTS)
    ap.add_argument("--out-dir", default=_DEFAULT_CANDIDATE)
    ap.add_argument("--registry", default=_DEFAULT_REGISTRY)
    ap.add_argument("--audit", default=_DEFAULT_AUDIT)
    ap.add_argument("--memory", default=_DEFAULT_MEMORY)
    ap.add_argument("--target", default=_DEFAULT_TARGET)
    ap.add_argument("--base", default=PROSE_FINAL)
    ap.add_argument("--config", default=PROSE_CFG)
    ap.add_argument("--semver", default="0.10.0")
    ap.add_argument("--policy", default="auto", choices=["auto", "manual"])
    ap.add_argument("--approve", action="store_true")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replay-ratio", type=float, default=0.5)
    ap.add_argument("--peak-lr", type=float, default=1e-4)
    ap.add_argument("--min-gain", type=float, default=0.05)
    ap.add_argument("--max-regress", type=float, default=0.1)
    ap.add_argument("--interval", type=int, default=120, help="idle poll seconds")
    ap.add_argument("--once", action="store_true", help="single pass then exit")
    ap.add_argument("--dry-run", action="store_true", help="validate/ingest/gate without training")
    args = ap.parse_args()
    if args.dry_run:
        args.steps = 0
    raise SystemExit(serve(args))


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Phase-6 self-improvement driver (docs/ROADMAP.md § Phase 6, docs/LEARNING.md § 1.11-1.13).

Supervised continuous-improvement: candidate train -> evaluate -> gate ->
promote/reject -> audit. This is the *automation layer* on top of the Phase-5
learning loop.

Pipeline:
  1. intake validated feedback -> ExperienceStore (deduped, leak-checked)
  2. train a candidate off the active checkpoint (replay to prevent forgetting)
  3. measure base-vs-candidate CE on target + regression partitions
  4. run the GateEngine (gain on target, no regression elsewhere)
  5. on ACCEPT: register candidate in the ModelRegistry and promote it active;
     on REJECT: keep base active and record the rejection.
  6. every decision is written to the append-only audit log.

``--rollback-drill`` exercises the rollback path offline: promote a fake
"regressing" candidate then roll back, asserting the registry returns to the
previous active sha and an audit trail is written. Used by the release gates
(docs/EVALUATION.md rollback drills).

Usage:
    python tools/self_improve.py --config configs/toy_name.json
    python tools/self_improve.py --config configs/toy_name.json --rollback-drill
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import numpy as np
from astra.learning.audit import AuditEntry, AuditLog
from astra.learning.candidate import CandidateConfig, train_candidate
from astra.learning.evaluate import partition_metrics
from astra.learning.experience import ExperienceStore, make_example
from astra.learning.feedback import make_feedback, validate_feedback
from astra.learning.gates import GateEngine, GateItem
from astra.model import ModelConfig
from astra.registry import ModelRegistry
from astra.tokenizer import ByteLevelBPE
from astra.utils import read_json, write_json


def _load(
    ap_args,
) -> tuple[ModelConfig, ByteLevelBPE, ExperienceStore]:
    raw = read_json(ap_args.config)
    cfg = ModelConfig.from_dict(raw["model"])
    tok = ByteLevelBPE.load(raw["tokenizer"])
    cfg.vocab_size = len(tok)
    cfg = ModelConfig.from_dict(cfg.to_dict())
    store = ExperienceStore(root=ap_args.store)
    return cfg, tok, store


def intake_nova(store: ExperienceStore) -> list[dict]:
    """Validate the Nova feedback corpus into deduped SFT examples."""
    novel_lines = Path("datasets/learning/nova_train.txt").read_text(encoding="utf-8").splitlines()
    examples: list[dict] = []
    for line in novel_lines:
        fb = make_feedback(
            "human",
            "human_verification",
            line,
            confidence=0.98,
            verification_status="human_verified",
            context="Astra fact correction",
        )
        res = validate_feedback(fb, store=store)
        if res["result"] != "accepted":
            continue
        payload = {"input": "", "output": line}
        ex = make_example(
            "sft", payload,
            feedback_id=fb["id"], source=fb["source"], confidence=fb["confidence"],
            verification_status=fb["verification_status"], trust_tier=fb["tier"],
        )
        ex_id, _added = store.add(ex)
        examples.append(store.get(ex_id))
    return examples


def leak_gate(examples: list[dict]) -> list[str]:
    """Return held-out lines that leaked into the experience set (must be empty)."""
    heldout = Path("datasets/learning/nova_heldout.txt").read_text(encoding="utf-8").splitlines()
    trained = {ex["payload"]["output"] for ex in examples}
    return [s for s in heldout if s in trained]


def build_gates() -> list[GateItem]:
    return [
        GateItem(id="target", type="gain", metric="target_nova_loss", min_gain=0.1,
                 note="candidate must measurably improve the held-out target partition"),
        GateItem(id="regress", type="no_regress", metric="regression_astra_loss", max_regress=0.1,
                 note="must not regress prior capability by more than 0.1 CE"),
    ]


def run_loop(args) -> dict:
    cfg, tok, store = _load(args)
    examples = intake_nova(store)
    print(f"[intake] {len(examples)} validated experiences (deduped)")

    leaked = leak_gate(examples)
    if leaked:
        raise SystemExit(f"leak: held-out lines found in experience set: {leaked}")

    replay = np.array(tok.encode(Path("datasets/name/train.txt").read_text(encoding="utf-8")), dtype=np.int32)
    cc = CandidateConfig(
        max_steps=args.steps,
        peak_lr=args.peak_lr,
        warmup_steps=max(5, args.steps // 15),
        batch_seq=4,
        replay_ratio=args.replay_ratio,
    )
    res = train_candidate(
        base_checkpoint=args.base,
        tokenizer=tok,
        model_config=cfg,
        experiences=examples,
        replay_ids=replay,
        config=cc,
        out_dir=args.out_dir,
        seed=args.seed,
    )
    print(f"[train ] {res.steps} steps, final CE {res.final_loss:.4f}, {res.checkpoint}")

    heldout = Path("datasets/learning/nova_heldout.txt").read_text(encoding="utf-8").splitlines()
    astra_lines = Path("datasets/name/train.txt").read_text(encoding="utf-8").splitlines()[:20]
    metrics = partition_metrics(
        args.base, res.checkpoint, tok, cfg,
        {"target_nova_loss": heldout, "regression_astra_loss": astra_lines},
    )
    for k, v in metrics.items():
        print(f"[eval  ] {k:26s} base={v['base']:.3f} cand={v['candidate']:.3f} delta={v['candidate']-v['base']:+.3f}")

    eng = GateEngine(build_gates(), policy=args.policy)
    decision = eng.evaluate(metrics, human_approval=args.approve)
    for r in decision.results:
        print(f"[gate  ] {r.gate.id:12s} {r.detail} => {'PASS' if r.passed else 'FAIL'}")
    print(f"[decision] {decision.summary}")

    report = {
        "base": args.base,
        "candidate": res.checkpoint,
        "candidate_sha": res.manifest.get("base_sha256", ""),
        "seed": args.seed,
        "decision": decision.to_dict(),
        "metrics": metrics,
        "experience_ids": [e["id"] for e in examples],
    }

    if decision.accepted:
        registry = ModelRegistry(path=args.registry)
        cfg_id = json.dumps(cfg.to_dict(), sort_keys=True)
        rec = registry.register(
            name="astra-name",
            semver=args.semver,
            path=res.checkpoint,
            config_id=cfg_id,
            data_manifest_id="nova-1.0",
            checklist_id="phase6-core",
            eval_report_id=f"{args.out_dir}/learning_report.json",
            params=res.manifest["params"],
            step=res.steps,
            notes="promoted by self_improve loop (gates passed)",
        )
        registry.promote("astra-name", rec.sha256, notes="gates passed")
        audit = AuditLog(path=args.audit)
        audit.append(AuditEntry(
            event="promote",
            model_name="astra-name",
            base_sha="",  # registry tracks history
            candidate_sha=rec.sha256,
            decision=decision.to_dict(),
            notes="promoted via self_improve loop",
        ))
        report["promoted"] = True
        report["promoted_sha"] = rec.sha256
        print(f"[promote] astra-name -> {rec.sha256[:12]} (semver {args.semver})")
    else:
        audit = AuditLog(path=args.audit)
        audit.append(AuditEntry(
            event="reject",
            model_name="astra-name",
            decision=decision.to_dict(),
            notes="gates failed; base remains active",
        ))
        report["promoted"] = False
        print("[reject ] base remains active (audit recorded)")

    write_json(f"{args.out_dir}/self_improve_report.json", report)
    print(f"[report] -> {args.out_dir}/self_improve_report.json")
    return report


def rollback_drill(args) -> dict:
    """Exercise auto-rollback: promote a 'regressing' candidate, then roll back."""
    print("[drill  ] rollback drill: simulate a post-deploy regression")
    registry = ModelRegistry(path=args.registry)
    base = registry.current("astra-name")
    base_sha = base.sha256 if base else ""
    print(f"[drill  ] current active: {base_sha[:12] if base_sha else '<none>'}")

    # register a deliberately-worse candidate under the same name; the drill is
    # an explicit audit action, so force=True re-registers the existing artifact
    worst = Path("checkpoints/phase0/final.npz")
    if not worst.exists():
        raise SystemExit("rollback drill needs checkpoints/phase0/final.npz registered artifact")
    cfg = read_json(args.config)
    cfg_id = json.dumps(cfg["model"], sort_keys=True)
    rec = registry.register(
        name="astra-name",
        semver=f"{args.semver}-drill",
        path=str(worst),
        config_id=cfg_id,
        data_manifest_id="drill",
        checklist_id="rollback-drill",
        notes="rollback drill artifact (should be rolled back)",
        force=True,  # explicit audited action; overrides the immutable re-register guard
    )
    registry.promote("astra-name", rec.sha256, notes="drill: promote regressing candidate")
    print(f"[drill  ] promoted regressing candidate {rec.sha256[:12]}")

    restored = registry.rollback("astra-name", notes="drill: simulated regression")
    audit = AuditLog(path=args.audit)
    audit.append(AuditEntry(
        event="rollback",
        model_name="astra-name",
        candidate_sha=rec.sha256,
        decision={"drill": True, "rolled_back_from": rec.sha256},
        notes="rollback drill completed",
    ))
    ok = restored is not None and restored.sha256 == base_sha
    print(f"[drill  ] rolled back to {getattr(restored, 'sha256', '?')[:12]} "
          f"({'OK' if ok else 'MISMATCH'})")
    if not ok:
        raise SystemExit("rollback drill FAILED: did not restore previous active sha")
    return {"drill": "ok", "restored_sha": restored.sha256}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/toy_name.json")
    ap.add_argument("--base", default="checkpoints/name/resumed/final.npz")
    ap.add_argument("--out-dir", default="checkpoints/candidate")
    ap.add_argument("--store", default="learning/store")
    ap.add_argument("--audit", default="learning/audit.jsonl")
    ap.add_argument("--registry", default="checkpoints/registry.json")
    ap.add_argument("--semver", default="0.9.0")
    ap.add_argument("--policy", default="auto", choices=["auto", "manual"])
    ap.add_argument("--approve", action="store_true", help="human approval (manual policy)")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replay-ratio", type=float, default=0.5)
    ap.add_argument("--peak-lr", type=float, default=1e-4)
    ap.add_argument("--rollback-drill", action="store_true",
                    help="run the offline rollback drill instead of the loop")
    args = ap.parse_args()

    if args.rollback_drill:
        rollback_drill(args)
        return
    run_loop(args)


if __name__ == "__main__":
    main()
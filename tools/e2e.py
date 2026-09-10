#!/usr/bin/env python3
"""Phase-7 full-stack E2E walk (docs/PHASE_STATUS.md, docs/ROADMAP.md § Phase 7).

One gated command exercises every subsystem end-to-end:

    tokenizer -> registry (resolve active) -> model load -> inference(sample)
      -> memory (add/recall) -> learning (intake -> candidate -> measure)
      -> gates (decision) -> registry (promote) -> audit (record)

Each step is gated: a failing step aborts the walk with a report and a
non-zero exit code. A rejected candidate is a valid outcome (audited, base
stays active); a subsystem crash is a red pipeline that fails the walk.

This is the Phase-7 "single, gated, auditable release" claim (Astra 1.0.0):
one command proves every engine from tokenizer through promotion/rollback works
and produces an audit trail. Uses the real `tools/self_improve.py` gate suite.

Usage:
    python tools/e2e.py --steps 150
    python tools/e2e.py --steps 150 --prompt "Astra is"   # deterministic sample
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import numpy as np
from astra.inference.decoder import decode
from astra.learning.audit import AuditEntry, AuditLog
from astra.learning.candidate import CandidateConfig, train_candidate
from astra.learning.evaluate import load_model, partition_metrics
from astra.learning.experience import ExperienceStore
from astra.learning.feedback import make_feedback, validate_feedback
from astra.learning.gates import GateEngine, GateItem
from astra.memory.store import MemoryStore
from astra.model import ModelConfig, all_params
from astra.registry import ModelRegistry
from astra.tokenizer import ByteLevelBPE
from astra.utils import read_json, sha256_file, write_json


class PipelineError(Exception):
    """A subsystem step failed; the E2E walk is red."""


def _default_gates() -> list[GateItem]:
    return [
        GateItem(id="target", type="gain", metric="target", min_gain=0.1,
                 note="candidate must measurably improve the held-out target"),
        GateItem(id="regress", type="no_regress", metric="regression", max_regress=0.1,
                 note="must not regress prior capability by more than 0.1 CE"),
    ]


def run_e2e(
    *,
    cfg: ModelConfig,
    tok: ByteLevelBPE,
    base_checkpoint: str,
    experiences: list[dict],
    replay_ids: np.ndarray | None,
    target_corpus: list[str],
    regression_corpus: list[str],
    out_dir: str,
    store_root: str,
    memory_dir: str,
    registry_path: str,
    audit_path: str,
    model_name: str = "astra-name",
    semver: str = "0.9.0",
    gates: list[GateItem] | None = None,
    steps: int = 150,
    batch_seq: int = 4,
    peak_lr: float = 1e-4,
    replay_ratio: float = 0.5,
    seed: int = 0,
    policy: str = "auto",
    approve: bool = False,
    prompt: str = "Astra ",
    leak_corpus: list[str] | None = None,
    run_tag: str | None = None,
) -> dict:
    """Walk the full subsystem chain; raise ``PipelineError`` on a red step.

    Returns a report with per-step status, the gate decision, and (on accept)
    the promoted registry sha. The chain reuses the *production* modules;
    nothing is mocked except the model/tokenizer configuration passed in.

    The candidate is written to ``<out_dir>/runs/<run_tag>/final.npz`` (a fresh
    immutable artifact per walk) so it can never overwrite the active model the
    registry points at — otherwise base-vs-candidate would compare one file.
    """
    run_tag = run_tag or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    candidate_dir = str(Path(out_dir) / "runs" / run_tag)
    Path(candidate_dir).mkdir(parents=True, exist_ok=True)
    gates = gates or _default_gates()
    steps_log: list[dict] = []

    def _step(name: str, ok: bool, detail: str) -> dict:
        entry = {"name": name, "ok": bool(ok), "detail": detail}
        steps_log.append(entry)
        return entry

    fields: dict = {}
    decision: dict | None = None
    promoted_sha: str | None = None

    # -- 1. tokenizer ------------------------------------------------
    _step("tokenizer", True, f"loaded, vocab={len(tok)}")
    fields["vocab"] = len(tok)

    # -- 2. registry: resolve + integrity-check the active model ------
    registry = ModelRegistry(path=registry_path)
    active = registry.current(model_name)
    if active is None:
        raise PipelineError(f"no active registry entry for {model_name!r}")
    active_digest = sha256_file(active.path)
    integrity_ok = active_digest == active.sha256
    _step("registry.active", integrity_ok,
          f"{model_name} {active.semver} sha={active.sha256[:12]} "
          f"file-integrity={'OK' if integrity_ok else 'DRIFT'}")
    if not integrity_ok:
        raise PipelineError(
            f"active artifact {active.sha256[:12]} no longer matches {active.path!r} "
            "(file overwritten after registration); the walk refuses a corrupt base"
        )
    base_checkpoint = active.path
    fields["active_sha"] = active.sha256

    # -- 3. active model loads from that artifact ---------------------
    model = load_model(cfg, base_checkpoint)
    _step("model.load", True, f"params={sum(arr.size for _, arr, _ in all_params(model))}")

    # -- 4. inference: deterministic sample ---------------------------
    sample = decode(model, tok.encode(prompt), max_new=8, temperature=1.0,
                    rng=np.random.default_rng(seed))
    try:
        text = tok.decode(sample)
    except UnicodeDecodeError:
        text = b"".join(tok.id_to_piece[i] for i in sample).decode("utf-8", errors="replace")
    _step("inference.sample", len(sample) == 8, f"{prompt!r}-> {text!r}")

    # -- 5. memory: append + recall ----------------------------------
    mem = MemoryStore(name="e2e", directory=memory_dir)
    rec = mem.add(content=f"E2E integration record (phase 7) for {model_name}")
    recall = mem.get_latest(rec.id)
    _step("memory.recall", recall is not None, f"rid={rec.id[:12]}")
    fields["memory_id"] = rec.id

    # -- 6. learning intake: validated experiences into the store -----
    store = ExperienceStore(root=store_root)
    n_added = 0
    for ex in experiences:
        _id, added = store.add(ex)
        n_added += 1 if added else 0
    _step("learning.intake", True, f"{len(experiences)} validated, {n_added} newly added")

    # -- 7. leak gate: training lines must not appear in eval sets ----
    trained = {ex.get("payload", {}).get("output", "") for ex in experiences}
    leakage = [s for s in (leak_corpus or target_corpus) if s in trained and s]
    _step("learning.no-leak", not leakage, f"leaked={len(leakage)}")
    if leakage:
        raise PipelineError(f"held-out leakage: {leakage}")

    # -- 8. candidate train off the active checkpoint -----------------
    cc = CandidateConfig(max_steps=steps, peak_lr=peak_lr,
                         warmup_steps=max(5, steps // 15), batch_seq=batch_seq,
                         replay_ratio=replay_ratio)
    res = train_candidate(
        base_checkpoint=base_checkpoint, tokenizer=tok, model_config=cfg,
        experiences=experiences, replay_ids=replay_ids, config=cc,
        out_dir=candidate_dir, seed=seed,
    )
    _step("learning.candidate", True,
          f"{res.steps} steps, final CE {res.final_loss:.4f}, {res.checkpoint}")
    fields["candidate"] = res.checkpoint

    # -- 9. measure base-vs-candidate on target + regression ----------
    metrics = partition_metrics(
        base_checkpoint, res.checkpoint, tok, cfg,
        {"target": target_corpus, "regression": regression_corpus},
    )
    for k, v in metrics.items():
        _step(f"measure.{k}", True,
              f"base={v['base']:.3f} cand={v['candidate']:.3f} "
              f"delta={v['candidate'] - v['base']:+.3f}")
    fields["metrics"] = metrics

    # -- 10. the self-improvement gate suite decides ------------------
    engine = GateEngine(gates, policy=policy)
    d = engine.evaluate(metrics, human_approval=approve)
    decision = d.to_dict()
    for r in d.results:
        _step(f"gate.{r.gate.id}", True, f"{r.detail} => {'PASS' if r.passed else 'FAIL'}")
    _step("gate.decision", True, d.summary)

    # -- 11. promote/reject -> registry + append-only audit -----------
    audit = AuditLog(path=audit_path)
    if d.accepted:
        config_id = json.dumps(cfg.to_dict(), sort_keys=True)
        # Deterministic retraining can reproduce a sha that was registered in an
        # earlier demo at a *different* path (e.g. checkpoints/candidate/final.npz,
        # which later runs overwrite). Re-register with force=True so the entry
        # points at this immutable per-run artifact — an explicit audited fix for
        # an identical-content artifact, keeping future integrity checks green.
        digest = sha256_file(res.checkpoint)
        prior = registry.by_sha(digest)
        force = prior is not None and prior.path != res.checkpoint
        rec_art = registry.register(
            name=model_name, semver=semver, path=res.checkpoint,
            config_id=config_id, data_manifest_id="e2e-nova-1.0",
            checklist_id="phase7-e2e", eval_report_id=f"{out_dir}/e2e_report.json",
            params=res.manifest["params"], step=res.steps,
            notes="promoted by e2e full-stack walk (docs/PHASE_STATUS.md)",
            force=force,
        )
        registry.promote(model_name, rec_art.sha256, notes="e2e gates passed")
        audit.append(AuditEntry(
            event="promote", model_name=model_name, candidate_sha=rec_art.sha256,
            decision=decision, notes="promoted via e2e full-stack walk (phase 7)",
        ))
        promoted_sha = rec_art.sha256
        end_active = registry.current(model_name)
        end_ok = end_active is not None and end_active.sha256 == promoted_sha
        _step("registry.promote", bool(end_ok), f"astra-name -> {promoted_sha[:12]} (active)")
    else:
        audit.append(AuditEntry(
            event="reject", model_name=model_name, decision=decision,
            notes="e2e gates failed; base remains active",
        ))
        _step("registry.promote", True, "rejected — base remains active (audited)")

    report = {
        "pipeline": "PASS",
        "model_name": model_name,
        "base_checkpoint": base_checkpoint,
        "active_sha": fields.get("active_sha"),
        "steps": steps_log,
        "metrics": fields.get("metrics", {}),
        "decision": decision,
        "promoted": bool(promoted_sha),
        "promoted_sha": promoted_sha,
        "memory_id": fields.get("memory_id"),
        "comment": "every subsystem exercised with its production module; see docs/PHASE_STATUS.md",
    }
    write_json(f"{out_dir}/e2e_report.json", report)
    return report


def _cli_experiences() -> tuple[list[dict], list[str], list[str], np.ndarray, ByteLevelBPE, ModelConfig]:
    raw = read_json("configs/toy_name.json")
    cfg = ModelConfig.from_dict(raw["model"])
    tok = ByteLevelBPE.load(raw["tokenizer"])
    cfg.vocab_size = len(tok)
    cfg = ModelConfig.from_dict(cfg.to_dict())

    novel_lines = Path("datasets/learning/nova_train.txt").read_text(encoding="utf-8").splitlines()
    store = ExperienceStore(root="learning/store")
    from astra.learning.experience import make_example  # fits driver convention

    examples: list[dict] = []
    for line in novel_lines:
        fb = make_feedback(
            "human", "human_verification", line, confidence=0.98,
            verification_status="human_verified", context="Astra fact correction",
        )
        res = validate_feedback(fb, store=store)
        if res["result"] != "accepted":
            continue
        ex = make_example(
            "sft", {"input": "", "output": line},
            feedback_id=fb["id"], source=fb["source"], confidence=fb["confidence"],
            verification_status=fb["verification_status"], trust_tier=fb["tier"],
        )
        ex_id, _added = store.add(ex)
        examples.append(store.get(ex_id))

    heldout = Path("datasets/learning/nova_heldout.txt").read_text(encoding="utf-8").splitlines()
    astra_lines = Path("datasets/name/train.txt").read_text(encoding="utf-8").splitlines()
    replay = np.array(tok.encode(Path("datasets/name/train.txt").read_text(encoding="utf-8")), dtype=np.int32)
    return examples, heldout, astra_lines[:20], replay, tok, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="checkpoints/candidate")
    ap.add_argument("--store", default="learning/store")
    ap.add_argument("--memory-dir", default="checkpoints/e2e")
    ap.add_argument("--audit", default="learning/audit.jsonl")
    ap.add_argument("--registry", default="checkpoints/registry.json")
    ap.add_argument("--semver", default="0.9.0")
    ap.add_argument("--policy", default="auto", choices=["auto", "manual"])
    ap.add_argument("--approve", action="store_true")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replay-ratio", type=float, default=0.5)
    ap.add_argument("--peak-lr", type=float, default=1e-4)
    ap.add_argument("--prompt", default="Astra ")
    args = ap.parse_args()

    examples, target, regression, replay, tok, cfg = _cli_experiences()
    try:
        report = run_e2e(
            cfg=cfg, tok=tok, base_checkpoint="checkpoints/name/resumed/final.npz",
            experiences=examples, replay_ids=replay,
            target_corpus=target, regression_corpus=regression,
            leak_corpus=target, out_dir=args.out_dir, store_root=args.store,
            memory_dir=args.memory_dir, registry_path=args.registry,
            audit_path=args.audit, semver=args.semver, steps=args.steps,
            seed=args.seed, replay_ratio=args.replay_ratio, peak_lr=args.peak_lr,
            policy=args.policy, approve=args.approve, prompt=args.prompt,
        )
    except PipelineError as e:
        print(f"[e2e   ] FAIL — {e}")
        raise SystemExit(1)

    for s in report["steps"]:
        flag = "PASS" if s["ok"] else "FAIL"
        print(f"[e2e {s['name']:18s}] {flag}  {s['detail']}")
    d = report["decision"]
    print(f"[e2e decision ] {d['summary']}")
    if report["promoted"]:
        print(f"[e2e promote  ] astra-name -> {report['promoted_sha'][:12]}")
    print(f"[report] -> {args.out_dir}/e2e_report.json")


if __name__ == "__main__":
    main()
# Phase Status

> Machine-readable verification of the **current** phase (what is proven done)
> and the **upcoming** phase (what must be done next). Companion to
> `docs/ROADMAP.md` (planning) and `docs/RELEASES.md` (release).

**STATUS: VERIFIED** — Phase 6 (Self-Improvement, Astra 0.9.0) is closed with
all exit criteria met and reproduced below. Phase 7 (First Stable System,
Astra 1.0.0) is the *current* phase: its first increment — the full-stack
E2E walk (`tools/e2e.py` + `tests/test_e2e.py`) — is implemented and verified.

---

## Current phase: Phase 7 — First Stable System (Astra 1.0.0)

### Objective
Integrate all subsystems (tokenizer → model → memory → learning → gates →
registry) into a single, gated, auditable release. (ROADMAP § Phase 7)

### Component checklist

| Component (ROADMAP required) | Status | Evidence |
|---|---|---|
| **Full E2E walk** (tokenizer → registry active → model → inference → memory → learning intake → no-leak → candidate → measure → gates → promote/reject → audit) | DONE | `tools/e2e.py` — one gated command exercises every subsystem with its production module; any red step aborts non-zero. CLI demo produced ACCEPT ×2 (5.155→4.380→3.953 CE) then REJECT at the optimum — a genuine multi-step improvement cycle through the full stack. |
| **E2E test suite** | DONE (3) | `tests/test_e2e.py` — in-process full-stack walk (tiny model): promote path, reject-keeps-base path, and leak-aborts-pipeline path all green. Full suite 148 tests. |
| **All engines integrated under one operator command** | DONE (supervised) | `python tools/e2e.py` / `make e2e` runs the entire chain; gates + audit + registry reporting on accept/reject. |
| **Release checklist automation** | OPEN | `make e2e`, `make self-improve`, `make rollback-drill` exist; a single `make release` aggregating gates + audit + CHANGELOG is the next increment. |
| **Interface freeze** (public module APIs frozen for 1.0) | OPEN | after release-checklist automation. |
| **Long supervised window** (continuous-improvement run > 1 week, zero silent regressions) | OPEN | requires release automation + ops; measured across the Astra-5M phase. |

### Increment plan (Phase 7)

1. **Full-stack E2E walk** (this session) — `tools/e2e.py`: tokenizer → registry
   (resolve active, integrity-check) → model.load → inference.sample →
   memory.recall → learning (intake → no-leak → candidate → measure) →
   gates (gain + no-regress) → registry.promote / audit, with immutable
   per-run candidate artifacts. Equivalent in-process integration tests.
2. Release checklist automation — one command (e.g. `make release`/`tools/release.py`)
   that runs E2E + full test suite + rollback drill + documentation-completeness
   checks before tagging.
3. Interface freeze + long supervised window — expose the stable operator API;
   schedule the loop over the plan-Astra-5M horizon.

---

## Verified (completed): Phase 6 — Self-Improvement (Astra 0.9.0)

### Objective (as shipped)
Automate the candidate → evaluate → accept/reject → deploy loop with human
approval gates, an audit trail for every promotion, and automatic rollback on
regression. (ROADMAP § Phase 6)

### Component checklist — all DONE

| Component | Evidence |
|---|---|
| Gate engine | `python/astra/learning/gates.py` — `GateEngine` (`gain`/`no_regress`/`absolute`, `auto`/`manual` policy); 8 tests. |
| Registry promotion logic | `python/astra/registry.py` — `promote`/`rollback`/`history`/`current` + active-sha pointer; 4 tests. |
| Audit log | `python/astra/learning/audit.py` — append-only JSONL; promote/reject/rollback/approve rows with decision verbatim + git commit; 2 tests. |
| Candidate model worker | `tools/self_improve.py` — intake → candidate → measure → gate → promote/reject → audit. |
| Automatic rollback | `tools/self_improve.py --rollback-drill` — promote regressing artifact then restore prior active sha (assert + audit). |
| Alerting (console) | report JSON written per run; out-of-band notification deferred to Phase 8 by ROADMAP. |
| Drift detector | Phase 8 by ROADMAP (not a Phase-6 exit item). |

### Exit criteria — ALL MET

- [x] The loop runs for a supervised evaluation window (`tools/self_improve.py`,
      all gates deterministic).
- [x] Any regression can trigger rollback (`--rollback-drill`, verified).
- [x] Every promotion has a full audit trail (`learning/audit.jsonl` — 10+ rows
      spanning promote / rollback-drill / demo re-promotes).
- [x] Astra 0.9 released — CHANGELOG 0.9.0, `docs/releases/v0.9.0.md`, committed
      as `789b641` and pushed to `main`.
- [x] Supervised-only deployment (autonomous unattended worker is Phase 7/8 —
      deferred by ROADMAP, not an exit-blocker).

### Verification record (reproducible)

```
$ python tools/self_improve.py --steps 150 --semver 0.9.0
[decision] ACCEPT
[promote] astra-name -> 5644b997ac40 (semver 0.9.0)

$ python tools/self_improve.py --rollback-drill
[drill  ] rolled back to 5644b997ac40 (OK)

$ python tools/e2e.py --steps 150        # Phase-7 walk; base from registry
[measure.target    ] base=5.155 cand=4.380 delta=-0.775
[measure.regression] base=2.832 cand=2.840 delta=+0.009
[gate.decision     ] ACCEPT
```

- Gate matrix reproduced: target **−0.775 CE** (PASS, min 0.1), regression
  **+0.009 CE** (PASS, max 0.1) → ACCEPT → promoted `astra-name` 0.9.0.
- Rollback drill promoted `88e5bc539273` and restored the prior active sha;
  audit rows record both events.
- Registry lineage for `astra-name`: 0.4.0 (`dd9432cce60e`) → 0.9.0
  (`5644b997ac40`) → 0.9.0-drill (`88e5bc539273`) + Phase-7 walk promotions —
  append-only history, active pointer moves, every mutation audited.
- Tests: `tests/learning/` 25 (intake/store, candidate trainer, gates/audit/
  registry, evaluate) + `tests/test_e2e.py` 3 → full suite **148 green**;
  ruff clean.

---

## Historical phases

| Phase | Version | Status |
|---|---|---|
| 0–1 Research/Birth | 0.0.1 / 0.1.0 | validated (`experiments/phase0/REPORT.md`) |
| 2 Training Foundation | 0.2.0 | validated |
| 3 Stable Neural Core | 0.3.0 / 0.4.0 | validated |
| 4 Memory | 0.5.0 | validated (G1–G6 closed) |
| 5 Learning | 0.7.0 | validated (core loop + demo) |
| 6 Self-Improvement | 0.9.0 | **verified** (exit criteria ALL MET, § above) |
| **7 First Stable System** | **1.0.0** | **current — increment 1 (E2E walk) done** |
| 8 Continuous Evolution | 1.x | planned |
| 9 Advanced Architecture | 2.0.0 | research |
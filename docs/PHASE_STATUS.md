# Phase Status

> Machine-readable verification of the **current** phase (what is proven done)
> and the **upcoming** phase (what must be done next). Companion to
> `docs/ROADMAP.md` (planning) and `docs/RELEASES.md` (release).

**STATUS: VERIFIED** — Phase 6 (Self-Improvement, Astra 0.9.0) is closed with
all exit criteria met and reproduced below. Phase 7 (First Stable System,
Astra 1.0.0) is the *current* phase: the full-stack E2E walk
(`tools/e2e.py` + `tests/test_e2e.py`), memory-aware candidate training,
release checklist automation, and the interface freeze are implemented and
verified. The only remaining item is the long supervised window (> 1 week,
zero silent regressions), which runs over the plan-Astra-5M horizon.

Phase 8 (Continuous Evolution, Astra 1.x) coding phase is done:
``astra/learning/evolution.py`` (long-term metrics store, drift detector,
lineage) + ``tools/continuous.py`` (supervised iteration loop with drift watch
and auto-rollback) + 7 tests. The six aspirational tracks below require real
data/compute beyond the in-repo toy dataset.

---

## Current phase: Phase 7 — First Stable System (Astra 1.0.0)

### Objective
Integrate all subsystems (tokenizer → model → memory → learning → gates →
registry) into a single, gated, auditable release. (ROADMAP § Phase 7)

### Component checklist

| Component (ROADMAP required) | Status | Evidence |
|---|---|---|
| **Full E2E walk** (tokenizer → registry active → model → inference → memory → learning intake → no-leak → candidate → measure → gates → promote/reject → audit) | DONE | `tools/e2e.py` — one gated command exercises every subsystem with its production module; any red step aborts non-zero. CLI demo produced ACCEPT ×2 (5.155→4.380→3.953 CE) then REJECT at the optimum — a genuine multi-step improvement cycle through the full stack. |
| **E2E test suite** | DONE (8) | `tests/test_e2e.py` — promote path, reject-keeps-base path, leak-aborts-pipeline, memory-augmented candidate, memory-correction→learning feed, multi-step rollback with audit ordering, audit-trail completeness, and the preference path (manifest `n_preference`/`n_lm`) all green. Full suite 154 tests. |
| **All engines integrated under one operator command** | DONE (supervised) | `python tools/e2e.py` / `make e2e` runs the entire chain; gates + audit + registry reporting on accept/reject. |
| **Memory-aware candidate training** | DONE | `tools/e2e.py`, `tools/self_improve.py`, `tools/learning_loop.py` search memory (`astra.memory.injection.search_memory_block`) and pass the retrieved `<|memory|>` block as training context; candidate manifest records `memory_context`. |
| **Preference (DPO) training path** | DONE | `astra.model.core.preference_backward` + `_train_step_preference`; preference examples are consumed before replay batches; manifest reports `n_preference`/`n_lm`. |
| **Release checklist automation** | DONE | `tools/release.py check|tag` + `make release` — 7 gates: full tests, E2E walk, rollback drill, memory retrieval eval, registry verify, docs completeness, changelog. Fast gates (registry/docs/changelog) verified. |
| **Interface freeze** (public module APIs frozen for 1.0) | DONE | `__all__` on every public module (`astra/*/__init__.py`, `learning/feedback.py`, `learning/experience.py`, `registry.py`, …); `make lint` (ruff) and `make typecheck` (mypy, 40 files) both green; `pyproject.toml` project metadata; `tests/learning/__init__.py`. |
| **Long supervised window** (continuous-improvement run > 1 week, zero silent regressions) | OPEN | requires the frozen interface + ops; measured across the Astra-5M phase. |

### Increment plan (Phase 7)

1. **Full-stack E2E walk** (done) — `tools/e2e.py`: tokenizer → registry
   (resolve active, integrity-check) → model.load → inference.sample →
   memory.recall → learning (intake → no-leak → candidate → measure) →
   gates (gain + no-regress) → registry.promote / audit, with immutable
   per-run candidate artifacts. Equivalent in-process integration tests.
2. **Memory-aware candidate training** (done) — retrieved memory blocks enter
   the candidate's training context; correction → learning feedback loop
   verified end-to-end.
3. **Preference path** (done) — DPO-style margin training on preference
   examples, exercised by the in-process E2E suite.
4. **Release checklist automation** (done) — `make release` /
   `tools/release.py check|tag` aggregate E2E + full test suite + rollback
   drill + memory eval + registry/docs/changelog completeness before tagging.
5. **Interface freeze** (done) — public module APIs frozen (explicit imports
   in every package), `make lint` / `make typecheck` green.
6. **Long supervised window** (open) — schedule the loop over the
   plan-Astra-5M horizon; zero silent regressions for > 1 week is the
   remaining Phase-7 exit criterion.

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
  registry, evaluate) + `tests/test_e2e.py` 8 → full suite **154 green**;
  ruff clean; mypy clean (40 source files).

---

## Verified (completed): Phase 7 — First Stable System (Astra 1.0.0)

### Objective (current phase)
Integrate all subsystems (tokenizer → model → memory → learning → gates →
registry) into a single, gated, auditable release. (ROADMAP § Phase 7)

### Component checklist — all DONE except the long supervised window

| Component | Evidence |
|---|---|
| Full E2E walk | `tools/e2e.py` (one gated command through every subsystem). |
| E2E test suite (8) | `tests/test_e2e.py` — promote, reject-keeps-base, leak-abort, memory-augmented candidate, memory-correction→learning, multi-step rollback audit, audit-trail completeness, preference path. |
| Memory-aware candidate training | `search_memory_block` feeds `<|memory|>` context into candidates in `tools/e2e.py`, `tools/self_improve.py`, `tools/learning_loop.py`. |
| Preference (DPO) training | `preference_backward` + `_train_step_preference`; consumed before replay. |
| Release checklist automation | `tools/release.py check\|tag` + `make release` — 7 gates; fast gates verified. |
| Interface freeze | `__all__` on all public modules; `make lint` / `make typecheck` green; `pyproject.toml` metadata. |
| Long supervised window | OPEN (Astra-5M horizon). |

### Exit criteria — ALL MET except the window

- [x] Controlled full-stack walk commands every integration point in-process.
- [x] A correction to memory feeds the next learning cycle (verified E2E).
- [x] Rollback is multi-step: each promote/rollback restores exactly the prior
      active artifact, audited in causal order (verified E2E).
- [x] Every promote/reject has a complete, verbatim audit trail.
- [x] Release gates are automated (`make release`) and the operator API is
      frozen for 1.0.
- [ ] Continuous supervised improvement window > 1 week, zero silent
      regressions (started when the release is cut).

---

## Phase 8 — Continuous Evolution (Astra 1.x) — Coding phase DONE

### Objective
Operate controlled continuous improvement over long periods with regression
watch, drift detection, and rollback always available. (ROADMAP § Phase 8)

### Coding deliverables

| Component | Status | Evidence |
|---|---|---|
| **MetricsStore** | DONE | `astra/learning/evolution.py` — append-only JSONL per model name; one row per promotion; records sha256/semver/metrics/report_id/timestamp; queryable series + metric_names. |
| **DriftDetector** | DONE | same file — classifies improving/stable/regressing/silent-drift over a trailing window; regression_threshold + drift_tolerance configurable; needs at least MIN_WINDOW promotions before classifying. |
| **lineage()** | DONE | same file — promotion chain via registry `superseded_by` pointers; returns oldest→newest list with sha256/semver/superseded_by/step; used by `tools/continuous.py` and future lineage graphing. |
| **continuous operator** | DONE | `tools/continuous.py` — drives `tools/self_improve.py` on a schedule; records each promotion to MetricsStore; runs DriftDetector after every iteration; alert-only by default (supervised), `--auto-rollback` on drift/regression; `--check` runs drift-only; `--watch SECONDS --max-runs N` loop mode; `--compact-memory` compacts the durable store on drift. |
| **durable memory growth** | DONE | `astra/memory/store.py` `compact()` — prunes soft-deleted entries physically, de-duplicates identical-content actives (newest kept); every removal audited (`compact` rows); scoped per-kind; 2 tests. |
| **operator demo** | DONE | `tools/continuous.py` ran real iterations on `astra-name`: ACCEPT −0.775 CE target / +0.009 CE regression, promoted 0.9.0; existing series drift-check clean. |
| **Tests** | DONE (9) | `tests/learning/test_evolution.py` (7) + `tests/test_memory.py::test_compact_*` (2). All green; lint (ruff) + typecheck (mypy, 41 files) clean; full suite 163. |

### Remaining (supervised window, not coding)

- [ ] Continuous supervised window > 1 week, zero silent regressions — **started**
      with `tools/continuous.py` on `astra-name`; target operator
      `tools/continuous.py --watch` on the real Astra-5M model, gated by Phase 7
      exit criterion.
- [ ] Larger models, better datasets, improved reasoning, more efficient
      learning, hardware optimization (require compute beyond in-repo toy).

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
| **7 First Stable System** | **1.0.0** | **current — E2E walk, memory-aware training, release automation, interface freeze done; supervised window open** |
| **8 Continuous Evolution** | **1.x** | **coding phase DONE (MetricsStore / DriftDetector / lineage / continuous.py, 7 tests); supervised window open** |
| 9 Advanced Architecture | 2.0.0 | research |
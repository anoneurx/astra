# Phase Status

> Machine-readable verification of the **current** phase (what is proven done)
> and the **upcoming** phase (what must be done next). Companion to
> `docs/ROADMAP.md` (planning) and `docs/RELEASES.md` (release).

**STATUS: VALIDATED** — Phase 5 (Learning, Astra 0.7.0) released; Phase 6
(Self-Improvement, Astra 0.9.0) is the *current* phase with its core
implemented and verified below.

---

## Current phase: Phase 6 — Self-Improvement (Astra 0.9.0)

### Objective
Automate the candidate → evaluate → accept/reject → deploy loop with human
approval gates, audit trail for every promotion, and automatic rollback on
regression. (ROADMAP § Phase 6)

### Component checklist

| Component (ROADMAP required) | Status | Evidence |
|---|---|---|
| **Gate engine** (automated evaluation gates) | DONE | `python/astra/learning/gates.py` — `GateEngine` with `gain` / `no_regress` / `absolute` rules, `auto` / `manual` policy. `tests/learning/test_gates.py` (8 gate tests). |
| **Registry promotion logic** | DONE | `python/astra/registry.py` — `promote`, `rollback`, `history`, `current` on the append-only immutable registry; active-sha pointer per name. `tests/learning/test_gates.py` (4 registry tests). |
| **Audit log** | DONE | `python/astra/learning/audit.py` — append-only JSONL (`learning/audit.jsonl`); every promote/reject/rollback/approve is an immutable row with provenance + git commit. 2 audit tests. |
| **Candidate model worker** (loop) | DONE | `tools/self_improve.py` — supervised continuous-improvement driver: intake → candidate train → measure → gate → promote/reject → audit. |
| **Automatic rollback** | DONE (drill) | `tools/self_improve.py --rollback-drill` — promotes a regressing artifact and rolls back to the previous active sha, asserting restore + audit. |
| Alerting (surfacing rejected/promoted candidates) | OPEN | Console report written; out-of-band alerting deferred (not required for toy exit). |
| Drift detector (distribution-drift watch) | OPEN → Phase 8 | `docs/ROADMAP.md` lists drift detection under Phase 8 continuous-evolution. |

### Verification (reproducible)

```
$ python tools/self_improve.py --steps 150 --semver 0.9.0
[intake] 10 validated experiences (deduped)
[train ] 150 steps, final CE 2.6436, checkpoints/candidate/final.npz
[eval  ] target_nova_loss      base=5.155 cand=4.380 delta=-0.775
[eval  ] regression_astra_loss base=2.832 cand=2.840 delta=+0.009
[gate  ] target   gain=+0.775 (need >= 0.1) => PASS
[gate  ] regress  delta=+0.009 (allow <= 0.1) => PASS
[decision] ACCEPT
[promote] astra-name -> 5644b997ac40 (semver 0.9.0)

$ python tools/self_improve.py --rollback-drill
[drill  ] current active: 5644b997ac40
[drill  ] promoted regressing candidate 88e5bc539273
[drill  ] rolled back to 5644b997ac40 (OK)
```

- Gate matrix: target **−0.775 CE** (PASS, min 0.1), regression **+0.009 CE**
  (PASS, max 0.1) → **ACCEPT** → promoted `astra-name` 0.9.0.
- Rollback drill: promoted a regressing artifact then restored the prior active
  sha; audit rows written for both the promote and the rollback.
- Tests: `tests/learning/` now 25 tests (intake/store, candidate trainer,
  gates/audit/registry, evaluate); full suite 143 green; ruff clean.

### Exit criteria (Phase 6)
- [x] The loop runs unattended for a supervised evaluation window
      (`tools/self_improve.py`, all gates deterministic).
- [x] Any regression can trigger rollback (`--rollback-drill`, verified).
- [x] Every promotion has a full audit trail (`learning/audit.jsonl`).
- [~] Astra 0.9 released — release notes drafted, not yet tagged/committed
      (this commit).
- [ ] Autonomous *unattended* deployment (worker that self-promotes on a
      schedule) — Phase 6.5 / supervised-only until Phase 7.

---

## Upcoming phase: Phase 7 — First Stable System (Astra 1.0.0)

### Objective
Integrate all subsystems (tokenizer → model → memory → learning → gates →
registry) into a single, gated, auditable release. (ROADMAP § Phase 7)

### Required components (not yet built)

| Component | Notes |
|---|---|
| **Full E2E test suite** | Tokenizer → training → memory → learning → promotion → rollback as one end-to-end script |
| **All engines integrated** | `memory`, `learning`, `registry`, `service/inference` under one supervised operator command |
| **Release checklist automation** | `make release`-style target running gates + audit + registry + CHANGELOG |
| **Interface freeze** | Public module APIs frozen for 1.0 |
| **Long supervised window** | Continuous-improvement run > 1 week with zero silent regressions |

### Phase-7 precedent (what Phase 6 already ships that Phase 7 builds on)

- Deterministic gate engine + audit trail + registry promotion/rollback.
- Supervised loop driver (`tools/self_improve.py`).
- Full memory + learning engines with test coverage.

### Exit criteria (Phase 7)
- Full E2E test suite green.
- A supervised continuous-improvement run over > 1 week with zero silent
  regressions.
- Complete document set.

---

## Historical phases (verify prior milestones)

| Phase | Version | Status |
|---|---|---|
| 0–1 Research/Birth | 0.0.1 / 0.1.0 | validated (`experiments/phase0/REPORT.md`) |
| 2 Training Foundation | 0.2.0 | validated |
| 3 Stable Neural Core | 0.3.0 / 0.4.0 | validated |
| 4 Memory | 0.5.0 | validated (G1–G6 closed) |
| 5 Learning | 0.7.0 | validated (core loop + demo) |
| **6 Self-Improvement** | **0.9.0** | **in progress (core done)** |
| 7 First Stable System | 1.0.0 | upcoming |
| 8 Continuous Evolution | 1.x | planned |
| 9 Advanced Architecture | 2.0.0 | research |
# Astra Versioning

> Semantic versioning rules for Astra releases, components, and research checkpoints.

**STATUS: VALIDATED (as policy)**

---

## 1. Version Scheme

Astra uses SemVer-style three-part versioning:

```
<major>.<minor>.<patch>
```

Example sequence mapped to roadmap phases:

| Version | Phase | Meaning |
|---|---|---|
| Astra 0.0.1 | 0 — Research | Research foundation |
| Astra 0.1.0 | 1 — Minimal LM | Working small model |
| Astra 0.2.0 | 2 — Training Foundation | Reproducible pipeline |
| Astra 0.3.0 | 3 — Stable Neural Core | Released core model |
| Astra 0.5.0 | 4 — Memory | Memory system |
| Astra 0.7.0 | 5 — Learning | Feedback-driven learning |
| Astra 0.9.0 | 6 — Self-Improvement | Continuous improvement |
| Astra 1.0.0 | 7 — First Stable System | Integrated stable system |
| Astra 1.1.0 | 8 — Continuous Evolution | 1.x evolution |
| Astra 2.0.0 | 9 — Advanced Architecture | Major architecture change |

Version labels are attached to:
- The **repository/package** version (this SemVer).
- The **model** artifact version (matching SemVer in drop 0.x; registered independently once registry exists).
- The **tokenizer**, **dataset manifests**, **configs**, and **memory schema** — each versioned, though not necessarily in lockstep.

---

## 2. Version Component Definitions

### Patch (X.Y.**Z**)
Applies when changes are backward-compatible and fix defects:
- Bug fixes with no API/behavior contract change.
- Documentation corrections, CI-only changes, test improvements.
- **Does not:** change model checkpoints' semantics or training a new release model.

### Minor (X.**Y**.0)
Adds functionality in a backward-compatible way within a phase family:
- New feature or subsystem (e.g., memory engine at 0.5.0) with preservation of existing docs/interfaces.
- New benchmarks, new training features that do not break active configs.
- Model-checkpoint behavior changes that are additive.

### Major (**X**.0.0)
Backward-incompatible or architectural change:
- Removing/replacing a module interface.
- New core architecture (Phase 9 → 2.0.0).
- Breaking change to the artifact format or registry semantics.

### Experimental Release
Unstable/feature-flagged releases. Tagged with suffix `-experimental.<n>` or `-rc<N>` (release candidate). Experimental releases must converge to a formal release or be withdrawn with a note.

### Research Checkpoint
A non-release numbering for training snapshots, not SemVer bump material:
- Format: `rcp-<model>-<family>-<date>-<commit>` (e.g., `rcp-100m-pretrain-20260906-a1b2c3`).
- Research checkpoints are never "released"; they exist for study and reproduction only.

---

## 3. What Constitutes a Release vs. a Snapshot

| Kind | Tagged? | Gates? | CHANGELOG? | Purpose |
|---|---|---|---|---|
| **Release** | Yes (SemVer) | Full gate suite | Yes | End-user / research-stable consumption |
| **Research checkpoint** | No | None required | No | Experiment reproduction |
| **Experimental release** | Yes (rc/experimental suffix) | Subset gates | Yes | Preview feedback |

---

## 4. Rules

- Version bumps only by a maintainer via the release process (`docs/RELEASES.md`).
- Every released artifact has a recorded checksum and eval report.
- A checked-out commit must be reproducible (commit hash versioned).
- Breaking changes always bump major; never disguise a breaking change as a patch.

---

## 5. Artifact Version Record (per registry entry)

Each registered model/artifact entry stores: `semver`, `commit`, `config_id`, `data_manifest_id`, `checklist_id`, `eval_report_id`, `sha256`. This uniquely identifies and reproduces any released state.
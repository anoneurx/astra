# Astra Safety System

> Safety mechanisms for a continuously learning AI.

**STATUS: PROPOSED**

---

## 1. Safety Principles

1. **Every data path is validated.** No corpus/bin enters training without a manifest, hash, and filter report.
2. **Every model change is gated and auditable.** No silent promotion; no in-place modification.
3. **Every learning step can be reverted.** Rollback is a first-class operation, drilled in CI.
4. **Uncertainty is handled, not hidden.** Low-confidence feedback is quarantined; conflicts surface.

---

## 2. Safety Components

### 2.1 Data Validation

- Manifests must include provenance, license, hash, cleaning/filter rule IDs.
- Checksum verification on every dataset read at training start.

### 2.2 Training-Data Filtering

- Safety filter set: PII redaction, toxicity scoring, private-data scanning, language-lock per corpus policy.
- Filters are deterministic, versioned, and audited. No black-box silent filtering.

### 2.3 Prompt-Injection Resistance

- Inference pipeline runs injection/indirect-prompt probes on every release candidate (suite `core-safety`).
- Memory retrieval content is treated as untrusted data: bounded token budget, delimited zones, no code-running during eval, and injection-testing for memory-retrieval contexts.

### 2.4 Malicious-Data Detection

- Contamination scanning (hash + n-gram overlaps).
- Feedback poisoning detection: source verification, cross-checking, rate limits, anomaly detection on feedback clusters.
- Dataset provenance review for external corpora.

### 2.5 Model Rollback

- Registry keeps previous versions with checksums; rollback restores prior active model artifact + config + memory pointer state (if memory changed).
- Rollback drill runs in CI (automated, measured RTO).

### 2.6 Version Control

- Models, memory schema, tokenizers, datasets, configs — all versioned (git for text, hash/S3 for binaries).
- Semantic versioning rules in `docs/VERSIONING.md`.

### 2.7 Evaluation Gates

- Candidates must pass the full evaluation suite prior to promotion; gate engine verdict recorded immutably.
- Thresholds are set by measurement, then documented; changing thresholds requires a PR with rationale.

### 2.8 Human Approval Options

- Deployment modes: `auto` (all gates auto), `manual` (human review required for any model promotion), `hybrid` (auto for benign thresholds, manual for sensitive/risk-flagged).
- Default for Phase 6+: `hybrid`.

### 2.9 Audit Logs

- Append-only audit trail covering: dataset writes, training launches, candidate creations, evaluation runs, gate decisions, promotions, rollbacks, feedback-influence steps, memory corrections.
- Audit record fields: timestamp, actor (system/human/agent), component, before/after hashes, reason, policy version.

---

## 3. Failure Modes of a Self-Learning System (Defense Matrix)

| Failure mode | Mechanism | Defense |
|---|---|---|
| **Data poisoning** | Malicious documents injected into training corpus alter behavior | Provenance checks, filters, contamination scans, quarantine, monitoring behavior probes |
| **Feedback poisoning** | Malicious or adversarial user feedback steers learning | Trust tiers, verification, aggregation, anomaly detection, human review for flashpoints |
| **Model collapse** | Self-generated data loops degrade diversity/capability over generations | Diversity metrics on experiences, CPU-size replay, synthetic-data guardrails, collapse probes in benchmark set |
| **Catastrophic forgetting** | New learning overwrites prior knowledge | Low LR, replay corpus, prior-version regression gates, capability probes per release |
| **Reward hacking** | System finds way to maximize reward signal without real improvement | Gates measure general capability (not reward-max), cross-checked reward, exploration restriction, independent audits |
| **Distribution drift** | World/task distribution shifts enough that model decays relative to current needs | Drift detector on eval slices; alerting → human review; re-training on fresh data; memory refresh |

---

## 4. Safety-Critical Rules

- **No** in-place active-weight modification.
- **No** model promotion without gates.
- **No** silent dataset merge into training.
- **No** automatic trust of raw feedback (see `docs/LEARNING.md` § 3).
- **No** evaluation-data training (see `docs/DATA.md`).

---

## 5. Incident Response

- A reported incident creates an issue; safety-critical incidents trigger retention freeze (no promotions until review).
- Rollback is the default mitigation; root-cause analysis documented before re-promotion.
- Incident records live in `experiments/` as structured markdown with hashes.
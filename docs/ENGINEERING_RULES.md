# Astra Engineering Rules

> The ten non-negotiable project rules (§ 26 of the master specification). These apply to every commit, PR, experiment, and release.

**STATUS: VALIDATED (as policy)**

---

1. **Never hide model limitations.** Limitations are documented in release notes and model cards; hiding them is a release-blocking defect.

2. **Never claim benchmarks that were not actually measured.** Every number in docs, commits, or releases must trace to a recorded run manifest. Unmeasured claims are removed.

3. **Never silently replace model versions.** Version changes require a registry entry, eval report, and changelog entry. In-place silent replacement is forbidden.

4. **Never train on evaluation data.** Eval/benchmark items are quarantined; training on them is a critical policy violation subject to incident response.

5. **Never trust unverified feedback blindly.** Feedback flows through trust tiers and verification before it can influence training (see `docs/LEARNING.md` § 3).

6. **Keep every model version reproducible.** Each registered model is accompanied by config, data manifest, seed, environment, and training record.

7. **Keep rollback capability.** The previous version is always available and restorable; rollback drills run in CI.

8. **Record experiments.** Every experiment writes its manifest and results to `experiments/`. Experiments without recorded symptoms are not part of the record.

9. **Document architectural decisions.** ADRs accompany significant decisions; the master spec is updated in the same PR.

10. **Measure before claiming improvement.** Improvement claims require the benchmark suite and significance analysis — training loss alone never suffices.

---

*These rules are enforced by review, CI checks, and the release gates in `docs/RELEASES.md`, and are incorporated by reference in `docs/ARCHITECTURE.md` § 3 (Development Philosophy).*
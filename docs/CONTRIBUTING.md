# Contributing to Astra

> Guidelines for contributors. Short and practical.

**STATUS: PROPOSED** — refine with maintainer experience.

---

## 1. Code of Conduct

Be respectful, constructive, and evidence-driven. Harassment or trolling is not tolerated. The project values technical rigor over volume.

## 2. Areas of Contribution

- **Research:** experiment design/analysis, benchmark items, literature notes.
- **Engineering:** Python stack, Rust runtime, memory, evaluation, safety.
- **Data work:** dataset manifests, cleaning/filter rules, contamination reports.
- **Documentation:** docs accuracy, ADRs, release notes.
- **Testing:** leaks, poisoning probes, reproduction harness.

## 3. Getting Started

1. Read `docs/ARCHITECTURE.md` and `docs/DEVELOPMENT.md`.
2. Pick a tracked issue labeled `good-first-issue` or propose an experiment.
3. Set up the environment per `docs/INSTALLATION.md`.
4. Branch from `main`, work, and open a PR.

## 4. PR Requirements

Every PR must:
- Reference an issue/experiment where applicable.
- Pass CI: lint, unit tests, Rust tests, doc checks.
- Update docs if behavior changed.
- Include benchmark/measurement evidence for behavior changes (see `docs/DEVELOPMENT.md` § 2).
- Not introduce trailing whitespace or unrelated edits.

Review flow: one approving maintainer review required; model-affecting changes require a second review from the area owner.

## 5. Data & Model Safety Rules

- Never commit dataset content to git; commit manifests and hashes only.
- Never commit checkpoints or secrets to git.
- New benchmark items must be added as quarantined splits (never trained on).
- All external data requires provenance + license in the dataset manifest.

## 6. Release Process

Releases follow `docs/RELEASES.md` + `docs/VERSIONING.md`. Contributors do not tag releases directly; maintainers cut them.

## 7. Commit Messages

Concise, imperative mood. Reference issue numbers where applicable. Example:

```
training: add configurable warmup fraction

Closes #142
```
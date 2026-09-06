# Astra

**Astra** is an independent AI research project building a self-learning, continuously improving language model from first principles.

Astra is not a wrapper around any commercial AI API. It is an open-source research codebase spanning tokenization, model architecture, training, memory, learning, evaluation, and safety — designed with the long-term objective of creating an AI system that observes, learns, verifies, and improves in a controlled and auditable way.

> **Note on claims:** This project makes no claims of AGI, consciousness, sentience, or human-level intelligence. Any capability described in these documents is a research objective, not a guarantee.

---

## Project Identity

- **Vision:** A machine-learning system that becomes measurably more capable over time through verified experience, without discarding what it already knows.
- **Mission:** Build a reproducible, modular, openly documented AI research stack that supports continuous controlled learning from high-quality data and verified experience.
- **Long-term objective:** An AI system capable of learning from high-quality data, learning from verified experience, retaining useful information, using feedback to improve, generating and evaluating candidate model versions, deploying only verified improvements, and rolling back safely.
- **Core principles:** Reproducibility, transparency, controlled continuous learning, evaluation before deployment, safety and rollback.

Full definitions, philosophy, and scope are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

---

## Repository Layout

```
astra/
├── docs/            Documentation (this spec hierarchy)
├── configs/         Experiment and model configuration files
├── datasets/        Dataset definitions, register of sources
├── tokenizer/       Tokenizer training and inference
├── model/           Model definitions (PyTorch)
├── training/        Training loops, optimizers, schedulers
├── inference/       Forward-pass and generation code
├── memory/          Memory engine
├── learning/        Learning system, candidate generation
├── evaluation/      Evaluation harness and benchmark suite
├── safety/          Data validation, gates, audit
├── runtime/         Rust runtime and bindings
├── rust/            Rust crate source
├── python/          Python package source
├── tests/           Unit and integration tests
├── benchmarks/      Benchmark definitions and results
├── experiments/     Experiment logs and research notes
├── checkpoints/     Model checkpoints (git-ignored)
└── tools/           Dev and CI utilities
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) § Repository Structure for full details.

---

## Quick Start

See [docs/INSTALLATION.md](docs/INSTALLATION.md) for prerequisites and setup.

```bash
# Python research stack
python -m venv .venv && source .venv/bin/activate
pip install -e python

# Rust runtime
cargo build --release -p astra-runtime

# Run the test suite
pytest tests/
```

---

## Status

- **Current phase:** Phase 0 — Research
- **Current version:** Astra 0.0.1
- **Architecture status:** Proposed. No component has yet been experimentally validated.

Versioning and phase definitions are in [docs/ROADMAP.md](docs/ROADMAP.md) and [docs/VERSIONING.md](docs/VERSIONING.md).

---

## Documentation Index

| Document | Purpose |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Project identity, scope, core objective, philosophy, overall system architecture |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Official project phases, releases, milestones, exit criteria |
| [docs/MODEL.md](docs/MODEL.md) | Neural architecture specification (Transformer core, configs) |
| [docs/TRAINING.md](docs/TRAINING.md) | Training pipeline: data → optimizer → checkpoint |
| [docs/DATA.md](docs/DATA.md) | Dataset system: raw → cleaned → train/eval/feedback |
| [docs/TOKENIZER.md](docs/TOKENIZER.md) | Tokenizer subsystem specification |
| [docs/MEMORY.md](docs/MEMORY.md) | Memory architecture (short/long/semantic/episodic context) |
| [docs/LEARNING.md](docs/LEARNING.md) | Self-learning loop and feedback system |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Evaluation framework and metrics |
| [docs/SAFETY.md](docs/SAFETY.md) | Safety mechanisms, poisoning, collapse, rollback |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | Internal benchmark suite |
| [docs/HARDWARE.md](docs/HARDWARE.md) | Hardware requirements (consumer → large scale) |
| [docs/INSTALLATION.md](docs/INSTALLATION.md) | Installation and environment setup |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Development methodology |
| [docs/ENGINEERING_RULES.md](docs/ENGINEERING_RULES.md) | The ten non-negotiable project rules |
| [docs/VERSIONING.md](docs/VERSIONING.md) | Semantic versioning, releases, research checkpoints |
| [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) | Contributor guidelines |
| [docs/RESEARCH.md](docs/RESEARCH.md) | Research track and experimental agenda |
| [docs/CHANGELOG.md](docs/CHANGELOG.md) | Changelog |
| [docs/RELEASES.md](docs/RELEASES.md) | Release gates and release process |
| [docs/FAQ.md](docs/FAQ.md) | Frequently asked questions |

---

## Engineering Rules (Abstract)

These rules are non-negotiable and apply to every commit, PR, experiment, and release:

1. Never hide model limitations.
2. Never claim benchmarks that were not actually measured.
3. Never silently replace model versions.
4. Never train on evaluation data.
5. Never trust unverified feedback blindly.
6. Keep every model version reproducible.
7. Keep rollback capability.
8. Record experiments.
9. Document architectural decisions.
10. Measure before claiming improvement.

Full definitions in [docs/ENGINEERING_RULES.md](docs/ENGINEERING_RULES.md).

---

## License

**STATUS: PROPOSED** — License decision pending. An open license (e.g., Apache-2.0) is recommended.

---

## Contact & Contribution

Contributions are welcome. See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).
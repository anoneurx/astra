# Astra Installation

> Environment setup for development, training, evaluation, and the Rust runtime.

**STATUS: PROPOSED** — commands reflect the intended layout; refine as build changes land.

---

## 1. Prerequisites

| Tool | Version (minimum) | Notes |
|---|---|---|
| Python | 3.11+ | CPython recommended |
| pip / uv | latest | |
| Rust toolchain | 1.7x+ | via rustup |
| CUDA | 12.x for NVIDIA | ROCm optional for AMD |
| Git | latest | |
| (optional) Docker | latest | for reproducible CI builds |

---

## 2. Clone & Layout

```bash
git clone <project-url> astra && cd astra
```

The repository already contains the layout described in `docs/ARCHITECTURE.md` § 7.

---

## 3. Python Research Stack

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e python        # astra-py package and bindings
pip install -r requirements-dev.txt   # test/lint tooling (create in Phase 1)
```

Typical core deps (to be pinned in `requirements*.txt`): `torch`, `numpy`, `tokenizers`, `datasets`, `matplotlib`, `pytest`, `maturin`. Exact pins recorded in the repo.

## 4. Rust Runtime

```bash
cargo build --release -p astra-runtime
cargo build --release -p astra-memory
```

Python ↔ Rust bridges built via `maturin develop` (dev) or bundled wheels (CI).

## 5. Verify Install

```bash
python -c "import astra; print(astra.__version__)"
pytest tests/            # unit suite
cargo test --workspace   # Rust tests
```

## 6. Data Setup

- Tokenizer artifacts produced by `tokenizer/` scripts; stored under `tokenizer/artifacts/`.
- Datasets are **not** committed to git; download or generate per manifest in `datasets/*.json`, then point the config at the local path.
- Checkpoints live under `checkpoints/` (git-ignored by default).

## 7. Environment Variables (expected)

| Variable | Use |
|---|---|
| `ASTRA_DATASETS` | root dir for dataset stores |
| `ASTRA_CHECKPOINTS` | root dir for checkpoints |
| `ASTRA_MEMORY_STORE` | memory DB location (Phase 4+) |
| `ASTRA_LOG_LEVEL` | logging level |

Defaults are set in `configs/`; documented in `docs/DEVELOPMENT.md`.

## 8. Troubleshooting

- **CUDNN/CUDA mismatch:** ensure driver ≥ runtime; record both in run manifests.
- **maturin build ordering:** build Rust crates first if `astra` import fails on bindings.
- **Data path not found:** set `ASTRA_DATASETS` or create local mirror; see `configs/`.

## 9. Optional: reproducible environment

- `Dockerfile` and `Makefile` will be shipped in later phases for pinned build reproduction. Package pinning policy is set by the DEV/CI requirements files.
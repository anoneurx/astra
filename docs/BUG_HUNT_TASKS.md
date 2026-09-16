# Bug-Hunt Tasks (guard shift 2026-09-16)

"Labs" = the repo's tool/eval/experiment scripts. Small tasks below identify
concrete flaws. Steps to complete each: read the file, reproduce, fix (if
asked), re-run its check.

## Task 1 — `tools/eval_prose.py`

**Bugs found:**

1. Docstring (line 6) points at `checkpoints/astra5m_prose/final.npz`, but the
   runner exports to `checkpoints/astra5m_prose/resumed/final.npz`. Stale path.
2. Line 56 sampling is stochastically coupled:
   `rng.integers(0, max(1, len(val_ids) - rng.integers(8, 64)), size=samples)`
   - the upper bound is **one** random draw shared by all samples,
   - reused `rng` state also feeds `decode(rng=rng)` downstream,
   - degenerate when `len(val_ids) < 64` (all starts forced to 0).
3. `word_like` compares against the *validation* word set, so sentences the
   model was legitimately trained to reproduce lower the score — metric bias.

**Steps:** run it against `checkpoints/astra5m_prose/resumed/final.npz` with
`configs/astra5m_prose.json` to confirm it runs; then fix the coupling (draw a
per-sample high or use a bounded slice) and refresh the docstring path.

**Add-on finding:** eval_prose "hangs" while training is running — the trainer
pins 6+ cores (OpenBLAS), so eval is CPU-starved and can exceed 5 min. Run it
after training completes, or pin `OPENBLAS_NUM_THREADS=2`.

## Task 2 — `tools/ablate.py`

**Bugs found:**

1. Line 112-113: `(out_root / name).with_suffix(".json").parent.mkdir(...)` is a
   dead mkdir (creates nothing useful) and its docstring claims baseline==variant
   control; verify uniqueness of the seed/`train_config` (`tr` mutated once by
   `--steps` then shared across variants — fine, but confirm same data/seed).
2. `results[label]["config"] = {**cfg.to_dict()}` — ok, but `params` is read
   from `report.manifest["params"]`; if the manifest ever drops that key this
   crashes. Verify current run report has it (it does today).

**Steps:** re-run `python tools/ablate.py --steps 400` in a scratch dir, confirm
`experiments/ablations/index.json` regenerates and both baseline/variant rows fill.

## Task 3 — `inference/generate.py` auto-resolve

**Bugs found:**

1. `auto_resolve()` prefers the local `final.npz` unconditionally (priority
   `10**9`); if that file is corrupt (e.g., interrupted export) chat breaks even
   when good drive snapshots exist. Add a size/sanity check before trusting it.
2. `_missing_warn` still says `--steps 900` (stale; the current horizon is 1800
   and climbing via the runner's auto-extend).

**Steps:** temporarily move `final.npz` aside, confirm `make generate` falls back
to the drive checkpoint; restore; fix sizes.

## Task 4 — `chunk_runner_prose.py` (drive)

**Bugs found/robustness notes:**

1. `next_stage()` counts `stage*` dirs, so an empty/stale stage dir inflates the
   counter and skips numbers on resume (harmless, just cosmetic).
2. After the runner writes `final.npz` at each extension boundary, the repo copy
   is 66 MB per bump — the drive has room; no action needed.
3. The corrupt-checkpoint skip (size>0 + manifest fallback) is in place; force a
   test by truncating a `checkpoint-*.npz` and confirming the runner ignores it.

**Steps:** truncate the highest checkpoint's `.npz` to 5 bytes, restart the
service, confirm it resumes from the next-lowest valid step, restore.

## Priority plan

1. eval_prose (used for model quality reporting once prose finishes).
2. generate.py fallback hardening (protects `make generate` for demo).
3. ablate regeneration (nice-to-have).
4. runner robustness check (already guarded, just prove it).

Guard status at handoff: `astra-prose.service` active, resumed 1095→5400 with
auto-extend; drive mounted; export at every boundary.
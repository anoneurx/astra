"""Experiment store (docs/ROADMAP.md Phase 2: "experiment store queryable").

Zero-dependency, self-hosted alternative to W&B/TensorBoard (ADR-0002): every
completed training run is recorded as one JSON document in the store directory
and is queryable by manifest/metric fields.
"""

from astra.experiments.store import ExperimentStore

__all__ = ["ExperimentStore"]
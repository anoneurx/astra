"""Shared utilities: hashing, seeding, token stats."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def stable_seed(value: str | int) -> int:
    if isinstance(value, int):
        return value
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16)


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def read_config(path: str) -> dict[str, Any]:
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError("PyYAML required for .yaml configs") from exc
        with open(path) as f:
            return yaml.safe_load(f)
    with open(path) as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def read_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)
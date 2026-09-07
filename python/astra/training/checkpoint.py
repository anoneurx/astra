"""Checkpoint persistence (docs/TRAINING.md § 2.9).

Checkpoints are .npz weight snapshots plus a manifest JSON recording the
environment, config, data hash, and loss history. Resume restores the exact
numeric state.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from astra.model.core import LiteLM, all_params
from astra.training.optim import AdamW, CosineSchedule
from astra.utils import sha256_file, write_json


def save_checkpoint(
    path: str,
    model: LiteLM,
    opt: AdamW,
    schedule: CosineSchedule,
    step: int,
    loss_hist: list[float],
    meta: dict,
) -> str:
    payload = {}
    for name, w, _g in all_params(model):
        payload[f"w:{name}"] = w
    if opt is not None:
        for key in ("m", "v"):
            for name, arr in getattr(opt, key).items():
                payload[f"{key}:{name}"] = arr
    np.savez_compressed(path, **payload)
    manifest = {
        **meta,
        "step": step,
        "opt_t": opt.t if opt is not None else 0,
        "loss_hist": loss_hist,
        "params": model.num_params,
    }
    mpath = str(path).rsplit(".npz", 1)[0] + ".manifest.json"
    write_json(mpath, manifest)
    return sha256_file(path)


def load_checkpoint(path: str, model: LiteLM, opt: AdamW | None, schedule: CosineSchedule | None):
    """Restore weights; returns (step, loss_hist, manifest)."""
    data = np.load(path)
    for name, w, _g in all_params(model):
        w[:] = data[f"w:{name}"]
    step, loss_hist, meta = 0, [], {}
    mpath = str(path).rsplit(".npz", 1)[0] + ".manifest.json"
    if Path(mpath).exists():
        import json

        m = json.loads(Path(mpath).read_text())
        step, loss_hist = m["step"], m["loss_hist"]
        meta = m
        if opt is not None:
            opt.t = m["opt_t"]
            for key in ("m", "v"):
                for name in list(getattr(opt, key)):
                    tkey = f"{key}:{name}"
                    if tkey not in data:
                        continue
                    getattr(opt, key)[name][:] = data[tkey]
    return step, loss_hist, meta
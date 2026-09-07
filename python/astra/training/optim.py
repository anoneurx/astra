"""AdamW optimizer and cosine-with-warmup learning rate schedule.

Straightforward NumPy implementation of docs/TRAINING.md § 3.
"""

from __future__ import annotations

import math

import numpy as np

from astra.model.core import all_params


class CosineSchedule:
    def __init__(self, max_steps: int, warmup_steps: int, peak_lr: float, min_lr: float = 0.0):
        self.max_steps = max(1, max_steps)
        self.warmup_steps = min(warmup_steps, self.max_steps)
        self.peak_lr = peak_lr
        self.min_lr = min_lr

    def lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.peak_lr * (step + 1) / max(1, self.warmup_steps)
        t = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        t = min(1.0, t)
        return self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1 + math.cos(math.pi * t))


class AdamW:
    """Decoupled weight decay; weight decay disabled for norm gamma parameters."""

    def __init__(
        self,
        model,
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.model = model
        self.m: dict[str, np.ndarray] = {}
        self.v: dict[str, np.ndarray] = {}
        self.t = 0

    def step(self, lr: float | None = None) -> None:
        self.t += 1
        lr = self.lr if lr is None else lr
        for name, w, g in all_params(self.model):
            m = self.m.get(name)
            if m is None:
                m = np.zeros_like(w)
                self.m[name] = m
                self.v[name] = np.zeros_like(w)
            v = self.v[name]
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * g * g
            self.m[name], self.v[name] = m, v
            mhat = m / (1 - self.beta1**self.t)
            vhat = v / (1 - self.beta2**self.t)
            upd = mhat / (np.sqrt(vhat) + self.eps)
            if self._decays(name):
                upd = upd + self.wd * w
            w -= lr * upd
        # remember last lr for reporting
        self.last_lr = lr

    @staticmethod
    def _decays(name: str) -> bool:
        return not (name.endswith((".ln1", ".ln2")) or name == "ln_f")


OPTIMIZER_REGISTRY: dict[str, type] = {"adamw": AdamW}
SCHEDULE_REGISTRY: dict[str, type] = {"cosine": CosineSchedule}


def build_optimizer(name: str, model, **kwargs) -> AdamW:
    """Resolve an optimizer by registry key (docs/TRAINING.md § 3)."""
    try:
        cls = OPTIMIZER_REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown optimizer {name!r}; available: {sorted(OPTIMIZER_REGISTRY)}") from None
    return cls(model, **kwargs)


def build_schedule(name: str, **kwargs) -> CosineSchedule:
    """Resolve a learning-rate schedule by registry key."""
    try:
        cls = SCHEDULE_REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown scheduler {name!r}; available: {sorted(SCHEDULE_REGISTRY)}") from None
    return cls(**kwargs)


def grad_norm(model) -> float:
    total = 0.0
    for _name, _w, g in all_params(model):
        total += float(np.sum(g.astype(np.float64) ** 2))
    return float(math.sqrt(total))


def clip_grad_norm(model, max_norm: float) -> float:
    nrm = grad_norm(model)
    if nrm > max_norm:
        scale = max_norm / (nrm + 1e-6)
        for _name, _w, g in all_params(model):
            g *= scale
    return nrm
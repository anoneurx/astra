"""Gate engine v2 (docs/LEARNING.md § 1.10-1.12, docs/ROADMAP.md § Phase 6).

The promotion decision is made by a deterministic gate engine, not by a single
metric. A gate is a named rule over a measured base-vs-candidate delta:

- ``gain``        : candidate must improve by >= ``min_gain`` (target metrics).
- ``no_regress``  : candidate must not drop more than ``max_regress`` (all
  other partitions — capability, safety, resource).
- ``absolute``    : candidate metric must satisfy ``{op, value}`` outright
  (e.g. val_ppl < 5.0), used for core-suite style gates.

Rules are expressed as JSON in a gate suite (``benchmarks/suites/``), each
item:

    {
      "id": "target_nova_loss",
      "type": "gain",
      "metric": "target_nova_loss",
      "min_gain": 0.1,           # delta = base - candidate > min_gain
      "note": "..."
    }

Decision policy (docs/LEARNING.md § 4.4): ``auto`` (promote when gates pass) or
``manual`` (promote only when gates pass AND a human approves). The gate result
and policy are recorded verbatim in the promotion audit trail (audit.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GateItem:
    id: str
    type: str                       # gain | no_regress | absolute
    metric: str
    min_gain: float = 0.0           # for gain
    max_regress: float = 0.0        # for no_regress
    op: str = "<"                   # for absolute
    value: float = 0.0              # for absolute
    note: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> GateItem:
        return cls(
            id=d["id"],
            type=d["type"],
            metric=d["metric"],
            min_gain=float(d.get("min_gain", 0.0)),
            max_regress=float(d.get("max_regress", 0.0)),
            op=d.get("op", "<"),
            value=float(d.get("value", 0.0)),
            note=d.get("note", ""),
        )


@dataclass
class GateResult:
    gate: GateItem
    base: float
    candidate: float
    delta: float                    # candidate - base
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "gate_id": self.gate.id,
            "type": self.gate.type,
            "metric": self.gate.metric,
            "base": self.base,
            "candidate": self.candidate,
            "delta": self.delta,
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass
class PromotionDecision:
    accepted: bool
    approved: bool                  # human approval recorded (manual policy)
    policy: str                     # auto | manual
    results: list[GateResult] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "approved": self.approved,
            "policy": self.policy,
            "summary": self.summary,
            "gates": [r.to_dict() for r in self.results],
        }


class GateEngine:
    """Evaluate base-vs-candidate metric dicts against a gate suite."""

    def __init__(self, gates: list[GateItem], policy: str = "auto"):
        self.gates = gates
        if policy not in ("auto", "manual"):
            raise ValueError("policy must be 'auto' or 'manual'")
        self.policy = policy

    @classmethod
    def from_suite(cls, suite: dict, policy: str = "auto") -> GateEngine:
        return cls([GateItem.from_dict(g) for g in suite.get("gates", [])], policy=policy)

    def evaluate(self, metrics: dict[str, dict], human_approval: bool = False) -> PromotionDecision:
        """``metrics`` = {metric_name: {"base": float, "candidate": float}}."""
        results: list[GateResult] = []
        missing: list[str] = []
        for g in self.gates:
            m = metrics.get(g.metric)
            if m is None:
                missing.append(g.metric)
                continue
            base = float(m["base"])
            cand = float(m["candidate"])
            delta = cand - base
            passed = self._check(g, base, cand, delta)
            results.append(GateResult(g, base, cand, delta, passed, self._describe(g, base, cand, delta)))

        accepted = bool(results) and not missing and all(r.passed for r in results)
        if missing:
            raise ValueError(f"gate suite expects metrics {sorted({g.metric for g in self.gates})}; "
                             f"missing {sorted(missing)}")
        if self.policy == "manual" and accepted:
            accepted = bool(human_approval)
        summary = (
            "ACCEPT" if accepted else ("PENDING_HUMAN" if all(r.passed for r in results) else "REJECT")
        )
        return PromotionDecision(
            accepted=accepted,
            approved=human_approval,
            policy=self.policy,
            results=results,
            summary=summary,
        )

    def _check(self, g: GateItem, base: float, cand: float, delta: float) -> bool:
        if g.type == "gain":
            return (base - cand) >= g.min_gain
        if g.type == "no_regress":
            return delta <= g.max_regress
        if g.type == "absolute":
            if g.op == "<":
                return cand < g.value
            if g.op == ">":
                return cand > g.value
            if g.op == "<=":
                return cand <= g.value
            if g.op == ">=":
                return cand >= g.value
            raise ValueError(f"unknown op {g.op!r}")
        raise ValueError(f"unknown gate type {g.type!r}")

    def _describe(self, g: GateItem, base: float, cand: float, delta: float) -> str:
        if g.type == "gain":
            return f"gain={base - cand:+.3f} (need >= {g.min_gain})"
        if g.type == "no_regress":
            return f"delta={delta:+.3f} (allow <= {g.max_regress})"
        return f"cand={cand:.3f} (need {g.op} {g.value})"
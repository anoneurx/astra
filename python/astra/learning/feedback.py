"""Feedback intake + validation (docs/LEARNING.md § 3).

Raw feedback is *not trusted blindly*. Every record passes a cascade:

1. **Source/authenticity** — known source, bounded confidence.
2. **Trust-tier policy**   — the tier the source maps to (see ``__init__``).
3. **Plausibility**        — fails if it contradicts an existing high-confidence
   memory (caller-supplied check) or does not meet minimum length.
4. **Confidence**          — too-low confidence is quarantined, never trained.
5. **Dedup**               — identical re-submissions collapse (no training
   weight inflation).

Anything failing any stage lands in the *quarantine* queue with a reason and is
reviewed by a human; quarantined feedback never reaches the ExperienceStore.

On success, feedback is *routed* to training examples (§ 1.8) and the resulting
example is committed to the ExperienceStore with provenance (``feedback_id``).
"""

from __future__ import annotations

import time
import uuid

from astra.learning import TRUST_TIERS
from astra.learning.experience import ExperienceStore, make_example

_MIN_CONFIDENCE_TO_TRAIN = 0.4
_MIN_PLAUSIBLE_CHARS = 4

# Trust tier -> minimum confidence to train (docs/LEARNING.md § 3 table).
TIER_MIN_CONFIDENCE = {
    "model_self_score": 0.5,      # lowest-trusted; near-certainty only and never alone
    "user_signal": 0.6,           # medium default; spoofable
    "automated_signal": 0.7,      # deterministic where possible
    "verified_correction": 0.8,   # high after verification
    "human_verification": 0.9,    # highest
}

# Which verification_status values indicate provenance is credible enough.
VERIFIED_STATUSES = {"verified", "human_verified", "automated"}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def make_feedback(
    source: str,
    tier: str,
    text: str,
    *,
    confidence: float,
    verification_status: str,
    context: str = "",
    feedback_id: str | None = None,
) -> dict:
    """Build a raw feedback record with provenance + dedup_key happy-paths."""
    if tier not in TRUST_TIERS:
        raise ValueError(f"unknown trust tier: {tier!r}")
    dk_in = text.strip().lower() + "|" + source
    return {
        "id": feedback_id or uuid.uuid4().hex[:16],
        "source": source,
        "tier": tier,
        "text": text,
        "confidence": float(confidence),
        "verification_status": verification_status,
        "context": context,
        "timestamp": _now(),
        "dedup_key": dk_in,
    }


def validate_feedback(
    fb: dict,
    *,
    store: ExperienceStore | None = None,
    contradiction_check=None,
) -> dict:
    """Run the intake cascade. Returns a result record ending in one of:
    ``accepted`` | ``quarantined`` | ``duplicate``.
    """
    errors: list[str] = []
    reasons: list[str] = []

    tier = fb.get("tier", "")
    conf = float(fb.get("confidence", 0.0))
    status = fb.get("verification_status", "")
    text = (fb.get("text") or "").strip()

    # 1. source/authenticity + trust-tier policy
    if tier not in TRUST_TIERS:
        errors.append("unknown_tier")
    if not fb.get("source"):
        errors.append("missing_source")
    if status not in VERIFIED_STATUSES:
        errors.append(f"unverifiable_status:{status or 'none'}")

    # 2. plausibility — minimum length + optional caller contradiction check
    if len(text) < _MIN_PLAUSIBLE_CHARS:
        errors.append("too_short")
    if contradiction_check is not None:
        try:
            if contradiction_check(text):
                reasons.append("contradicts_high_confidence_memory")
        except TypeError:
            # allow bool-or-string return; treat truthy as a contradiction
            if contradiction_check(text):
                reasons.append("contradicts_high_confidence_memory")

    # 3. confidence threshold per tier
    min_conf = TIER_MIN_CONFIDENCE.get(tier, 1.0)
    if conf < min_conf:
        errors.append(f"low_confidence:{conf:.2f}:{tier}:{min_conf}")

    if errors or reasons:
        return {
            "id": fb.get("id"),
            "result": "quarantined",
            "errors": errors,
            "reasons": reasons,
            "feedback": fb,
        }

    # 4. dedup against existing examples (training weight not inflated)
    if store is not None:
        for ex in store.active():
            if ex["feedback_id"] == fb.get("id") or ex["dedup_key"] == _dk_from_feedback(fb):
                return {
                    "id": fb.get("id"),
                    "result": "duplicate",
                    "reasons": ["already_an_example"],
                    "feedback": fb,
                    "existing_example_id": ex["id"],
                }
    return {"id": fb.get("id"), "result": "accepted", "feedback": fb}


def _dk_from_feedback(fb: dict) -> str:
    from astra.learning.experience import dedup_key

    kind = "fact" if "reward" not in fb.get("context", "") else "preference"
    payload = {"text": fb.get("text", "")}
    return dedup_key(kind, payload)


def route_to_examples(fb: dict, kind: str, payload: dict, store: ExperienceStore) -> dict:
    """Turn *validated* feedback into a training-example and commit it."""
    example = make_example(
        kind,
        payload,
        feedback_id=fb["id"],
        source=fb["source"],
        confidence=fb["confidence"],
        verification_status=fb["verification_status"],
        trust_tier=fb["tier"],
    )
    ex_id, added = store.add(example)
    return {"example_id": ex_id, "added": added, "kind": kind}


def build_sft_payload(fb: dict, accepted_output: str) -> dict:
    """SFT-style (input, accepted output) from feedback."""
    ctx = fb.get("context", "").strip()
    return {"input": ctx, "output": accepted_output}


def build_preference_payload(fb: dict, good: str, bad: str) -> dict:
    """Preference pair (input, good, bad) from compared feedback."""
    return {"input": fb.get("context", "").strip(), "good": good, "bad": bad}
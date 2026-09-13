"""Learning engine (docs/LEARNING.md).

Phase 5 (Astra 0.7) implements the core loop: feedback intake
(``feedback.py``), experience store (``experience.py``), candidate training
off the active checkpoint (``candidate.py``) and the comparison driver
(``tools/learning_loop.py``). The trust tiers below are the policy the intake
cascade must enforce; the automated accept/reject promotion gate arrives in
Phase 6 (self-improvement).
"""

from __future__ import annotations

# Feedback trust tiers (docs/LEARNING.md § 3): lowest at top, verified at bottom.
TRUST_TIERS = [
    "model_self_score",     # lowest-trusted; never used alone for training
    "user_signal",          # spoofable; medium default
    "automated_signal",     # deterministic where possible
    "verified_correction",  # high after verification
    "human_verification",   # highest
]

# Every feedback record that may influence training must carry:
#   source, confidence, verification_status, timestamp, dedup_key
# Unverifiable or low-confidence feedback is quarantined (never autotrusted).

__all__ = ["TRUST_TIERS"]
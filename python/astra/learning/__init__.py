"""Learning engine (docs/LEARNING.md).

Phase 0 provides the interface contract only; the candidate
observe -> feedback -> train -> evaluate -> accept/reject loop arrives in
Phase 5 (Astra 0.7). The trust tiers below are the policy the implementation
must enforce.
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
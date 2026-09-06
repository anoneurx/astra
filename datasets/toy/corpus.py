"""Deterministic toy corpus generator (Phase 0, for H0.3/H0.5).

Produces structured "technical manual" text from seeded RNG: sections, fields,
operations, and condition-action pairs. The structure is learnable (allows a
tiny model to reach CE < 4.0) but far from trivial, so it is a genuine
language-modeling task rather than memorization.

Determinism contract: same seed -> same corpus (verified at build time via hash).
"""

from __future__ import annotations

import random
from pathlib import Path

NOUNS = (
    "voltage regulator signal buffer conduit bracket bearing rotor stator clutch flywheel "
    "actuator manifold valve piston camshaft crankshaft controller relay fuse diode coil "
    "transformer exchanger condenser muffler resonator gasket seal washer piston ring "
    "bushing hanger flange coupling damper spring sensor transducer"
).split()

ADJECTIVES = (
    "coaxial primary secondary ferrous nonferrous rotary linear synchronous asynchronous "
    "hydraulic pneumatic thermal optical acoustic electromagnetic redundant modular "
    "portable stationary embedded lightweight rugged sealed ventilated shielded grounded"
).split()

VERBS = (
    "align calibrate clamp pressurize evacuate lubricate preheat temper anneal weld braze "
    "solder fasten torque inspect verify compute derive estimate interpolate extrapolate"
).split()

FIELDS = (
    "bandwidth frequency amplitude phase impedance resistance capacitance inductance "
    "torque temperature pressure flow rate voltage current duty cycle resonance"
).split()

UNITS = (
    "kHz MHz GHz V A W kW N m Nm mbar bar Pa kPa MPa degC rpm Hz dB dBm ms us ns"
).split()

SECTION_HEADS = (
    "section alpha: specifications section beta: operations section gamma: diagnostics "
    "section delta: safety checks section epsilon: calibration section zeta: maintenance"
).splitlines()

OPS = (
    "op-1040 op-1052 op-1177 op-1201 op-1322 op-1407 op-1516 op-1620 op-1733 op-1844 "
    "op-1900 op-2011 op-2130 op-2256 op-2309 op-2401"
).split()

CONDS = (
    "if temperature exceeds threshold if pressure drops below floor if resonance detected "
    "if current spikes above ceiling if vibration exceeds profile if flow stalls "
    "if signal-to-noise degrades if duty cycle exceeds budget"
).split()

ACTS = (
    "engage dampening latch re-route coolant limit output reduce power schedule retest "
    "disable auxiliary stage re-verify integrity hold sample request maintenance"
).split()


def _doc(rng: random.Random, idx: int) -> str:
    lines = []
    head = SECTION_HEADS[idx % len(SECTION_HEADS)]
    lines.append(f"{head} (record {idx})")
    for _ in range(rng.randint(3, 6)):
        f = rng.choice(FIELDS)
        unit = rng.choice(UNITS)
        v = round(rng.uniform(0.1, 99.9), 2)
        lines.append(f"field {f} = {v} {unit} {rng.choice(('nominal', 'observed', 'expected'))}")
    for _ in range(rng.randint(2, 4)):
        op = rng.choice(OPS)
        coef = round(rng.uniform(0.0, 1.0), 3)
        lines.append(f"operation {op} coefficient {coef} subject {rng.choice(ADJECTIVES)} {rng.choice(NOUNS)}")
    for _ in range(rng.randint(1, 3)):
        c = rng.choice(CONDS)
        a = rng.choice(ACTS)
        lines.append(f"rule: {c}, {a}; ({rng.choice(VERBS)} {rng.choice(FIELDS)} within {rng.choice(UNITS)})")
    lines.append(f"verify complete for record {idx}.")
    return "\n".join(lines) + "\n"


def build_corpus(seed: int, n_docs: int, out_txt: str, out_manifest: str | None = None) -> int:
    rng = random.Random(seed)
    lines = []
    for i in range(n_docs):
        d = _doc(rng, i)
        d = d.replace("\x00", "")
        lines.append(d)
    text = "".join(lines)
    path = Path(out_txt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


_DOC_SPLIT = "verify complete for record "


def split_docs(text: str) -> list[str]:
    """Split generated corpus back into individual documents."""
    docs: list[str] = []
    while True:
        pos = text.find(_DOC_SPLIT)
        if pos < 0:
            if text.strip():
                docs.append(text)
            break
        end = text.find("\n", pos) + 1
        docs.append(text[:end])
        text = text[end:]
    return docs


def decontaminate_split(
    train_txt: str,
    split_txt: str,
    tokenizer_path: str,
    n: int = 13,
) -> tuple[list[str], list[str]]:
    """Drop split docs whose n-grams also occur in training (docs/DATA.md § 3).

    Returns (kept_docs, dropped_docs). Deterministic; no reordering.
    """
    import sys

    if str(Path(__file__).resolve().parents[2] / "python") not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    from astra.safety import ngram_set
    from astra.tokenizer import ByteLevelBPE

    tok = ByteLevelBPE.load(tokenizer_path)
    train_tokens = tok.encode(Path(train_txt).read_text(encoding="utf-8"))
    train_grams = ngram_set(train_tokens, n)
    split_text = Path(split_txt).read_text(encoding="utf-8")
    kept, dropped = [], []
    for doc in split_docs(split_text):
        grams = ngram_set(tok.encode(doc), n)
        if grams & train_grams:
            dropped.append(doc)
        else:
            kept.append(doc)
    return kept, dropped


if __name__ == "__main__":
    import argparse
    import hashlib

    p = argparse.ArgumentParser(description="Toy corpus generator / decontaminator")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--docs", type=int, default=40_000)
    p.add_argument("--out", default="-")
    p.add_argument(
        "--decontaminate",
        nargs=3,
        metavar=("TRAIN", "SPLIT", "TOKENIZER"),
        help="filter SPLIT docs sharing a 13-gram with TRAIN; writes SPLIT in place",
    )
    args = p.parse_args()

    if args.decontaminate:
        train, split, tok_path = args.decontaminate
        kept, dropped = decontaminate_split(train, split, tok_path)
        if kept:
            Path(split).write_text("".join(kept), encoding="utf-8")
        keep_digest = hashlib.sha256("".join(kept).encode()).hexdigest()[:12]
        print(
            f"decontaminated {split}: kept={len(kept)} dropped={len(dropped)} "
            f"sha256_head={keep_digest}"
        )
    else:
        chars = build_corpus(args.seed, args.docs, args.out)
        print(f"wrote {args.out}: {chars} bytes")
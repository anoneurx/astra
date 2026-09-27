"""The salvage converter and the V2 notebook must agree on the data contract.

`tools/salvage_distillation_v1.py` repairs V1 output into the V2 shape. It carries its own
copies of the thresholds and predicates, because it has to run standalone without importing a
notebook. That only stays safe while the copies match `colab/Distliation/*.ipynb`, which is
what this test checks. A silent drift here means salvaged rows get rejected by the notebook's
audit cell, or worse, pass it while violating the intent of a rule.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "colab/Distliation/astra_v1_distillation_v2.ipynb"
SALVAGE = REPO / "tools/salvage_distillation_v1.py"

pytestmark = pytest.mark.skipif(not NOTEBOOK.is_file(), reason="v2 notebook not present")


def _load_salvage():
    spec = importlib.util.spec_from_file_location("salvage_distillation_v1", SALVAGE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _notebook_sources() -> list[str]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


# Constants that define the contract. The notebook sets these in three separate cells
# (generate, audit, materialize), so every cell that mentions one must agree.
# DEDUP_HEADROOM is deliberately absent: it is a generation-time knob (how far past a quota the
# generator may run to absorb duplicate rejections) and has no meaning when repairing a fixed
# file of 1002 rows. The converter dedups without a target to hit.
CONSTS = [
    "MIN_INPUT_CHARS", "MAX_INPUT_CHARS",
    "MIN_ANALYSIS_CHARS", "MAX_ANALYSIS_CHARS",
    "MIN_ACTION_CHARS", "MAX_CONFIDENCE",
    "REPEAT_ANCHOR", "MAX_REPEATS",
    "SEED", "VAL_FRAC", "EVAL_FRAC",
]

VALUES = [
    "normal", "suspicious", "malicious", "unknown",
    "INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL",
]


def _notebook_assignments(names):
    """Every literal assignment to `names` in the notebook's code cells.

    Handles the tuple form the notebook uses for paired constants, e.g.
    `MIN_INPUT_CHARS, MAX_INPUT_CHARS = 120, 480`, and strips trailing comments.
    """
    want = set(names)
    out = {n: set() for n in names}
    for src in _notebook_sources():
        for line in src.splitlines():
            line = line.split("#")[0]
            m = re.match(r"^([A-Z_][A-Z_0-9]*(?:\s*,\s*[A-Z_][A-Z_0-9]*)*)\s*=\s*(.+)$", line)
            if not m:
                continue
            targets = [t.strip() for t in m.group(1).split(",")]
            values = m.group(2).strip()
            if len(targets) != len(values.split(",")):
                continue  # a computed RHS, not a literal pair
            for t, v in zip(targets, values.split(",")):
                if t in want:
                    out[t].add(v.strip())
    return out


@pytest.mark.parametrize("name", CONSTS)
def test_salvage_matches_notebook_constant(name):
    salvage = _load_salvage()
    assert hasattr(salvage, name), f"converter lost {name}"

    found = _notebook_assignments([name])[name]
    assert found, f"{name} is not literally assigned in any notebook code cell"
    for literal in found:
        got = eval(literal, {"__builtins__": {}}, {})
        assert got == getattr(salvage, name), (
            f"{name} = {literal} in the notebook but the converter says "
            f"{getattr(salvage, name)}")


@pytest.mark.parametrize("value", VALUES)
def test_salvage_enums_match_notebook(value):
    salvage = _load_salvage()
    blob = "\n".join(_notebook_sources())
    if value in ("normal", "suspicious", "malicious", "unknown"):
        assert value in salvage.CLASSIFICATIONS
    else:
        assert value in salvage.RISK_LEVELS
    assert '"%s"' % value in blob or "'%s'" % value in blob


def test_required_fields_match():
    salvage = _load_salvage()
    for src in _notebook_sources():
        m = re.search(r"^REQUIRED_FIELDS = \((.*?)\)", src, re.S | re.M)
        if m:
            fields = set(re.findall(r"'([a-z_]+)'", m.group(1)))
            assert fields == set(salvage.REQUIRED_FIELDS)
            return
    pytest.skip("no REQUIRED_FIELDS assignment found")


def test_known_fields_allowlist_is_a_superset():
    """A salvaged record must never carry a field the notebook's audit would reject."""
    salvage = _load_salvage()
    extra = salvage.KNOWN_FIELDS - set(salvage.REQUIRED_FIELDS)
    assert extra, "KNOWN_FIELDS collapsed onto REQUIRED_FIELDS"
    # the provenance the audit demands on every record has to be in the allowlist
    for field in ("id", "dedup_key", "grid_index", "source", "synthetic",
                  "teacher_model", "created_at", "hardness"):
        assert field in salvage.KNOWN_FIELDS


def test_early_warning_classes_match():
    """`early_warning` is repaired as a function of the class; both sides must agree."""
    salvage = _load_salvage()
    blob = "\n".join(_notebook_sources())
    assert "EARLY_WARNING_CLASSIFICATIONS" in blob
    assert set(salvage.EARLY_WARNING_CLASSIFICATIONS) == {"suspicious", "malicious"}
    m = re.search(r"EARLY_WARNING_CLASSIFICATIONS = \((.*?)\)", blob, re.S)
    if m:
        assert set(re.findall(r"'([a-z]+)'", m.group(1))) == set(
            salvage.EARLY_WARNING_CLASSIFICATIONS)


def test_no_op_rule_is_scoped_identically():
    """The converter keeps a no-op target on `normal` rows; the audit must agree.

    If the audit cell reverted to a blanket ban it would report every salvaged normal row as
    a fault, which is how the strict rule quietly deleted the whole normal class.
    """
    blob = "\n".join(_notebook_sources())
    assert "action_is_no_op_on_normal" in blob, (
        "the audit cell no longer reports normal-class no-ops separately; the converter keeps "
        "them, so the audit would now flag salvaged data as broken")
    salvage = _load_salvage()
    assert salvage.keeps_no_op("normal", strict=False) is True
    assert salvage.keeps_no_op("suspicious", strict=False) is False
    assert salvage.keeps_no_op("malicious", strict=False) is False
    assert salvage.keeps_no_op("normal", strict=True) is False

"""The reuse-and-resume helpers in v4_training.ipynb, tested outside Colab.

A 50,000-step run is 10-25 GPU hours. The cell that launches it therefore has to be safe to
re-run after a disconnect, and "safe" has two failure modes that both look like success:

  * it resumes from a stale point and silently retrains hours of finished work, or
  * it picks an older `final.npz` and warm-starts the next stage from the wrong weights.

Neither raises. Both are cheap to get right and invisible until a GPU budget is spent, so the
logic is pinned here. The code under test lives in the notebook rather than in a module
because it has to run in Colab, which means it is read out of the .ipynb and executed.
"""

import json
import os
import time
from pathlib import Path

import pytest

NB = Path(__file__).resolve().parents[1] / "colab" / "Distliation" / "v4_training.ipynb"
CELL_INDEX = 11


MAX_STEPS = 50000  # the real 50k run this notebook is configured for


def _load_helpers(tmp_path, max_steps=MAX_STEPS):
    """Execute the notebook cell up to (not including) the training block."""
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cell = nb["cells"][CELL_INDEX]
    assert cell["cell_type"] == "code", "cell %d is no longer the training cell" % CELL_INDEX
    src = "".join(cell["source"])
    body = src.split("import torch\nif not RUN_TRAIN:")[0]
    # The helpers use os/glob/json from earlier cells, and read MAX_STEPS from the config the
    # previous cell wrote. Stand in for both so the cell runs outside a notebook session.
    body = body.replace(
        "with open('configs/astra_distill_v4_alerts.json', encoding='utf-8') as _fh:\n"
        "    MAX_STEPS = json.load(_fh)['training']['max_steps']",
        "MAX_STEPS = %d" % max_steps)
    ns = {"__name__": "v4_helpers"}
    exec(compile("import os, sys, json, glob, shutil\n" + body, "<v4 cell %d>" % CELL_INDEX,
                 "exec"), ns)
    return ns


def _write_checkpoint(run, relpath, step, optimizer_included, age_s=0):
    """A checkpoint stands in for a multi-GB npz; only the manifest is read."""
    path = run / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a real npz")
    (path.parent / (path.stem + ".manifest.json")).write_text(json.dumps({
        "step": step, "optimizer_included": optimizer_included, "compressed": not optimizer_included,
    }))
    if age_s:
        t = time.time() - age_s
        os.utime(path, (t, t))
    return path


@pytest.fixture
def helpers(tmp_path):
    return _load_helpers(tmp_path)


def _write_resume(run, step, age_s=0):
    return _write_checkpoint(run, "resume-%d.npz" % step, step, True, age_s=age_s)


def test_resume_point_is_found_and_is_the_newest(helpers, tmp_path):
    run = tmp_path / "run"
    _write_resume(run, 10000, age_s=600)
    _write_resume(run, 30000, age_s=60)
    found = helpers["find_resume_point"](str(run))
    assert found.endswith("resume-30000.npz"), found


def test_weights_only_snapshots_are_never_treated_as_resume_points(helpers, tmp_path):
    """The file the trainer writes most often must not be picked as a resume point: the loader
    refuses it, and a 3.3 GB snapshot that cannot continue a run is worse than no file."""
    run = tmp_path / "run"
    _write_checkpoint(run, "checkpoint-40000.npz", 40000, False, age_s=10)
    _write_resume(run, 20000, age_s=900)
    found = helpers["find_resume_point"](str(run))
    assert found.endswith("resume-20000.npz"), found


def test_a_resume_point_at_max_steps_is_ignored(helpers, tmp_path):
    """A point written at the final step has nothing left to continue."""
    run = tmp_path / "run"
    _write_resume(run, MAX_STEPS)
    assert helpers["find_resume_point"](str(run)) is None


def test_a_finished_run_is_not_resumed_because_of_its_leftover_point(helpers, tmp_path):
    """The real 50k trap, and why the guard lives in the cell rather than in the finder.

    A completed run leaves its last resume point behind: resume_every is 10000 and the run is
    50000, so `resume-40000.npz` survives a perfectly successful run and is *below* max_steps,
    so the finder legitimately returns it. Resuming from it would silently retrain the last
    10,000 steps of a finished model. The cell therefore asks `completed_final` first and only
    looks for a resume point when there is no finished checkpoint - so both halves are pinned
    here, because either one alone is wrong.
    """
    run = tmp_path / "run"
    _write_resume(run, 40000)
    _write_checkpoint(run, "final.npz", MAX_STEPS, True)

    assert helpers["find_resume_point"](str(run)).endswith("resume-40000.npz"), (
        "the finder is deliberately dumb: it only filters on the step ceiling")
    assert helpers["completed_final"](str(run), MAX_STEPS) is not None, (
        "this is the check the cell actually branches on, and it is what prevents the re-run")

    # Same tree, no finished checkpoint: now the point is the right thing to continue from.
    (run / "final.npz").unlink()
    assert helpers["completed_final"](str(run), MAX_STEPS) is None
    assert helpers["find_resume_point"](str(run)).endswith("resume-40000.npz")


def test_completed_final_requires_the_target_step(helpers, tmp_path):
    run = tmp_path / "run"
    _write_checkpoint(run, "resumed/final.npz", 50000, True)
    assert helpers["completed_final"](str(run), MAX_STEPS) is not None
    assert helpers["completed_final"](str(run), MAX_STEPS + 10000) is None, (
        "a short run must not be mistaken for a finished one")


def test_completed_final_prefers_the_newest_file_not_the_first_sorted(helpers, tmp_path):
    """The regression this guards: a resumed run writes <out>/resumed/final.npz, so a
    directory can hold an older <out>/final.npz alongside it. Sorting and taking the first
    returns the older model, and the next stage then warm-starts from the wrong weights with
    no error at all."""
    run = tmp_path / "run"
    _write_checkpoint(run, "final.npz", MAX_STEPS, True, age_s=3600)
    newer = _write_checkpoint(run, "resumed/final.npz", MAX_STEPS, True, age_s=1)
    picked = helpers["completed_final"](str(run), MAX_STEPS)
    assert picked == str(newer), 'picked %s' % picked


def test_completed_final_is_none_on_a_fresh_run(helpers, tmp_path):
    assert helpers["completed_final"](str(tmp_path / "nothing"), MAX_STEPS) is None


def test_find_resume_point_tolerates_a_truncated_manifest(helpers, tmp_path):
    """A manifest truncated by a crash mid-write must not take the cell down before training
    starts; the point it describes is unusable either way."""
    run = tmp_path / "run"
    good = _write_resume(run, 10000, age_s=900)
    bad = run / "resume-20000.npz"
    bad.write_bytes(b"npz")
    (run / "resume-20000.manifest.json").write_text("{ this is not json")
    # The unreadable one is the newer of the two, so a naive max() would pick it.
    found = helpers["find_resume_point"](str(run))
    assert found == str(good), found

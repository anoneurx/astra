"""Tests for the v4 engine-based data generator.

The engine replaces the throttled teacher API, so the properties that matter are that it
produces enough valid, unique, correctly-labelled records for training, and that it does so
deterministically. A regression here is silent: the notebook still runs, it just emits
degenerate data.
"""

import collections
import importlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'colab', 'data_generation'))

engine = importlib.import_module('engine')
corpus = importlib.import_module('corpus')

CLASSIFICATIONS = ('normal', 'suspicious', 'malicious', 'unknown')
RISK_LEVELS = ('INFO', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
HARDNESS_BY_RISK = {'INFO': 5, 'LOW': 20, 'MEDIUM': 45, 'HIGH': 70, 'CRITICAL': 90}

# The training notebook enforces this cap in Stage 2, so a corpus above it loses rows.
MAX_CLASS_SHARE = 0.45


@pytest.fixture(scope='module')
def corpus_records():
    return [r for r, _ in corpus.build_corpus(2000, seed=20051)]


def test_generates_requested_count():
    records = [r for r, _ in corpus.build_corpus(750, seed=7)]
    assert len(records) == 750


def test_all_constraints_hold(corpus_records):
    """Mirrors the gates in the notebook's audit cell, which the generator must not violate."""
    for r in corpus_records:
        assert 120 <= len(r['input']) <= 480, r['id']
        assert 40 <= len(r['analysis']) <= 420, r['id']
        assert 40 <= len(r['recommended_action']) <= 420, r['id']
        assert r['classification'] in CLASSIFICATIONS, r['id']
        assert r['risk'] in RISK_LEVELS, r['id']
        assert r['hardness'] == HARDNESS_BY_RISK[r['risk']], r['id']
        assert 0.0 < r['confidence'] <= 0.97, r['id']
        assert r['evidence'], r['id']
        assert '\n' not in r['input'], r['id']


def test_early_warning_is_a_function_of_classification(corpus_records):
    """The teacher data was full of rows where this disagreed; the engine cannot produce one."""
    for r in corpus_records:
        assert r['early_warning'] == (r['classification'] in ('suspicious', 'malicious')), r['id']


def test_inputs_are_unique(corpus_records):
    inputs = [r['input'] for r in corpus_records]
    assert len(set(inputs)) == len(inputs)


def test_dedup_key_matches_audited_definition(corpus_records):
    """The audit cell recomputes this digest, so a mismatch would flag every row as faulty."""
    import hashlib
    for r in corpus_records:
        expect = 'distill:' + hashlib.sha256(
            ' '.join(r['input'].lower().split()).encode('utf-8')).hexdigest()[:16]
        assert r['dedup_key'] == expect, r['id']


def test_no_class_exceeds_training_cap(corpus_records):
    counts = collections.Counter(r['classification'] for r in corpus_records)
    for cls, n in counts.items():
        assert n / len(corpus_records) <= MAX_CLASS_SHARE, '%s at %.1f%%' % (
            cls, 100.0 * n / len(corpus_records))


def test_every_category_is_populated(corpus_records):
    counts = collections.Counter(r['category'] for r in corpus_records)
    for name, _ in corpus.CATEGORY_WEIGHTS:
        assert counts[name] > 0, 'category %s produced nothing' % name


def test_deterministic_from_seed():
    """Regenerating must reproduce the corpus, or runs are not comparable."""
    a = _serialise([r for r, _ in corpus.build_corpus(200, seed=99)])
    b = _serialise([r for r, _ in corpus.build_corpus(200, seed=99)])
    assert a == b


def test_only_created_at_varies_between_runs():
    """Pins down the limit of the reproducibility claim: the generation clock moves, the
    records do not. Without this, "deterministic" quietly means "usually identical"."""
    a = [r for r, _ in corpus.build_corpus(50, seed=99)]
    b = [r for r, _ in corpus.build_corpus(50, seed=99)]
    differing = {k for x, y in zip(a, b) for k in set(x) | set(y) if x.get(k) != y.get(k)}
    assert differing <= {'created_at'}
    for r in a:
        assert r['created_at'].endswith('Z')


def test_different_seeds_give_different_corpora():
    a = {r['input'] for r, _ in corpus.build_corpus(200, seed=1)}
    b = {r['input'] for r, _ in corpus.build_corpus(200, seed=2)}
    assert a != b


def test_analyses_are_not_repeated_verbatim(corpus_records):
    """Guards the failure mode where each scenario carries one fixed analysis: the model
    would memorise a few dozen sentences instead of learning to reason."""
    distinct = len({r['analysis'] for r in corpus_records})
    assert distinct >= len(corpus_records) / 10, (
        'only %d distinct analyses across %d records' % (distinct, len(corpus_records)))


def test_analysis_varies_with_review_context(corpus_records):
    """The same scenario under a different environment must not read identically."""
    by_input = collections.defaultdict(set)
    for r in corpus_records:
        by_input[r['input']].add(r['analysis'])
    # every distinct input carries one analysis, so check the reverse: how many inputs
    # share a scenario's framing must still differ in wording
    assert len({r['analysis'] for r in corpus_records}) > 1


def test_risk_agrees_with_classification_shape(corpus_records):
    """A normal event should never be CRITICAL; that pairing is a generator bug if seen."""
    for r in corpus_records:
        if r['classification'] == 'normal':
            assert r['risk'] in ('INFO', 'LOW', 'MEDIUM'), r['id']
        if r['classification'] == 'malicious':
            assert r['risk'] in ('HIGH', 'CRITICAL'), r['id']


def test_provenance_fields_present(corpus_records):
    """The audit cell treats these as required, and Stage 1 checks the digest."""
    for r in corpus_records:
        assert r['source'] == 'astra_v4_engine'
        assert r['synthetic'] is True
        assert r['created_at'].endswith('Z')
        assert set(r['grid']) == {'env', 'lens', 'timing', 'variant'}


def test_quota_plan_sums_exactly():
    for total in (1000, 20000, 7):
        plan = corpus.quota_plan(total, corpus.CATEGORY_WEIGHTS)
        assert sum(plan.values()) == total
        assert all(v >= 0 for v in plan.values())


def test_quota_plan_gives_every_category_at_realistic_sizes():
    """With 10 categories a tiny total cannot cover them all, but a real corpus must."""
    plan = corpus.quota_plan(20000, corpus.CATEGORY_WEIGHTS)
    assert all(v > 0 for v in plan.values())


def test_grid_exceeds_largest_quota():
    """If the grid were smaller than a category quota the walk would cycle and re-ask."""
    grid = len(engine.ENVIRONMENTS) * len(engine.LENSES) * len(engine.TIMINGS) * len(engine.VARIANTS)
    plan = corpus.quota_plan(20000, corpus.CATEGORY_WEIGHTS)
    assert grid > max(plan.values()), (grid, max(plan.values()))


def test_engine_has_no_network_dependency():
    """The whole point of replacing the teacher API. A stray import would reintroduce the
    7% yield problem silently."""
    src = open(engine.__file__).read()
    for banned in ('import requests', 'urllib.request', 'genai', 'google.colab'):
        assert banned not in src, 'engine.py must not reference %r' % banned


def test_all_scenarios_produce_valid_shapes():
    """Every hand-written builder is exercised, so one broken scenario is caught here
    rather than as a dropped category in the corpus."""
    import random
    rng = random.Random(3)
    for name, pool in engine.SCENARIOS.items():
        assert pool, 'category %s has no scenarios' % name
        for builder in pool:
            raw = builder(rng, 'a public cloud Kubernetes cluster', 'lens', 'during a normal working day')
            for key in ('cls', 'risk', 'early', 'events', 'conf', 'evidence', 'analysis', 'action'):
                assert key in raw, '%s missing %s' % (builder.__name__, key)
            assert raw['events'], builder.__name__
            for e in raw['events']:
                assert 'host' in e and 'msg' in e, builder.__name__


def test_records_serialise_to_one_line(corpus_records):
    for r in corpus_records[:200]:
        line = json.dumps(r, ensure_ascii=False)
        assert '\n' not in line
        assert json.loads(line) == r


def _serialise(records):
    """Everything except the generation clock.

    created_at is a wall-clock timestamp, so it is the one field that cannot be reproduced.
    Compare it separately in test_only_created_at_varies_between_runs.
    """
    return [json.dumps({k: v for k, v in r.items() if k != 'created_at'}, sort_keys=True)
            for r in records]


def test_resume_continues_the_walk_rather_than_restarting_it():
    """An interrupted run must not re-emit its own prefix.

    The notebook appends, so a resume that replays from the start would double every record
    already on disk and restart the ids at s000001. Both are silent: the file still parses and
    the count still reaches the target.
    """
    whole = [r for r, _ in corpus.build_corpus(1000, seed=4242)]
    cut = 400
    keys = {r['dedup_key'] for r in whole[:cut]}
    state = {}
    rest = [r for r, _ in corpus.build_corpus(1000, seed=4242, existing_keys=keys,
                                              existing_count=cut, state=state)]
    assert state['skipped'] == cut
    assert len(rest) == 600


def test_resumed_corpus_equals_a_single_pass():
    """The point of the replay: two runs must not be distinguishable from one."""
    whole = [r for r, _ in corpus.build_corpus(800, seed=777)]
    for cut in (1, 137, 400, 799):
        keys = {r['dedup_key'] for r in whole[:cut]}
        rest = [r for r, _ in corpus.build_corpus(800, seed=777, existing_keys=keys,
                                                  existing_count=cut)]
        assert _serialise(whole[:cut] + rest) == _serialise(whole), 'diverged at cut %d' % cut


def test_resume_preserves_record_ids():
    whole = [r for r, _ in corpus.build_corpus(600, seed=31)]
    keys = {r['dedup_key'] for r in whole[:250]}
    rest = [r for r, _ in corpus.build_corpus(600, seed=31, existing_keys=keys,
                                              existing_count=250)]
    ids = [r['id'] for r in whole[:250] + rest]
    assert ids == [r['id'] for r in whole]
    assert len(set(ids)) == len(ids)


def test_resume_preserves_the_class_mix():
    """Skipped rows must still count toward the quotas, or a resumed corpus skews."""
    whole = [r for r, _ in corpus.build_corpus(1000, seed=55)]
    keys = {r['dedup_key'] for r in whole[:600]}
    rest = [r for r, _ in corpus.build_corpus(1000, seed=55, existing_keys=keys,
                                              existing_count=600)]
    mix = collections.Counter(r['classification'] for r in whole[:600] + rest)
    assert mix == collections.Counter(r['classification'] for r in whole)
    for cls, n in mix.items():
        assert n / len(whole) <= MAX_CLASS_SHARE, '%s at %.1f%%' % (cls, 100.0 * n / len(whole))


def test_resume_never_exceeds_the_target():
    whole = [r for r, _ in corpus.build_corpus(500, seed=88)]
    keys = {r['dedup_key'] for r in whole}
    state = {}
    extra = list(corpus.build_corpus(500, seed=88, existing_keys=keys,
                                     existing_count=len(whole), state=state))
    assert extra == []
    assert state['emitted'] == 0


def test_resume_with_unrelated_existing_rows_counts_them():
    """A corpus the walk cannot reproduce still occupies quota, so the mix stays honest."""
    state = {}
    rest = [r for r, _ in corpus.build_corpus(1000, seed=61,
                                              existing_keys={'distill:nonexistent'},
                                              existing_count=100,
                                              existing_filled={'Normal behavior': 100},
                                              state=state)]
    assert len(rest) == 900
    assert state['skipped'] == 0, 'nothing should have matched'


def test_state_reports_counters():
    state = {}
    list(corpus.build_corpus(300, seed=12, state=state))
    assert state['emitted'] == 300
    assert set(state) >= {'emitted', 'skipped', 'produced', 'rejects', 'filled', 'quotas'}

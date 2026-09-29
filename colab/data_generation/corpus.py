"""Corpus driver: walk the framing grid, fill per-category quotas, write records.jsonl.

Separate from engine.py so the notebook can inline both while the repo can test them
independently. Pure standard library, no network, deterministic from SEED.
"""

import hashlib
import json
import os
import random
import time

from engine import (
    ENVIRONMENTS, LENSES, TIMINGS, VARIANTS, SCENARIOS,
    make_record, validate,
)

# Weights must sum to 1.0. 'Unknown' exists so the model learns that not every alert can be
# resolved from logs alone, which is what stops it confidently mislabelling ambiguity.
CATEGORY_WEIGHTS = [
    ('Normal behavior', 0.13),
    ('Suspicious behavior', 0.13),
    ('Early-warning sequences', 0.13),
    ('False positives', 0.10),
    ('Multi-signal correlation', 0.10),
    ('Authentication', 0.08),
    ('Network', 0.08),
    ('Process behavior', 0.08),
    ('Vulnerability / code security', 0.07),
    ('Unknown', 0.10),
]


def quota_plan(total, weights):
    """Largest-remainder apportionment: integer counts summing to exactly *total*."""
    raw = [(name, w * total) for name, w in weights]
    counts = {name: int(frac) for name, frac in raw}
    order = sorted(range(len(raw)), key=lambda i: (-(raw[i][1] - counts[raw[i][0]]), i))
    for i in order[: total - sum(counts.values())]:
        counts[raw[i][0]] += 1
    assert sum(counts.values()) == total
    return counts


def build_corpus(target, seed=20051, progress_every=2000, on_record=None,
                 existing_keys=None, existing_count=0, existing_filled=None, state=None):
    """Yield (record, index) pairs until the corpus holds *target* records in total.

    Walks the full grid - environment x lens x timing x review variant - so the framing
    never repeats while a category still has quota, and re-draws the entities inside each
    scenario so the same framing yields different concrete logs. Stops early on a full
    walk if the grid is exhausted, which is a loud signal that the corpus is saturated.

    Resuming continues the walk instead of restarting it. The walk is deterministic from
    *seed*, so a resumed run replays the prefix already on disk, skips those rows, and
    carries on from there; a corpus finished in two runs is byte-identical to one built in
    a single run of the same *target*, including the record ids. Skipped rows still count
    toward the category quotas, because they are the rows an uninterrupted run would have
    emitted at those positions.

    *existing_keys* holds the dedup keys already on disk. *existing_count* is how many
    rows are there in total, and *existing_filled* counts their categories for rows the
    walk cannot reproduce (an older corpus with a different seed, for instance) so they
    still count toward the quota. *state*, if given, is filled in with the run's counters.
    """
    rng = random.Random(seed)
    quotas = quota_plan(target, CATEGORY_WEIGHTS)
    filled = {name: 0 for name, _ in CATEGORY_WEIGHTS}
    for name, n in (existing_filled or {}).items():
        if name in filled:
            filled[name] += n
    seen = set()
    existing_keys = existing_keys or set()
    produced = 0
    emitted = 0
    skipped = 0
    rejects = {}
    order = list(ENVIRONMENTS)
    rng.shuffle(order)

    def full():
        return (existing_count + emitted >= target
                or all(filled[n] >= quotas[n] for n, _ in CATEGORY_WEIGHTS))

    # A full pass over the grid, then another with a rotated lens offset, so a category
    # short of quota keeps finding unused framings instead of cycling.
    for sweep in range(64):
        if full():
            break
        envs = order[sweep % len(order):] + order[: sweep % len(order)]
        for env in envs:
            for lens_i in range(len(LENSES)):
                for when_i in range(len(TIMINGS)):
                    for var_i in range(len(VARIANTS)):
                        if full():
                            break
                        short = [n for n, _ in CATEGORY_WEIGHTS if filled[n] < quotas[n]]
                        name = rng.choice(short)
                        lens = LENSES[(lens_i + sweep) % len(LENSES)]
                        when = TIMINGS[(when_i + sweep) % len(TIMINGS)]
                        variant = VARIANTS[(var_i + sweep) % len(VARIANTS)]

                        rec, text = make_record(rng, name, env, lens, when, variant)
                        produced += 1
                        if rec is None:
                            rejects[text] = rejects.get(text, 0) + 1
                            continue
                        digest = hashlib.sha1(text.encode('utf-8')).hexdigest()
                        if digest in seen:
                            rejects['duplicate'] = rejects.get('duplicate', 0) + 1
                            continue
                        seen.add(digest)
                        # The audit cell recomputes dedup_key as this digest over the
                        # whitespace-normalised lowercase input, so the two must agree or
                        # every row is reported as a mismatch.
                        key = 'distill:' + hashlib.sha256(
                            ' '.join(text.lower().split()).encode('utf-8')).hexdigest()[:16]
                        if key in existing_keys:
                            skipped += 1
                            filled[name] += 1
                            continue
                        rec['id'] = 'astra-v4-s%06d' % produced
                        rec['dedup_key'] = key
                        rec['grid'] = {'env': env, 'lens': lens, 'timing': when, 'variant': variant}
                        rec['source'] = 'astra_v4_engine'
                        rec['synthetic'] = True
                        rec['created_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                        filled[name] += 1
                        emitted += 1
                        yield rec, produced
                        if on_record and emitted % progress_every == 0:
                            on_record(rec, filled, quotas, emitted, target, rejects)
                    if full():
                        break
                if full():
                    break
            if full():
                break

    if state is not None:
        state.update(emitted=emitted, skipped=skipped, produced=produced,
                     rejects=dict(rejects), filled=dict(filled), quotas=dict(quotas))

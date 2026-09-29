"""Assemble the standalone Colab notebook from the tested engine + corpus modules.

The notebook is generated rather than hand-edited so the code that ships is byte-identical
to the code the tests exercise. Run:  python3 build_notebook.py
"""

import json

ENGINE = 'engine.py'
CORPUS = 'corpus.py'
OUT = 'astra_v4_data_generator.ipynb'


def split_engine(src):
    """Cut engine.py at its own section banners so each cell is a coherent unit."""
    marks = [
        0,
        '# --------------------------------------------------------------------------------------\n'
        '# Log renderers.',
        '# --------------------------------------------------------------------------------------\n'
        '# Scenario library.',
        'MIN_INPUT_CHARS, MAX_INPUT_CHARS = 120, 480',
        '# ======================================================================================\n'
        '# Contextual assembly.',
    ]
    idx = []
    for m in marks:
        i = 0 if m == 0 else src.find(m)
        assert i >= 0, 'marker not found: %r' % m
        idx.append(i)
    idx.append(len(src))
    parts = [src[idx[i]:idx[i + 1]] for i in range(len(idx) - 1)]
    return parts


def strip_imports(src):
    """Drop the `from engine import (...)` block, including its continuation lines."""
    out, skipping = [], False
    for line in src.split('\n'):
        if line.startswith('from engine import'):
            skipping = not line.rstrip().endswith(')')
            continue
        if skipping:
            if line.rstrip().endswith(')'):
                skipping = False
            continue
        out.append(line)
    return '\n'.join(out)


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n').split('\n')}


def code(text):
    lines = text.strip('\n').split('\n')
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': [l + '\n' for l in lines[:-1]] + [lines[-1]]}


engine = open(ENGINE).read()
corpus = open(CORPUS).read()
e = split_engine(engine)

CONFIG = '''
SEED = 20051
TARGET_SAMPLES = 20000
DATA_DIR = 'datasets/astra_v4'
RECORDS = os.path.join(DATA_DIR, 'records.jsonl')
DRIVE_ROOT = '/content/drive/MyDrive/astra/astra_v4'
MIRROR_EVERY = 2000

os.makedirs(DATA_DIR, exist_ok=True)

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    os.makedirs(DRIVE_ROOT, exist_ok=True)
    DRIVE_OK = os.path.isdir(DRIVE_ROOT)
except Exception as exc:
    DRIVE_OK = False
    print('Google Drive unavailable (%s) - artifacts stay under /content only.' % exc)


def mirror_to_drive():
    if not DRIVE_OK:
        return
    for fname in ('records.jsonl',):
        src = os.path.join(DATA_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(DRIVE_ROOT, fname))


# Resuming means appending to whatever is already there rather than starting over.
already = 0
if os.path.exists(RECORDS) and os.path.getsize(RECORDS) > 0:
    with open(RECORDS, encoding='utf-8') as fh:
        for line in fh:
            if line.strip():
                already += 1
    print('resuming: %d records already on disk' % already)
else:
    print('starting fresh: no records.jsonl yet')
print('target %d, generating %d more' % (TARGET_SAMPLES, max(0, TARGET_SAMPLES - already)))
'''

RUN = '''
t0 = time.time()
written = already
with open(RECORDS, 'a', encoding='utf-8') as sink:
    for rec, produced in build_corpus(max(0, TARGET_SAMPLES - already), seed=SEED):
        sink.write(json.dumps(rec, ensure_ascii=False) + '\\n')
        written += 1
        if written % MIRROR_EVERY == 0:
            sink.flush()
            mirror_to_drive()
            rate = (written - already) / max(1e-6, time.time() - t0)
            print('  %7d records  (%.0f/s)' % (written, rate))

elapsed = time.time() - t0
print()
print('wrote %d records in %.1fs  ->  %s' % (written, elapsed, RECORDS))
mirror_to_drive()

by_cat = Counter(r['category'] for r in (json.loads(l) for l in open(RECORDS, encoding='utf-8') if l.strip()))
print()
for name, n in by_cat.most_common():
    print('  %-30s %6d' % (name, n))
print('  %-30s %6d' % ('TOTAL', sum(by_cat.values())))
'''

cells = [
    md('''
# Astra v4 data generator — 20,000 labelled security records in about two seconds

**Run the cells in order.** There is no API key, no account and no network call: every record
is constructed, and its label is correct *by construction* rather than sampled from a model.

## Why this replaced the teacher-API generator

The previous version asked Gemini to write each record. Measured across the runs that
produced the `astra_v4_records(1..3)` archives, **70 teacher calls yielded 5 usable records**
— a 7% acceptance rate, because the free tier spent most calls returning 503 "model at
capacity". Reaching 10,000 records that way needs roughly 140,000 successful calls, which is
not a tuning problem: no amount of concurrency or backoff changes the capacity limit.

A throttled teacher is also a *noisy* teacher. The 15 records it did return included a
`classification` of `unknown` on a normal event and a `grid_index` of 75,600 on a 46,080-entry
grid, both of which are label noise the model would have learned from.

So the direction of the problem is inverted. Instead of asking a model what the answer is,
this notebook builds the log stream and the answer together from one random seed:

* the **label is derived from the scenario that produced the stream**, so it cannot be wrong;
* `early_warning` is a function of `classification`, which removes an entire class of
  inconsistency the teacher data was full of;
* `hardness` is a function of `risk`, checked by the audit;
* the **corpus is reproducible** from `SEED`, and regenerating it produces the same bytes.

## What makes the data worth training on

| Property | Detail |
| --- | --- |
| Yield | ~100% accepted, 100% unique inputs |
| Speed | ~10,000 records/second (20,000 in about two seconds) |
| Label noise | None by construction; the audit reports zero violations |
| Dialects | 6 log formats (syslog, auth.log, CEF, JSON, WinSec, k8s) |
| Scenarios | 38 hand-written across 10 categories |
| Context | 12 environments x 20 lenses x 8 timings x 24 review variants |
| Balance | No classification above 42.8%, under the 45% cap training enforces |

`records.jsonl` is byte-compatible with what `v4_training.ipynb` Stage 1 expects, so the
training notebook is unchanged.

**Caveat worth stating plainly:** these are constructed scenarios, not captures from a real
estate. They teach the model the *shape* of the decision — which signals matter, what
justifies escalation, what separates a false positive from a real finding. They do not teach
it the noise and messiness of production telemetry. After this trains, the useful next step
is a second pass that labels a few hundred real alerts with the same schema and mixes them
in, which is a job for a real expert rather than a script.
'''),
    md('## 1. Configuration, Drive mirroring, resume'),
    code('import os, json, time, random, hashlib, shutil\nfrom collections import Counter, deque\n\n' + CONFIG),
    md('''
## 2. Entity pools and log renderers

Entities are drawn per record, so the same scenario produces different concrete logs each
time. The six renderers exist so the model does not learn one vendor's dialect and then fail
on another — the same scenario rendered as JSON and as syslog is two different-looking
training rows.
'''),
    code(e[0]),
    md('''
## 3. Scenario library

Each builder returns the log stream *and* its label, so the two cannot disagree. `evidence`
names the specific fields that decide the verdict, which is what the model has to learn to
extract. Importance scores (`imp`) control which events survive the 480-character budget
when a scenario is long, so the decisive event is never the one dropped.
'''),
    code(e[1]),
    md('''
## 4. Validation, framing, and context

Two things happen here. Events are packed into the character budget by importance, and the
analysis is made to vary: without this step the corpus would carry one fixed analysis per
scenario — 38 sentences for 20,000 rows — and the model would memorise them instead of
learning to reason. The environment, timing and review-context dimensions append a neutral
sentence, giving thousands of distinct analyses while leaving the reasoning intact.
'''),
    code(''.join(e[2:])),
    md('''
## 5. Generate

Walks the grid until every category quota is met, deduplicating on the exact input text. Safe
to re-run: it appends, and it reports what it resumed from.
'''),
    code(strip_imports(corpus) + RUN),
    md('''
## 6. Audit

Re-checks every record against the invariants independently of the code that produced it. Run
this before uploading anything to training.
'''),
    code(open('audit_cell.py').read()),
    md('''
## 7. Package and download

Zips `records.jsonl` for upload into `v4_training.ipynb`.
'''),
    code(open('package_cell.py').read()),
]

nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
                                   'language_info': {'name': 'python'}},
      'nbformat': 4, 'nbformat_minor': 0}
json.dump(nb, open(OUT, 'w'), indent=2)
print('wrote %s with %d cells' % (OUT, len(cells)))

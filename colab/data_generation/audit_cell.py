import os, json, hashlib
from collections import Counter

CLASSIFICATIONS = ('normal', 'suspicious', 'malicious', 'unknown')
RISK_LEVELS = ('INFO', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
HARDNESS_BY_RISK = {'INFO': 5, 'LOW': 20, 'MEDIUM': 45, 'HIGH': 70, 'CRITICAL': 90}
REQUIRED_FIELDS = ('input', 'classification', 'risk', 'early_warning',
                   'evidence', 'analysis', 'recommended_action', 'confidence')

# V1 drifted: 3 of 1002 records carried an extra "action" key. Anything outside this set is a
# schema change, not data, and is reported rather than silently carried into the corpus.
KNOWN_FIELDS = frozenset(REQUIRED_FIELDS) | {
    'id', 'dedup_key', 'category', 'grid', 'grid_index', 'source', 'synthetic',
    'teacher_model', 'created_at', 'hardness'}


MIN_INPUT_CHARS, MIN_ANALYSIS_CHARS = 120, 40
# a 1200-character input plus its target comfortably fits and a runaway one provably does not.
# Capping at generation time is far cheaper than discovering 5000 over-long rows in the audit.
MAX_INPUT_CHARS, MAX_ANALYSIS_CHARS = 480, 420
# A teacher that loops will happily emit the same six events fifty times. That is valid JSON,
# passes every enum check and is worse than useless as training data, so reject it: if any
# 60-character span of the row occurs three or more times, the generation degenerated.
# Calibrated against every corpus in datasets/ - zero false positives over 7958 chat blocks.
REPEAT_ANCHOR, MAX_REPEATS = 60, 3
# A flag the model can actually learn: anything actionable is an early warning.
EARLY_WARNING_CLASSIFICATIONS = ('suspicious', 'malicious')
MAX_CONFIDENCE = 0.97
MIN_ACTION_CHARS = 40
# Phrases that assert there is nothing to do. Measured on V1: 122 of 1002 targets were under
# 40 characters and the single most common was "No action required." 93 times.
NO_OP_ACTIONS = ('no action required', 'no action needed', 'none', 'no action',
                 'no further action', 'nothing', 'n/a', 'not applicable')
CATEGORIES = ['Normal behavior', 'Suspicious behavior', 'Early-warning sequences',
              'False positives', 'Multi-signal correlation', 'Authentication',
              'Network', 'Process behavior', 'Vulnerability / code security', 'Unknown']

RECORDS = 'datasets/astra_v4/records.jsonl'
if not (os.path.exists(RECORDS) and os.path.getsize(RECORDS) > 0):
    raise RuntimeError('records.jsonl not found or empty at %s. Run the generation cell '
                       '(step 5) first.' % RECORDS)
records = [json.loads(l) for l in open(RECORDS, encoding='utf-8') if l.strip()]
print('auditing %d records' % len(records))
print()

def is_no_op(action):
    """True when the action asserts that nothing needs to happen.

    V1 emitted "No action required." 93 times, "None." 18 times and several padded variants.
    Training on those teaches the model that the useful answer is sometimes nothing at all.
    """
    body = action.strip().rstrip('.!? ').lower()
    if body in NO_OP_ACTIONS:
        return True
    # also catch a padded form such as "No action required. This is standard telemetry."
    return any(body.startswith(phrase) and len(body) < len(phrase) + 90
               for phrase in NO_OP_ACTIONS)


def is_repetitive(text, anchor=REPEAT_ANCHOR, max_repeats=MAX_REPEATS, window=MAX_INPUT_CHARS):
    """The generator's span test, re-run here against what actually landed on disk."""
    head = text[:window]
    if len(head) < 4 * anchor:
        return False
    for off in (0, len(head) // 2, len(head) - 2 * anchor):
        frag = head[off:off + anchor]
        if len(frag) < anchor or not frag.strip():
            continue
        if head.count(frag) >= max_repeats:
            return True
    return False


faults = Counter()
accepted = Counter()
keys, examples = set(), []
for rec in records:
    rid = rec.get('id', '?')
    faults_missing = [f for f in REQUIRED_FIELDS if f not in rec]
    if faults_missing:
        faults['missing_fields'] += 1
    if rec.get('classification') not in CLASSIFICATIONS:
        faults['bad_classification'] += 1
    if rec.get('risk') not in RISK_LEVELS:
        faults['bad_risk'] += 1
    if not 0.0 <= float(rec.get('confidence', -1)) <= 1.0:
        faults['bad_confidence'] += 1
    if not isinstance(rec.get('early_warning'), bool):
        faults['early_warning_not_bool'] += 1
    if rec.get('hardness') != HARDNESS_BY_RISK.get(rec.get('risk')):
        faults['hardness_mismatch'] += 1
    if not isinstance(rec.get('evidence'), list) or not rec['evidence']:
        faults['evidence_empty'] += 1
    if '\n' in str(rec.get('input', '')):
        faults['input_multiline'] += 1
    if len(str(rec.get('input', ''))) < MIN_INPUT_CHARS:
        faults['input_too_short'] += 1
    if len(str(rec.get('input', ''))) > MAX_INPUT_CHARS:
        faults['input_too_long'] += 1
    if is_repetitive(str(rec.get('input', ''))):
        faults['input_repetitive'] += 1
    if len(str(rec.get('analysis', ''))) < MIN_ANALYSIS_CHARS:
        faults['analysis_too_short'] += 1
    if len(str(rec.get('analysis', ''))) > MAX_ANALYSIS_CHARS:
        faults['analysis_too_long'] += 1
    if is_repetitive(str(rec.get('analysis', ''))):
        faults['analysis_repetitive'] += 1
    # A no-op target is only a fault when an action is actually owed. Generated rows never
    # carry one (the prompt bans it), but salvage_v1.py keeps them on `normal` rows, because
    # that is the correct answer there and rejecting them deletes the entire normal class.
    # Counted separately so the deviation is visible rather than silently tolerated.
    if is_no_op(str(rec.get('recommended_action', ''))):
        if rec.get('classification') in EARLY_WARNING_CLASSIFICATIONS:
            faults['action_is_no_op'] += 1
        elif rec.get('classification') == 'normal':
            accepted['action_is_no_op_on_normal'] += 1
    if len(str(rec.get('recommended_action', ''))) < MIN_ACTION_CHARS:
        faults['action_too_short'] += 1
    if bool(rec.get('early_warning')) != (rec.get('classification') in EARLY_WARNING_CLASSIFICATIONS):
        faults['early_warning_inconsistent'] += 1
    if float(rec.get('confidence') or 0) > MAX_CONFIDENCE:
        faults['confidence_at_ceiling'] += 1
    for k in rec:
        if k not in KNOWN_FIELDS:
            faults['unexpected_field:' + k] += 1
    if rec.get('category') not in CATEGORIES:
        faults['unknown_category'] += 1
    if rec.get('synthetic') is not True or not rec.get('source'):
        faults['missing_provenance'] += 1
    expect = 'distill:' + hashlib.sha256(
        ' '.join(str(rec.get('input', '')).lower().split()).encode('utf-8')).hexdigest()[:16]
    if rec.get('dedup_key') != expect:
        faults['dedup_key_mismatch'] += 1
    if rec.get('dedup_key') in keys:
        faults['duplicate_record'] += 1
    keys.add(rec.get('dedup_key'))
    if rec.get('early_warning') is True and rec.get('classification') == 'normal':
        faults['early_warning_on_normal'] += 1
        if len(examples) < 3:
            examples.append(rid)

print('invariant violations:')
if faults:
    for name, n in faults.most_common():
        print('  %-26s %5d  (%.2f%%)' % (name, n, 100.0 * n / max(1, len(records))))
else:
    print('  none - every record satisfies every invariant')
if accepted:
    print()
    print('accepted deviations (reported, not counted as faults):')
    for name, n in accepted.most_common():
        print('  %-26s %5d  (%.2f%%)'
              % (name, n, 100.0 * n / max(1, len(records))))
print()

print('classification :', dict(Counter(r['classification'] for r in records).most_common()))
print('risk           :', dict(Counter(r['risk'] for r in records).most_common()))
print('category       :', dict(Counter(r['category'] for r in records).most_common()))
print()
print('early_warning rate : %.1f%%'
      % (100.0 * sum(1 for r in records if r['early_warning']) / max(1, len(records))))
print('mean confidence    : %.3f'
      % (sum(float(r['confidence']) for r in records) / max(1, len(records))))
print('mean input chars   : %.0f'
      % (sum(len(r['input']) for r in records) / max(1, len(records))))
print('duplicate keys     : %d' % (len(records) - len(keys)))
if examples:
    print()
    print('early_warning=true on a normal row, e.g.:', ', '.join(examples))
    print('these are legitimate (early signal on benign-looking telemetry) but worth eyeballing')
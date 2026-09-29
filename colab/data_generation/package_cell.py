import os, json, zipfile, hashlib

DATA_DIR = 'datasets/astra_v4'
RECORDS = os.path.join(DATA_DIR, 'records.jsonl')
PROGRESS = os.path.join(DATA_DIR, 'progress.json')
ZIP_NAME = 'astra_v4_records.zip'

if not (os.path.exists(RECORDS) and os.path.getsize(RECORDS) > 0):
    raise RuntimeError('records.jsonl not found or empty at %s. Run the generation cell first.' % RECORDS)

count, bad = 0, 0
with open(RECORDS, encoding='utf-8') as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            json.loads(line)
            count += 1
        except json.JSONDecodeError:
            bad += 1
print('validated %d JSONL rows (%d unreadable)' % (count, bad))
if bad:
    print('WARNING: unreadable lines present - do not upload this archive to training yet.')

# progress.json travels with the archive so a later generation run can resume rather
# than re-spend calls on framings it already used.
with zipfile.ZipFile(ZIP_NAME, 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write(RECORDS, 'records.jsonl')
    if os.path.exists(PROGRESS):
        zf.write(PROGRESS, 'progress.json')

digest = hashlib.sha256(open(RECORDS, 'rb').read()).hexdigest()
print('records.jsonl sha256:', digest)
print('%s  %.1f MB' % (ZIP_NAME, os.path.getsize(ZIP_NAME) / 1e6))
print()
print('Upload %s into the v4_training notebook, which unpacks it to datasets/distillation_v3.'
      % ZIP_NAME)
try:
    from google.colab import files
    files.download(ZIP_NAME)
except Exception as exc:
    print('download unavailable (%s) - the archive is still at %s' % (exc, ZIP_NAME))

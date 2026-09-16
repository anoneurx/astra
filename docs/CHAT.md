# Astra Chat — Terminal Guide

## Start Chat

```bash
cd /home/kashie/Documents/Projects/astra && source .venv/bin/activate
```

```bash
make generate
```

With no arguments, `generate.py` auto-picks the best available **language model**:

1. `checkpoints/astra5m_prose/resumed/final.npz` — the finished prose run (copied
   there automatically when training hits 900 steps)
2. the highest-step live snapshot on the training drive
   (`astra_tmp/run_prose_chunk1/stageNN/resumed/checkpoint-*.npz`) — usable mid-run
3. the toy name model (`checkpoints/name/`) — fallback only; it just echoes
   memorized name-facts, which is what produced the garbled output before

The prose run lands at the drive as `astra-prose.service` (self-healing runner:
mount-guard remounts the drive, resumes from the last good checkpoint every time).
When it finishes, `final.npz` is exported into the repo automatically.

## What You See

```
Astra loaded (step 100, 6032640 params)   ← 6.0M prose model, still training
Checkpoint: astra_tmp/run_prose_chunk1/stage02/resumed/checkpoint-100.npz
Temperature: 0.6

You:
```

Type a prompt, press Enter. Astra responds. Repeat. Type `quit`, `exit`, or `Ctrl-C` to leave.

Early-training checkpoints output rough English (fragments, bad spellings) —
expected until ~step 300+. The toy fallback (133k params) is the one that answers
"name"-style prompts with **Astra** but garbles everything else.

## Options

| Flag | Default | Effect |
|---|---|---|
| `--temperature 0.6` | 0.6 | Lower = more deterministic, higher = more random |
| `--max-new 64` | 64 | Max tokens generated per response |
| `--top-k 8` | 8 | Keep only the 8 most-likely tokens per step (0 = off) |
| `--seed 42` | 42 | Changes the randomness seed for generation |
| `--checkpoint PATH` | auto-resolve | Load a specific checkpoint |
| `--config PATH` | auto-resolve | Config/tokenizer for that checkpoint |

### Example with options

```bash
python3 inference/generate.py \
  --checkpoint checkpoints/astra5m_prose/resumed/final.npz \
  --config configs/astra5m_prose.json \
  --temperature 0.8 \
  --max-new 128
```

## Exit

```
You: quit
Goodbye.
```

Or press `Ctrl-C`.
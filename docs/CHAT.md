# Astra 0.1 Chat — Terminal Guide

## Start Chat

```bash
cd /home/kashie/Documents/Projects/astra && source .venv/bin/activate
```

```bash
python3 inference/generate.py \
  --checkpoint checkpoints/phase0/final.npz \
  --config configs/toy_pretrain.json
```

**Short version** (default args work):

```bash
make generate
```

## What You See

```
Astra 0.1 loaded (step 2000, 133440 params)
Checkpoint: checkpoints/phase0/final.npz
Temperature: 0.8

You:
```

Type a prompt, press Enter. Astra responds. Repeat. Type `quit`, `exit`, or `Ctrl-C` to leave.

## Options

| Flag | Default | Effect |
|---|---|---|
| `--temperature 0.5` | 0.8 | Lower = more deterministic, higher = more random |
| `--max-new 256` | 128 | Max tokens generated per response |
| `--seed 123` | 42 | Changes the randomness seed for generation |
| `--checkpoint PATH` | `checkpoints/phase0/final.npz` | Load a different checkpoint |

### Example with options

```bash
python3 inference/generate.py \
  --checkpoint checkpoints/phase0/final.npz \
  --temperature 0.5 \
  --max-new 256
```

## What to Type

Astra 0.1 was trained on a synthetic engineering/scientific corpus. Best prompts use that vocabulary:

```
You: section alpha: specifications
You: field voltage =
You: operation op-100 coefficient
You: verify complete for record
You: temperature within
```

It will also respond to anything, but output quality depends on how close your prompt is to the training data distribution.

## Exit

```
You: quit
Goodbye.
```

Or press `Ctrl-C`.

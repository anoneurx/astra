# Astra Chat — Terminal Guide

## Start Chat (name-tuned model — knows her name is Astra)

```bash
cd /home/kashie/Documents/Projects/astra && source .venv/bin/activate
```

```bash
python3 inference/generate.py \
  --checkpoint checkpoints/name/resumed/final.npz \
  --config configs/toy_name.json
```

**Short version** (default args work — they point at the name-tuned model):

```bash
make generate
```

## What You See

```
Astra loaded (step 2600, 133440 params)
Checkpoint: checkpoints/name/resumed/final.npz
Temperature: 0.6

You:
```

Type a prompt, press Enter. Astra responds. Repeat. Type `quit`, `exit`, or `Ctrl-C` to leave.

## Options

| Flag | Default | Effect |
|---|---|---|
| `--temperature 0.6` | 0.6 | Lower = more deterministic, higher = more random |
| `--max-new 64` | 64 | Max tokens generated per response |
| `--top-k 8` | 8 | Keep only the 8 most-likely tokens per step (0 = off) |
| `--seed 123` | 42 | Changes the randomness seed for generation |
| `--checkpoint PATH` | `checkpoints/name/resumed/final.npz` | Load a different checkpoint |

### Example with options

```bash
python3 inference/generate.py \
  --checkpoint checkpoints/phase0/final.npz \
  --config configs/toy_pretrain.json \
  --temperature 0.8 \
  --max-new 128
```

## What to Type

The **name-tuned model** (default) was fine-tuned from phase0 on short name-focused
sentences (`datasets/name/`). Best prompts mirror that distribution:

```
You: What is your name?
You: My name is
You: Her name is
You: The assistant is called
```

It tends to answer with **Astra** on these, but the model is tiny (133k params), so
sampled responses are short and can drift into the pretraining-corpus tail. The base
model (`checkpoints/phase0/final.npz`) was trained on the toy engineering/scientific
corpus, so prompts in that vocabulary work best there (`section alpha: specifications`,
`field voltage =`, `operation op-100`) and it will not know the name Astra.

## Exit

```
You: quit
Goodbye.
```

Or press `Ctrl-C`.
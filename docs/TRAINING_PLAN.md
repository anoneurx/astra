# Tomorrow's 8-Hour Training Plan (word-level English)

> Status: PLANNED. Latest update 2026-09-18.
> Current state below is the resume baseline; tomorrow starts exactly there.

## Current training state (as of now)

| Phase | Config | Progress | Resume checkpoint | Val ppl (latest) |
|---|---|---|---|---|
| Word prose base | `configs/astra5m_word_prose.json` | **3215 / 5400** | `stage05/resumed/checkpoint-3215.npz` | 58.90 |
| Word chat fine-tune | `configs/astra5m_word_chat.json` | not started | (warm-start from prose final) | — |

Drive: `/dev/sda1` (flaky USB; remount via `udisksctl mount -b /dev/sda1`).
Checkpoint 3215 verified present + manifest valid (`step: 3215`, `params: 8129792`).
Measurements: ~4.8 s/step, checkpoints every 50 steps (crash-safe resume).

## The numbers that force the schedule

| Phase | Steps needed | Time @4.8 s/step |
|---|---|---|
| Prose 3215→5400 | 2185 | ~2.9 h |
| Chat 0→4800 (full) | 4800 | ~6.4 h |
| **Total** | 6985 | **~9.3 h** |

Full plan needs 9.3 h > 8 h budget. Two options given below — pick one.

## Option A — Fit exactly in 8 h (recommended)

Trim chat target to ~3800 steps so everything completes inside the window.
Prose finishes in the morning; chat fine-tune runs ~5.1 h and finishes complete.

| Time slot | Activity | Expectation |
|---|---|---|
| 09:00–09:10 | Mount drive, verify checkpoint-3215, launch runner | resume confirmed |
| 09:10–12:00 | Prose resume 3215→5400 | val_ppl → ~57–58 |
| 12:00–12:15 | Verify prose `final.npz` export | `checkpoints/astra5m_word_prose/resumed/final.npz` |
| 12:15–17:20 | Chat fine-tune, warm-start prose final (`--reset-step`), **max_steps 3800** | val_ppl → chat-quality |
| 17:20–17:30 | Verify chat `final.npz` export | `checkpoints/astra5m_word_chat/resumed/final.npz` |

Start command (prose): `python chunk_runner_word_prose.py` (auto-resumes from 3215).
Start command (chat): `python chunk_runner_word_chat.py` (auto warm-starts prose final).

Config change needed: `astra5m_word_chat.json` `max_steps` 4800 → 3800.

## Option B — Full 9.3 h plan (full chat 4800, runs ~9.5 h)

Same as A but chat keeps `max_steps 4800`; totals ~9.5 h wall (08:45 start → 18:15).

## After both options — check

- `make chat` → Astra speaks with the word-level chat model
- Verify no gibberish (word tokenizer, not byte)
- Resume/commit guards are runner-driven and self-healing on drive drops
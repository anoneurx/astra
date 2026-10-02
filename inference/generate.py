#!/usr/bin/env python3
"""Interactive text generation for Astra.

Chat using the incremental KV-cache decoder (astra.inference.decode) with
top-k + temperature sampling. Each turn starts a fresh cache (the prose
windows are independent), which keeps answers tightly focused on the prompt.

When --checkpoint/--config are omitted, the best available LANGUAGE checkpoint
is auto-selected: the finished prose model in checkpoints/astra5m_prose/ if
present, else the highest-step prose snapshot on the training drive, else the
toy name model (a stand-in that only reproduces memorized name-facts).

Usage:
    python inference/generate.py
    python inference/generate.py --checkpoint checkpoints/name/resumed/final.npz \
        --config configs/toy_name.json
    python inference/generate.py --temperature 0.8
    python inference/generate.py --tools . --chat

Type your prompt and press Enter. Astra responds. Type 'quit' or Ctrl-C to exit.

With --tools DIR, Astra can read files: each reply may contain a tool call, the
result is fed back, and Astra gets another turn to answer with what it found.
File access is confined to DIR.
"""

from __future__ import annotations

import argparse
import glob as _glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import numpy as np
from astra.inference import KVCache, decode
from astra.model import LiteLM, ModelConfig
from astra.tokenizer import load_tokenizer
from astra.training.checkpoint import load_checkpoint
from astra.utils import read_json

TOY_CKPT = "checkpoints/name/resumed/final.npz"
TOY_CFG = "configs/toy_name.json"
PROSE_CFG = "configs/astra5m_prose.json"
LOCAL_FINAL = "checkpoints/astra5m_word_prose/resumed/final.npz"
LOCAL_CFG = "configs/astra5m_word_prose.json"
CHAT_FINAL = "checkpoints/astra5m_prose_chat/resumed/final.npz"
CHAT_CFG = "configs/astra5m_prose_chat.json"
WORD_CHAT_FINAL = "checkpoints/astra5m_word_chat/resumed/final.npz"
WORD_CHAT_CFG = "configs/astra5m_word_chat.json"
RUN_BASE = "astra_tmp/run_prose_chunk1"
SELFLEARN_REGISTRY = ("/run/media/kashie/8cace107-39d5-4713-ac43-f0499e1dd2c0/"
                      "astra_tmp/selflearn/registry/registry.json")

_missing_warn = ("[warn] no trained language model yet — fell back to the toy "
                 "name model. Train with: python training/train.py --config "
                 f"configs/astra5m_prose.json --steps 900 --out {RUN_BASE}")


def _checkpoint_config_matches(path: Path, config_path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        raw = read_json(str(config_path))
        expect = int((raw.get("model") or {}).get("vocab_size", 0))
        if expect <= 0:
            return True
        with np.load(path, mmap_mode="r") as data:
            actual = int(data["w:wte"].shape[0])
        if actual != expect:
            print(f"[warn] skipping incompatible checkpoint {path}: vocab {actual} != config vocab {expect} ({config_path})")
            return False
        return True
    except Exception as exc:
        print(f"[warn] unable to validate checkpoint {path}: {exc}")
        return False


def auto_resolve() -> tuple[str, str]:
    """Return (checkpoint, config) for the best available language model."""
    choices: list[tuple[int, Path, Path]] = []
    # 1st choice: the finished word-level prose model (cleanest, native-English
    # output; supersedes the byte-level BPE models below).
    local = Path(LOCAL_FINAL)
    if _checkpoint_config_matches(local, Path(LOCAL_CFG)):
        choices.append((10**10 + 2, local, Path(LOCAL_CFG)))
    # 2nd choice: a self-learned model promoted by the drive-side daemon.
    sl_reg = Path(SELFLEARN_REGISTRY)
    if sl_reg.exists():
        try:
            data = read_json(SELFLEARN_REGISTRY)
            sha = data.get("active", {}).get("astra-prose")
            if sha:
                rec = data["entries"].get(sha)
                sl_ckpt = Path(rec["path"]) if isinstance(rec, dict) else None
                if _checkpoint_config_matches(sl_ckpt, Path(PROSE_CFG)):
                    choices.append((10**10 + 1, sl_ckpt, Path(PROSE_CFG)))
        except (OSError, KeyError, TypeError, ValueError):
            pass
    for m in _glob.glob(str(Path(RUN_BASE) / "stage*" / "resumed" / "checkpoint-*.npz")):
        p = Path(m)
        if not _checkpoint_config_matches(p, Path(PROSE_CFG)):
            continue
        if p.stat().st_size == 0:
            continue
        stem = p.stem
        try:
            step = int(stem.split("-")[1])
        except (IndexError, ValueError):
            continue
        choices.append((step, p, Path(PROSE_CFG)))
    if not choices:
        return TOY_CKPT, TOY_CFG
    _, ckpt, cfg = max(choices)
    return str(ckpt), str(cfg)


def chat_resolve() -> tuple[str, str]:
    """Return (checkpoint, config) for the best chat model if available.

    Priority: word-level chat fine-tune (cleanest output) > byte-level chat
    fine-tune > best language base model.
    """
    word = Path(WORD_CHAT_FINAL)
    if _checkpoint_config_matches(word, Path(WORD_CHAT_CFG)):
        return str(word), WORD_CHAT_CFG
    chat = Path(CHAT_FINAL)
    if _checkpoint_config_matches(chat, Path(CHAT_CFG)):
        return str(chat), CHAT_CFG
    return auto_resolve()


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive Astra text generation")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rep-penalty", type=float, default=1.15,
                    help="repetition penalty (>1 = suppress repeats in sampled output; 0 = off)")
    ap.add_argument("--chat", action="store_true",
                    help="chat mode: wrap prompts in the You/Astra turn format "
                         "and prefer the chat fine-tune checkpoint")
    ap.add_argument("--memory", default=None, help="memory store to recall from per turn")
    ap.add_argument("--memory-embedder", default="hash", choices=["hash", "litelm"])
    ap.add_argument("--memory-budget-tokens", type=int, default=192)
    ap.add_argument("--memory-k", type=int, default=5)
    ap.add_argument("--tools", default=None, metavar="DIR",
                    help="let Astra read files: enables read_file, search_text and "
                         "find_files confined to DIR (no tool argument can change "
                         "the root)")
    ap.add_argument("--tool-max-steps", type=int, default=4,
                    help="tool round trips allowed per reply")
    ap.add_argument("--tool-max-new", type=int, default=192,
                    help="tokens generated per turn while tools are enabled")
    args = ap.parse_args()

    if not args.checkpoint:
        args.checkpoint, args.config = chat_resolve() if args.chat else auto_resolve()
        if args.checkpoint == TOY_CKPT:
            print(_missing_warn)

    raw = read_json(args.config)
    tok = load_tokenizer(raw["tokenizer"])
    cfg = ModelConfig.from_dict({**raw["model"], "vocab_size": len(tok)})

    model = LiteLM(cfg, seed=0)
    step, _hist, _meta = load_checkpoint(args.checkpoint, model, opt=None, schedule=None)
    memory = None
    if args.memory:
        from astra.memory import HashEmbedder, LiteLMExtractor, MemoryStore, build_memory_block

        embedder = HashEmbedder() if args.memory_embedder == "hash" else LiteLMExtractor(model, tok)
        memory = MemoryStore.open(args.memory, embedder=embedder)
    print(f"Astra loaded (step {step}, {model.num_params} params)")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Temperature: {args.temperature}" + (f" | memory: {args.memory}" if memory else ""))

    rng = np.random.default_rng(args.seed)

    def generate_once(prompt_text: str) -> str:
        """Sample one reply from an already-framed prompt.

        Tool mode hands over the whole transcript, so the You/Astra framing is
        applied by AgentSession rather than here; wrapping it twice would repeat
        the role markers on every round trip.
        """
        ids = tok.encode(prompt_text)
        if len(ids) < 1:
            return ""
        gen_ids = decode(
            model, ids, args.tool_max_new if args.tools else args.max_new,
            temperature=args.temperature,
            rng=rng,
            top_k=args.top_k,
            cache=KVCache(model.cfg),
            rep_penalty=args.rep_penalty,
            forbidden=set(range(tok.num_special)),
        )
        try:
            return tok.decode(gen_ids)
        except (UnicodeDecodeError, KeyError):
            return "".join(chr(b) if 32 <= b < 127 else "." for b in
                           b"".join(tok.id_to_piece.get(i, b"?") for i in gen_ids))

    agent = None
    if args.tools:
        from astra.tools import AgentSession, RootJail, ToolRegistry, make_file_tools

        registry = ToolRegistry(make_file_tools(RootJail(args.tools)))
        agent = AgentSession(
            registry, generate_once,
            max_steps=args.tool_max_steps,
            max_new=args.tool_max_new,
        )
        print(f"Tools: {', '.join(s.name for s in registry.specs)} (root {args.tools})")
    print()

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        text = prompt
        if memory:
            hits = memory.search(query=prompt, k=args.memory_k,
                                 budget_tokens=args.memory_budget_tokens, tokenizer=tok)
            block = build_memory_block(hits, tok, budget_tokens=args.memory_budget_tokens)
            if block.included:
                print(f"  [memory] {len(block.included)} recalled: "
                      + ", ".join(h.record.id for h in block.included))
                text = block.prepend(prompt)
        if agent is not None:
            # AgentSession owns the framing and the transcript from here: it
            # appends each result and regenerates, so this turn can be several
            # model calls rather than one.
            turn = agent.ask(text)
            for call, payload in zip(turn.calls, turn.results):
                print(f"  [tool] {call.name} -> {payload[:200]}")
            if turn.exhausted:
                print(f"  [tool] gave up after {turn.steps} call(s) with no reply yet")
            print(f"Astra: {turn.text}")
            print()
            continue
        if args.chat:
            # Chat format seen at training time: "You: <ask>\nAstra:" then the
            # decoder continues with Astra's reply.
            text = f"You: {text}\nAstra:"
        seed_ids = tok.encode(text)
        if len(seed_ids) < 1:
            continue

        cache = KVCache(model.cfg)
        gen_ids = decode(
            model, seed_ids, args.max_new,
            temperature=args.temperature,
            rng=rng,
            top_k=args.top_k,
            cache=cache,
            rep_penalty=args.rep_penalty,
            forbidden=set(range(tok.num_special)),
        )
        try:
            response = tok.decode(gen_ids)
        except (UnicodeDecodeError, KeyError):
            response = "".join(chr(b) if 32 <= b < 127 else "." for b in
                              b"".join(tok.id_to_piece.get(i, b"?") for i in gen_ids))
        print(f"Astra: {response}")
        print()


if __name__ == "__main__":
    main()
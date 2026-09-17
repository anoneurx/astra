import json
from pathlib import Path

from astra.tokenizer.bpe import ByteLevelBPE
from astra.tokenizer.ids import BOS_ID, EOS_ID, PAD_ID, UNK_ID
from astra.tokenizer.word import WordLevel

TOKENIZER_REGISTRY = {
    "ByteLevelBPE": ByteLevelBPE,
    "WordLevel": WordLevel,
}


def load_tokenizer(path) -> ByteLevelBPE | WordLevel:
    """Load a tokenizer artifact, dispatching on its ``class`` field.

    ``ByteLevelBPE`` artifacts predate the class field, so they default to the
    byte-level implementation.
    """
    payload = json.loads(Path(path).read_text())
    kind = payload.get("class", "ByteLevelBPE")
    cls = TOKENIZER_REGISTRY.get(kind)
    if cls is None:
        raise ValueError(f"unknown tokenizer class {kind!r} in {path}")
    assert cls is not None
    return cls.load(path)


__all__ = [
    "BOS_ID",
    "EOS_ID",
    "PAD_ID",
    "TOKENIZER_REGISTRY",
    "UNK_ID",
    "ByteLevelBPE",
    "WordLevel",
    "load_tokenizer",
]
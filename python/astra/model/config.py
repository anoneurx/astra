"""Model configuration (docs/MODEL.md). Kanonical config resolved from configs/*.json."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 2048
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_head: int = 16
    d_ffn: int = 128
    max_seq_len: int = 64
    rope_theta: float = 10000.0
    eps: float = 1e-6
    tie_embeddings: bool = True
    norm_type: str = "rmsnorm"        # rmsnorm | layernorm
    ffn_type: str = "swiglu"          # swiglu | gelu
    pos_type: str = "rope"            # rope | learned

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.d_head * self.n_heads != self.d_model:
            raise ValueError("d_head * n_heads must equal d_model")
        if self.norm_type not in ("rmsnorm", "layernorm"):
            raise ValueError(f"unknown norm_type {self.norm_type!r}")
        if self.ffn_type not in ("swiglu", "gelu"):
            raise ValueError(f"unknown ffn_type {self.ffn_type!r}")
        if self.pos_type not in ("rope", "learned"):
            raise ValueError(f"unknown pos_type {self.pos_type!r}")

    @property
    def name(self) -> str:
        params_k = (self.vocab_size * self.d_model) / 1000
        return f"astra-toy-{params_k:.0f}k"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
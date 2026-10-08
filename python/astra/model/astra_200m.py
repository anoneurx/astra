import torch
import torch.nn as nn
import math

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config["hidden_size"]
        self.n_heads = config["num_heads"]
        self.n_kv_heads = config["num_kv_heads"]
        self.head_dim = self.d_model // self.n_heads
        
        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Flash attention or scaled dot product
        y = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)

class AstraBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config["hidden_size"])
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config["hidden_size"])
        self.ffn = SwiGLU(config["hidden_size"], config["intermediate_size"])

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x

class Astra200M(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList([AstraBlock(config) for _ in range(config["num_layers"])])
        self.norm = RMSNorm(config["hidden_size"])
        self.head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        
        if config.get("tie_embeddings", True):
            self.head.weight = self.embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        x = self.embed(idx)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    import json
    cfg = json.load(open("configs/astra_200m.json"))["model"]
    model = Astra200M(cfg)
    total = count_parameters(model)
    print(f"Astra 200M")
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {total:,}")
    assert 190_000_000 <= total <= 210_000_000, f"Parameter count {total:,} out of range (190M-210M)"

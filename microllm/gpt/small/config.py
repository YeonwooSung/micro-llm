from dataclasses import dataclass

CONFIG_DICT = {
    'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
    'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
    'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
    'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
}

@dataclass(frozen=True)
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1

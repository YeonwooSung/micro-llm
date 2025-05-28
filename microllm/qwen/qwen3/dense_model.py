import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Union
import math

# custom modules
from microllm.utils.common import RMSNorm, apply_rotary_emb, precompute_freqs_cis


@dataclass
class Qwen3Config:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    use_scaled_rope: bool = False
    max_batch_size: int = 32
    max_seq_len: int = 2048
    flash: bool = False # whether to use flash attention


    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

        assert self.n_kv_heads <= self.n_heads
        assert self.n_heads % self.n_kv_heads == 0
        assert self.dim % self.n_heads == 0


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # model_parallel_size is 1 for 1 GPU
        model_parallel_size = 1

        self.flash = config.flash # use flash attention?
        self.n_kv_heads = config.n_heads if config.n_kv_heads is None else config.n_kv_heads

        self.n_local_heads = config.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.dim // config.n_heads

        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

        # will be KVCache object managed by inference context manager
        self.cache = None


    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor = None
    ):
        bsz, seqlen, _ = x.shape

        # calculate query, key, value and split out heads
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # rotate query, keys (RoPE)
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        # KV cache update
        if self.cache is not None:
            # update the KV cache with current KV and get all the previous KVs
            xk, xv = self.cache.update(start_pos, xk, xv)

        # repeat keys and values for KV cache
        xk = xk.repeat(1, 1, self.n_rep, 1)
        xv = xv.repeat(1, 1, self.n_rep, 1)

        # make heads be a batch dim
        xq, xk, xv = (x.transpose(1, 2) for x in (xq, xk, xv))

        # attention
        if self.flash:
            output = F.scaled_dot_product_attention(xq, xk, xv, mask)
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        # concatenate all the heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # output projection
        proj = self.wo(output)
        return proj


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        # hidden dim gymnastics that Meta simplified only later
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Qwen3Block(nn.Module):
    def __init__(self, config: Qwen3Config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = config

        self.attention = Qwen3Attention(config)
        self.ffn = FeedForward(
            dim=config.dim,
            hidden_dim=config.dim,
            multiple_of=config.multiple_of,
            ffn_dim_multiplier=config.ffn_dim_multiplier
        )
        self.norm1 = RMSNorm(config.dim, eps=config.norm_eps)
        self.norm2 = RMSNorm(config.dim, eps=config.norm_eps)


    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: torch.Tensor = None):
        # attention + normalization with skip connection
        x = x + self.attention(self.norm1(x), start_pos, freqs_cis, mask)

        # feed-forward + normalization with skip connection
        x = x + self.ffn(self.norm2(x))

        return x


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.params = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList(
            [Qwen3Block(config) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=True)

        self.freqs_cis = precompute_freqs_cis(
            config.dim // config.n_heads,
            config.max_seq_len * 2,
            config.rope_theta,
            config.use_scaled_rope,
        )


    def forward(self, input_ids: torch.Tensor, attention_mask: Union[torch.Tensor, None] = None, start_pos: int = 0):
        # for use during inference
        _bsz, seqlen = input_ids.shape
        h = self.embedding(input_ids)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        # attention mask (if not provided)
        if seqlen > 1 and attention_mask is None:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=input_ids.device)
            mask = torch.triu(mask, diagonal=1)
            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            attention_mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=input_ids.device), mask]
            ).type_as(h)

        for block in self.blocks:
            h = block(h, start_pos, freqs_cis, attention_mask)
        h = self.norm(h)
        logits = self.output(h)
        return logits

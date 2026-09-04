from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig


@dataclass
class ModelOutput:
    logits: Tensor
    loss: Tensor | None = None


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


def _rope_frequencies(seq_len: int, head_dim: int, base: float, device: torch.device) -> tuple[Tensor, Tensor]:
    inverse = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    angles = torch.outer(positions, inverse)
    return angles.cos()[None, None, :, :], angles.sin()[None, None, :, :]


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        if self.head_dim % 2:
            raise ValueError("attention head dimension must be even for RoPE")
        self.rope_base = config.rope_base
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def heads(tensor: Tensor) -> Tensor:
            return tensor.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = heads(q), heads(k), heads(v)
        cos, sin = _rope_frequencies(seq_len, self.head_dim, self.rope_base, x.device)
        q = _apply_rope(q, cos.to(q.dtype), sin.to(q.dtype))
        k = _apply_rope(k, cos.to(k.dtype), sin.to(k.dtype))
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.output(attended.transpose(1, 2).contiguous().view(batch, seq_len, width))


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.mlp_hidden_size, bias=False)
        self.value = nn.Linear(config.d_model, config.mlp_hidden_size, bias=False)
        self.output = nn.Linear(config.mlp_hidden_size, config.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.output(F.silu(self.gate(x)) * self.value(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class RecurrentTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.prelude = nn.ModuleList(TransformerBlock(config) for _ in range(config.prelude_layers))
        self.recurrent_core = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.recurrent_core_layers)
        )
        self.coda = nn.ModuleList(TransformerBlock(config) for _ in range(config.coda_layers))
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _run_blocks(self, x: Tensor, blocks: nn.ModuleList) -> Tensor:
        for block in blocks:
            x = block(x)
        return x

    def num_parameters(self, trainable_only: bool = True) -> int:
        unique: dict[int, nn.Parameter] = {}
        for parameter in self.parameters():
            if not trainable_only or parameter.requires_grad:
                unique[id(parameter)] = parameter
        return sum(parameter.numel() for parameter in unique.values())

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        num_recurrences: int | None = None,
    ) -> ModelOutput:
        recurrences = num_recurrences if num_recurrences is not None else self.config.min_recurrences
        if not self.config.min_recurrences <= recurrences <= self.config.max_recurrences:
            raise ValueError(
                f"num_recurrences must be between {self.config.min_recurrences} "
                f"and {self.config.max_recurrences}"
            )
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("input sequence exceeds max_seq_len")

        hidden = self._run_blocks(self.token_embedding(input_ids), self.prelude)
        for _ in range(recurrences):
            hidden = self._run_blocks(hidden, self.recurrent_core)
        hidden = self._run_blocks(hidden, self.coda)
        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must match input_ids shape")
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
                labels[:, 1:].contiguous().view(-1),
            )
        return ModelOutput(logits=logits, loss=loss)

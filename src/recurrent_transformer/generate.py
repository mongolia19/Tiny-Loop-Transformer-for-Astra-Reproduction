from __future__ import annotations

import torch
from torch import Tensor

from .model import RecurrentTransformer


@torch.no_grad()
def generate_ids(
    model: RecurrentTransformer,
    prompt_ids: Tensor,
    *,
    max_new_tokens: int,
    num_recurrences: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    eos_id: int | None = None,
    seed: int = 42,
) -> Tensor:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    device = next(model.parameters()).device
    generated = prompt_ids.to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    for _ in range(max_new_tokens):
        context = generated[:, -model.config.max_seq_len :]
        logits = model(context, num_recurrences=num_recurrences).logits[:, -1]
        if temperature == 0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                if top_k <= 0:
                    raise ValueError("top_k must be positive")
                values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, 1, generator=generator)
        generated = torch.cat((generated, next_token), dim=1)
        if eos_id is not None and bool((next_token == eos_id).all()):
            break
    return generated

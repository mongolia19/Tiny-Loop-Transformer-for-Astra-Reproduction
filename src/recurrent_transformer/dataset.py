from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from .tokenizer import Tokenizer


def pack_documents(documents: Sequence[str], tokenizer: Tokenizer, seq_len: int) -> Tensor:
    if seq_len <= 1:
        raise ValueError("seq_len must exceed one")
    token_ids: list[int] = []
    for document in documents:
        token_ids.extend((tokenizer.bos_id, *tokenizer.encode(document), tokenizer.eos_id))
    usable = len(token_ids) // seq_len * seq_len
    if usable == 0:
        raise ValueError(f"corpus has fewer than {seq_len} tokens")
    return torch.tensor(token_ids[:usable], dtype=torch.long).view(-1, seq_len)

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from contextlib import ExitStack
import os

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


class DiskTokenBatches:
    """Re-openable iterable of token batches backed by an int32 disk file."""

    def __init__(self, path: str | Path | Sequence[Path], *, seq_len: int, batch_size: int) -> None:
        if seq_len <= 1 or batch_size <= 0:
            raise ValueError("seq_len must exceed one and batch_size must be positive")
        self.paths = [Path(path)] if isinstance(path, (str, Path)) else [Path(p) for p in path]
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.sizes = [p.stat().st_size for p in self.paths]
        if any(size % 4 for size in self.sizes):
            raise ValueError("token file contains a partial int32 token")
        self.counts = [size // (4 * seq_len) for size in self.sizes]
        self.next_batch = 0
        if not len(self):
            raise ValueError(f"corpus has fewer than {seq_len} tokens")

    def __len__(self) -> int:
        return (sum(self.counts) + self.batch_size - 1) // self.batch_size

    def state_dict(self) -> dict:
        return {"next_batch": self.next_batch, "sizes": self.sizes,
                "seq_len": self.seq_len, "batch_size": self.batch_size}

    def load_state_dict(self, state: dict) -> None:
        for key in ("sizes", "seq_len", "batch_size"):
            if state[key] != getattr(self, key):
                raise ValueError(f"checkpoint data {key} does not match")
        if not 0 <= state["next_batch"] <= len(self):
            raise ValueError("checkpoint data cursor out of range")
        self.next_batch = state["next_batch"]

    def _locate(self, position: int) -> tuple[int, int]:
        low, high = 0, max(self.counts)
        while low + 1 < high:
            mid = (low + high) // 2
            if sum(min(mid, n) for n in self.counts) <= position:
                low = mid
            else:
                high = mid
        offset = position - sum(min(low, n) for n in self.counts)
        active = [i for i, n in enumerate(self.counts) if n > low]
        return active[offset], low

    def __iter__(self) -> Iterator[Tensor]:
        import numpy as np

        if self.next_batch == len(self):
            self.next_batch = 0
        with ExitStack() as stack:
            handles = [stack.enter_context(path.open("rb")) for path in self.paths]
            for batch_index in range(self.next_batch, len(self)):
                blocks = []
                end = min((batch_index + 1) * self.batch_size, sum(self.counts))
                for pos in range(batch_index * self.batch_size, end):
                    file_index, block_index = self._locate(pos)
                    handle = handles[file_index]
                    handle.seek(block_index * self.seq_len * 4)
                    raw = handle.read(self.seq_len * 4)
                    if len(raw) != self.seq_len * 4:
                        raise ValueError("token file was truncated during training")
                    blocks.append(torch.from_numpy(np.frombuffer(raw, dtype="<i4").astype("int64")))
                self.next_batch = batch_index + 1
                yield torch.stack(blocks)


def pack_documents_to_disk(
    documents: Iterable[str],
    tokenizer: Tokenizer,
    seq_len: int,
    path: str | Path,
) -> DiskTokenBatches:
    """Encode documents directly to a compact on-disk token stream."""
    if seq_len <= 1:
        raise ValueError("seq_len must exceed one")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    import array

    values = array.array("i")
    with temp.open("wb") as handle:
        for document in documents:
            values.extend((tokenizer.bos_id, *tokenizer.encode(document), tokenizer.eos_id))
            if len(values) >= 65536:
                values.tofile(handle)
                values = array.array("i")
        if values:
            values.tofile(handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, target)
    return DiskTokenBatches(target, seq_len=seq_len, batch_size=1)

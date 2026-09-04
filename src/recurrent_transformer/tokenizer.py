from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import sentencepiece as spm


class Tokenizer:
    def __init__(self, model_file: str | Path) -> None:
        self.model_file = Path(model_file)
        self.processor = spm.SentencePieceProcessor(model_file=str(self.model_file))
        self._validate()

    def _validate(self) -> None:
        actual = (self.pad_id, self.unk_id, self.bos_id, self.eos_id)
        if actual != (0, 1, 2, 3):
            raise ValueError(f"unexpected special token ids: {actual}")

    @property
    def pad_id(self) -> int:
        return self.processor.pad_id()

    @property
    def unk_id(self) -> int:
        return self.processor.unk_id()

    @property
    def bos_id(self) -> int:
        return self.processor.bos_id()

    @property
    def eos_id(self) -> int:
        return self.processor.eos_id()

    @property
    def vocab_size(self) -> int:
        return self.processor.vocab_size()

    def encode(self, text: str) -> list[int]:
        return list(self.processor.encode(text, out_type=int))

    def decode(self, ids: Sequence[int]) -> str:
        return self.processor.decode(list(ids))


def train_tokenizer(
    documents: Iterable[str],
    output_dir: str | Path,
    vocab_size: int = 16_000,
) -> Tokenizer:
    materialized = [document for document in documents if document.strip()]
    if not materialized:
        raise ValueError("tokenizer training requires non-empty documents")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prefix = output / "tokenizer"
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(materialized),
        model_prefix=str(prefix),
        model_type="bpe",
        vocab_size=vocab_size,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        character_coverage=1.0,
        hard_vocab_limit=False,
        shuffle_input_sentence=False,
        minloglevel=2,
    )
    return Tokenizer(prefix.with_suffix(".model"))

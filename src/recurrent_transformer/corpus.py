from __future__ import annotations

import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Sequence


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def _iter_path(path: Path, text_key: str) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if text_key not in record:
                    raise ValueError(f"{path}:{line_number} has no {text_key!r} field")
                yield str(record[text_key])
        else:
            yield from handle


def read_local_documents(
    paths: Sequence[str | Path],
    max_bytes: int,
    text_key: str = "text",
) -> list[str]:
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    documents: list[str] = []
    seen: set[str] = set()
    total_bytes = 0
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"corpus file not found: {path}")
        for raw in _iter_path(path, text_key):
            document = normalize_text(raw)
            if not document or document in seen:
                continue
            size = len(document.encode("utf-8"))
            if total_bytes + size > max_bytes:
                return documents
            seen.add(document)
            documents.append(document)
            total_bytes += size
    if not documents:
        raise ValueError("corpus contains no usable documents")
    return documents


def mix_bilingual(english: Sequence[str], chinese: Sequence[str], seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    sides = [list(english), list(chinese)]
    for side in sides:
        rng.shuffle(side)
    mixed: list[str] = []
    for index in range(max(map(len, sides))):
        for side in sides:
            if index < len(side):
                mixed.append(side[index])
    return mixed


def split_documents(documents: Sequence[str], validation_fraction: float, seed: int = 42) -> tuple[list[str], list[str]]:
    if len(documents) < 2:
        raise ValueError("at least two documents are required for a split")
    shuffled = list(documents)
    random.Random(seed).shuffle(shuffled)
    validation_size = max(1, round(len(shuffled) * validation_fraction))
    validation_size = min(validation_size, len(shuffled) - 1)
    return shuffled[validation_size:], shuffled[:validation_size]

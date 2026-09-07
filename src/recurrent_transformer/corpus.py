from __future__ import annotations

import json
import random
import re
import shutil
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


def iter_local_documents(paths: Sequence[str | Path], max_bytes: int, text_key: str = "text") -> Iterable[str]:
    """Read one document at a time; duplicate removal belongs to corpus preparation."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    total = 0
    for raw_path in paths:
        for raw in _iter_path(Path(raw_path), text_key):
            document = normalize_text(raw)
            if not document:
                continue
            size = len(document.encode("utf-8"))
            if total + size > max_bytes:
                return
            total += size
            yield document


def collect_streamed_documents(
    rows: Iterable[dict],
    *,
    text_key: str,
    max_bytes: int,
) -> list[str]:
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    documents: list[str] = []
    seen: set[str] = set()
    total_bytes = 0
    for row_number, row in enumerate(rows, 1):
        if text_key not in row:
            raise ValueError(f"stream row {row_number} has no {text_key!r} field")
        document = normalize_text(str(row[text_key]))
        if not document or document in seen:
            continue
        size = len(document.encode("utf-8"))
        if total_bytes + size > max_bytes:
            break
        documents.append(document)
        seen.add(document)
        total_bytes += size
    if not documents:
        raise ValueError("stream contains no usable documents within the byte cap")
    return documents


def check_disk_budget(target_dir: str | Path, requested_bytes: int, reserve_bytes: int = 2 * 1024**3) -> None:
    target = Path(target_dir)
    existing = target if target.exists() else target.parent
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    if requested_bytes + reserve_bytes > free:
        raise OSError(
            f"insufficient disk space: need {requested_bytes + reserve_bytes} bytes "
            f"including reserve, have {free}"
        )


def stream_huggingface_documents(
    dataset_id: str,
    *,
    dataset_config: str | None,
    split: str,
    revision: str,
    text_key: str,
    max_bytes: int,
) -> list[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("remote corpus download requires the 'datasets' package") from exc
    try:
        rows = load_dataset(
            dataset_id,
            dataset_config,
            split=split,
            revision=revision,
            streaming=True,
        )
        return collect_streamed_documents(rows, text_key=text_key, max_bytes=max_bytes)
    except Exception as exc:
        raise RuntimeError(
            f"failed to stream {dataset_id}; check DNS/network access or use local text files"
        ) from exc


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

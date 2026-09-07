"""Prepare counted, deduplicated bilingual text and tokens with bounded RAM."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import time
from urllib.request import urlopen

import numpy as np

from recurrent_transformer.corpus import check_disk_budget, normalize_text
from recurrent_transformer.tokenizer import Tokenizer

PARAMETERS = 33_104_384
SEQUENCE_LENGTH = 2048
# Equal languages and four microbatches per optimizer step.
TOKENS_PER_LANGUAGE = math.ceil(PARAMETERS * 20 / 8192) * 4096
SOURCES = {
    "english": {"dataset": "HuggingFaceFW/fineweb-edu", "config": "sample-10BT",
                "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9", "text_key": "text", "license": "ODC-By-1.0"},
    "chinese": {"dataset": "0xDing/wikipedia-cn-20230720-filtered",
                "revision": "4cef256a3f426ae1d3f6930c8cd59a32d785d99d", "text_key": "completion", "license": "CC-BY-SA-3.0"},
}


def source_rows(language, skip_documents=0):
    source = SOURCES[language]
    if language == "chinese":
        import ijson
        url = (f"https://huggingface.co/datasets/{source['dataset']}/resolve/"
               f"{source['revision']}/wikipedia-cn-20230720-filtered.json")
        # The source is a JSON array, not JSONL. Incremental parsing avoids a
        # multi-gigabyte json.load / Arrow JSON conversion.
        with urlopen(url, timeout=120) as response:
            for index, row in enumerate(ijson.items(response, "item")):
                if index >= skip_documents:
                    yield row
    else:
        from datasets import load_dataset
        rows = load_dataset(source["dataset"], source["config"],
                            revision=source["revision"], split="train", streaming=True)
        for index, row in enumerate(rows):
            if index >= skip_documents:
                yield row


def prepare_language(rows, *, language, output, tokenizer, target_tokens):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checksum = hashlib.sha256(tokenizer.model_file.read_bytes()).hexdigest()
    metadata_path = output / f"{language}.json"
    token_path = output / f"{language}.int32"
    text_path = output / f"{language}.txt"
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text())
        if (meta["tokens"] == target_tokens and meta["tokenizer_sha256"] == checksum
                and token_path.stat().st_size == target_tokens * 4 and text_path.exists()):
            print(f"{language}: verified existing complete corpus", flush=True)
            return meta
        raise ValueError(f"existing {language} corpus does not match requested tokenizer/budget")
    if token_path.with_suffix(".int32.partial").exists() and text_path.with_suffix(".txt.partial").exists():
        raise ValueError("incomplete corpus files found; remove them and rerun preparation")
    database = output / f"{language}.dedup.sqlite"
    database.unlink(missing_ok=True)
    db = sqlite3.connect(database)
    db.execute("PRAGMA cache_size=-4096")
    db.execute("CREATE TABLE seen (hash BLOB PRIMARY KEY) WITHOUT ROWID")
    temporary_tokens = token_path.with_suffix(".int32.partial")
    temporary_text = text_path.with_suffix(".txt.partial")
    tokens = docs = text_bytes = duplicates = oversized = 0
    started = last_report = time.monotonic()
    try:
        with temporary_tokens.open("wb") as binary, temporary_text.open("w", encoding="utf-8") as text:
            for row in rows:
                document = normalize_text(row[SOURCES[language]["text_key"]])
                if not document:
                    continue
                encoded = document.encode("utf-8")
                if len(encoded) > 2 * 1024**2:
                    oversized += 1
                    continue
                digest = hashlib.sha256(encoded).digest()
                if db.execute("INSERT OR IGNORE INTO seen VALUES (?)", (digest,)).rowcount == 0:
                    duplicates += 1
                    continue
                ids = [tokenizer.bos_id, *tokenizer.encode(document), tokenizer.eos_id]
                ids = ids[:target_tokens - tokens]
                np.asarray(ids, dtype="<i4").tofile(binary)
                text.write(document + "\n")
                tokens += len(ids)
                docs += 1
                text_bytes += len(encoded) + 1
                now = time.monotonic()
                if now - last_report >= 15 or tokens == target_tokens:
                    binary.flush()
                    text.flush()
                    db.commit()
                    print(json.dumps({"language": language, "tokens": tokens, "target": target_tokens,
                                      "documents": docs, "tokens_per_second": round(tokens / (now - started)),
                                      "text_bytes": text_bytes}), flush=True)
                    last_report = now
                if tokens == target_tokens:
                    break
            if tokens < target_tokens:
                raise ValueError(f"{language} source exhausted at {tokens:,}, requires {target_tokens:,}; no repetition allowed")
            binary.flush()
            os.fsync(binary.fileno())
        os.replace(temporary_tokens, token_path)
        os.replace(temporary_text, text_path)
        meta = {"language": language, "tokens": tokens, "documents": docs, "text_bytes": text_bytes,
                "duplicates_skipped": duplicates, "oversized_skipped": oversized,
                "source": SOURCES[language], "tokenizer_sha256": checksum,
                "token_file": token_path.name, "text_file": text_path.name, "dtype": "<i4"}
        metadata_path.write_text(json.dumps(meta, indent=2) + "\n")
        return meta
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/33m-20x")
    parser.add_argument("--tokenizer", default="artifacts/mps-safe-10m/tokenizer/tokenizer.model")
    parser.add_argument("--language", choices=["english", "chinese", "all"], default="all")
    parser.add_argument("--target-tokens", type=int, help="per-language target; default is half of the 20x budget")
    parser.add_argument("--skip-documents", type=int, default=0,
                        help="discard source rows already stored in an earlier shard")
    parser.add_argument("--english-config", default=None,
                        help="FineWeb-Edu config for an additional non-overlapping English shard")
    args = parser.parse_args()
    output = Path(args.output)
    check_disk_budget(output, TOKENS_PER_LANGUAGE * 2 * 12)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output / "tokenizer.model"
    if not tokenizer_path.exists():
        shutil.copyfile(args.tokenizer, tokenizer_path)
    tokenizer = Tokenizer(tokenizer_path)
    if tokenizer.vocab_size != 16000:
        raise ValueError("33M run requires the existing 16K tokenizer")
    if args.english_config:
        SOURCES["english"]["config"] = args.english_config
    target = args.target_tokens or TOKENS_PER_LANGUAGE
    for language in ([args.language] if args.language != "all" else ["chinese", "english"]):
        prepare_language(source_rows(language, args.skip_documents), language=language, output=output,
                         tokenizer=tokenizer, target_tokens=target)
    if all((output / f"{language}.json").exists() for language in SOURCES):
        languages = [json.loads((output / f"{language}.json").read_text()) for language in SOURCES]
        manifest = {"complete": True, "parameters": PARAMETERS, "target_tokens": PARAMETERS * 20,
                    "tokens": sum(x["tokens"] for x in languages), "sequence_length": SEQUENCE_LENGTH,
                    "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
                    "languages": languages}
        temp = output / "manifest.json.tmp"
        temp.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(temp, output / "manifest.json")
        print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()

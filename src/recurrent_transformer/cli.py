from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .checkpoint import load_checkpoint
from .config import ModelConfig
from .config import load_config
from .corpus import (
    check_disk_budget,
    mix_bilingual,
    read_local_documents,
    stream_huggingface_documents,
)
from .dataset import pack_documents
from .generate import generate_ids
from .tokenizer import Tokenizer, train_tokenizer
from .train import train_steps


@dataclass(frozen=True)
class PipelineResult:
    checkpoint: Path
    generated_text: str
    steps: int


def run_tiny_pipeline(
    *,
    english_path: str | Path,
    chinese_path: str | Path,
    output_dir: str | Path,
    steps: int = 1,
) -> PipelineResult:
    output = Path(output_dir)
    tokenizer_dir = output / "tokenizer"
    english = read_local_documents([english_path], max_bytes=1_000_000)
    chinese = read_local_documents([chinese_path], max_bytes=1_000_000)
    documents = mix_bilingual(english, chinese)
    tokenizer = train_tokenizer(documents, tokenizer_dir, vocab_size=128)
    blocks = pack_documents(documents, tokenizer, seq_len=16)
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_heads=4,
        mlp_hidden_size=64,
        prelude_layers=1,
        recurrent_core_layers=2,
        coda_layers=1,
        max_seq_len=64,
        min_recurrences=1,
        max_recurrences=2,
    )
    from .model import RecurrentTransformer

    model = RecurrentTransformer(config)
    training = train_steps(
        model,
        blocks.split(1),
        device="cpu",
        max_steps=steps,
        min_recurrences=1,
        max_recurrences=2,
        output_dir=output,
        seed=42,
    )
    restored, _ = load_checkpoint(training.checkpoint, expected_config=config)
    prompt = documents[0]
    prompt_ids = [tokenizer.bos_id, *tokenizer.encode(prompt)]
    import torch

    generated = generate_ids(
        restored,
        torch.tensor([prompt_ids]),
        max_new_tokens=4,
        num_recurrences=2,
        temperature=0.0,
        eos_id=tokenizer.eos_id,
    )
    return PipelineResult(
        checkpoint=training.checkpoint,
        generated_text=tokenizer.decode(generated[0].tolist()),
        steps=training.steps,
    )


def _tiny_command(args: argparse.Namespace) -> None:
    result = run_tiny_pipeline(
        english_path=args.english,
        chinese_path=args.chinese,
        output_dir=args.output,
        steps=args.steps,
    )
    print(f"checkpoint: {result.checkpoint}")
    print(f"generated: {result.generated_text}")


def _write_documents(path: Path, documents: list[str]) -> None:
    path.write_text("\n".join(documents) + "\n", encoding="utf-8")


def _prepare_data_command(args: argparse.Namespace) -> None:
    cfg = load_config(args.config).data
    output = Path(args.output or cfg.artifact_dir)
    check_disk_budget(output, 2 * cfg.max_bytes_per_language)
    output.mkdir(parents=True, exist_ok=True)
    english = stream_huggingface_documents(
        cfg.english_dataset_id,
        dataset_config=cfg.english_dataset_config,
        split="train",
        revision=cfg.dataset_revision,
        text_key=cfg.english_text_key,
        max_bytes=cfg.max_bytes_per_language,
    )
    chinese = stream_huggingface_documents(
        cfg.chinese_dataset_id,
        dataset_config=cfg.chinese_dataset_config,
        split="train",
        revision=cfg.dataset_revision,
        text_key=cfg.chinese_text_key,
        max_bytes=cfg.max_bytes_per_language,
    )
    _write_documents(output / "english.txt", english)
    _write_documents(output / "chinese.txt", chinese)
    for language, documents in (("english", english), ("chinese", chinese)):
        byte_count = sum(len(doc.encode("utf-8")) for doc in documents)
        print(f"{language}: {len(documents)} documents, {byte_count} bytes")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recurrent-transformer")
    subparsers = parser.add_subparsers(required=True)
    tiny = subparsers.add_parser("tiny-pipeline", help="run the offline acceptance pipeline")
    tiny.add_argument("--english", required=True)
    tiny.add_argument("--chinese", required=True)
    tiny.add_argument("--output", required=True)
    tiny.add_argument("--steps", type=int, default=1)
    tiny.set_defaults(handler=_tiny_command)
    prepare = subparsers.add_parser("prepare-data", help="download bounded bilingual corpora")
    prepare.add_argument("--config", default="configs/smoke.yaml")
    prepare.add_argument("--output")
    prepare.set_defaults(handler=_prepare_data_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

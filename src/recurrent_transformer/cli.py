from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .checkpoint import load_checkpoint
from .config import ModelConfig
from .corpus import mix_bilingual, read_local_documents
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recurrent-transformer")
    subparsers = parser.add_subparsers(required=True)
    tiny = subparsers.add_parser("tiny-pipeline", help="run the offline acceptance pipeline")
    tiny.add_argument("--english", required=True)
    tiny.add_argument("--chinese", required=True)
    tiny.add_argument("--output", required=True)
    tiny.add_argument("--steps", type=int, default=1)
    tiny.set_defaults(handler=_tiny_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)

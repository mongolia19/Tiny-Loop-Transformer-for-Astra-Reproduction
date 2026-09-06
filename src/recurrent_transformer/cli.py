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
from .train import select_device, train_steps


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


def _train_command(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    output = Path(args.output)
    english = read_local_documents([args.english], max_bytes=cfg.data.max_bytes_per_language)
    chinese = read_local_documents([args.chinese], max_bytes=cfg.data.max_bytes_per_language)
    documents = mix_bilingual(english, chinese, seed=cfg.data.seed)
    tokenizer = train_tokenizer(
        documents,
        output / "tokenizer",
        vocab_size=cfg.data.tokenizer_vocab_size,
    )
    if tokenizer.vocab_size > cfg.model.vocab_size:
        raise ValueError("tokenizer vocabulary exceeds model vocabulary")
    sequence_length = args.sequence_length or cfg.data.sequence_length
    blocks = pack_documents(documents, tokenizer, seq_len=sequence_length)
    from .model import RecurrentTransformer

    model = RecurrentTransformer(cfg.model)
    model.set_gradient_checkpointing(True)
    print(f"physical parameters: {model.num_parameters():,}")
    steps = 1 if args.memory_probe else (args.steps or cfg.train.max_steps)
    result = train_steps(
        model,
        blocks.split(cfg.train.micro_batch_size),
        device=args.device or cfg.train.device,
        max_steps=steps,
        min_recurrences=cfg.model.min_recurrences,
        max_recurrences=cfg.model.max_recurrences,
        output_dir=output,
        seed=cfg.train.seed,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        max_grad_norm=cfg.train.max_grad_norm,
        gradient_accumulation_steps=(
            args.gradient_accumulation
            if args.gradient_accumulation is not None
            else cfg.train.gradient_accumulation_steps
        ),
        precision=args.precision,
        save_every_steps=(args.save_every if args.save_every is not None else cfg.train.save_every_steps),
        resume_checkpoint=args.resume,
        target_loss=args.target_loss,
        target_loss_window=args.target_loss_window,
    )
    final_loss = f"{result.losses[-1]:.6f}" if result.losses else "unchanged"
    print(f"steps: {result.steps}; final loss: {final_loss}")
    print(f"checkpoint: {result.checkpoint}")


def _generate_command(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint)
    model, _ = load_checkpoint(checkpoint)
    tokenizer_path = Path(args.tokenizer) if args.tokenizer else checkpoint.parent / "tokenizer" / "tokenizer.model"
    tokenizer = Tokenizer(tokenizer_path)
    import torch

    device = select_device(args.device)
    model.to(device).eval()
    prompt_ids = torch.tensor([[tokenizer.bos_id, *tokenizer.encode(args.prompt)]], device=device)
    for recurrences in (int(value) for value in args.recurrences.split(",")):
        generated = generate_ids(
            model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            num_recurrences=recurrences,
            temperature=args.temperature,
            top_k=args.top_k,
            eos_id=tokenizer.eos_id,
            seed=args.seed,
        )
        print(f"[recurrences={recurrences}] {tokenizer.decode(generated[0].tolist())}")


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
    train = subparsers.add_parser("train", help="train from bounded local bilingual files")
    train.add_argument("--config", default="configs/smoke.yaml")
    train.add_argument("--english", required=True)
    train.add_argument("--chinese", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--steps", type=int)
    train.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"))
    train.add_argument("--memory-probe", action="store_true")
    train.add_argument("--sequence-length", type=int)
    train.add_argument("--gradient-accumulation", type=int)
    train.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    train.add_argument("--save-every", type=int, help="overwrite checkpoint-latest.pt every N steps")
    train.add_argument("--resume", help="resume from a checkpoint")
    train.add_argument("--target-loss", type=float, help="stop when the loss-window mean is below this value")
    train.add_argument("--target-loss-window", type=int, default=100)
    train.set_defaults(handler=_train_command)
    generate = subparsers.add_parser("generate", help="compare generation at recurrence depths")
    generate.add_argument("--checkpoint", required=True)
    generate.add_argument("--tokenizer")
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--recurrences", default="1,2,4")
    generate.add_argument("--max-new-tokens", type=int, default=24)
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--top-k", type=int)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    generate.set_defaults(handler=_generate_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

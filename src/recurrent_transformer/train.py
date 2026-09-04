from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor

from .checkpoint import save_checkpoint
from .model import RecurrentTransformer


@dataclass(frozen=True)
class TrainResult:
    steps: int
    losses: list[float]
    checkpoint: Path


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        requested = "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available; use --device cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(requested)
    print(f"selected device: {device}")
    return device


def ensure_finite(value: float | Tensor, what: str, step: int) -> None:
    tensor = torch.as_tensor(value)
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"non-finite {what} at step {step}")


def train_steps(
    model: RecurrentTransformer,
    batches: Iterable[Tensor],
    *,
    device: str | torch.device,
    max_steps: int,
    min_recurrences: int,
    max_recurrences: int,
    output_dir: str | Path,
    seed: int,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
) -> TrainResult:
    if max_steps <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("max_steps and gradient_accumulation_steps must be positive")
    resolved_device = select_device(str(device)) if not isinstance(device, torch.device) else device
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.to(resolved_device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    losses: list[float] = []
    batch_iterator = iter(batches)
    optimizer.zero_grad(set_to_none=True)

    with (output / "metrics.jsonl").open("w", encoding="utf-8") as metrics:
        for step in range(1, max_steps + 1):
            accumulated_loss = 0.0
            for _ in range(gradient_accumulation_steps):
                try:
                    batch = next(batch_iterator)
                except StopIteration as exc:
                    batch_iterator = iter(batches)
                    try:
                        batch = next(batch_iterator)
                    except StopIteration:
                        raise ValueError("training batches are empty") from exc
                batch = batch.to(resolved_device)
                recurrences = rng.randint(min_recurrences, max_recurrences)
                result = model(batch, labels=batch, num_recurrences=recurrences)
                if result.loss is None:
                    raise RuntimeError("model did not return a training loss")
                ensure_finite(result.loss, "loss", step)
                (result.loss / gradient_accumulation_steps).backward()
                accumulated_loss += float(result.loss.detach().cpu())
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            ensure_finite(grad_norm, "gradient norm", step)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            mean_loss = accumulated_loss / gradient_accumulation_steps
            losses.append(mean_loss)
            metrics.write(json.dumps({"step": step, "loss": mean_loss}) + "\n")
            metrics.flush()

    checkpoint = save_checkpoint(
        output / f"checkpoint-{max_steps:06d}.pt",
        model=model,
        optimizer=optimizer,
        step=max_steps,
    )
    return TrainResult(steps=max_steps, losses=losses, checkpoint=checkpoint)

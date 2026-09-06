from __future__ import annotations

import json
import math
import random
from collections import deque
from dataclasses import dataclass
from itertools import count
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


def read_loss_window(path: Path, *, checkpoint_step: int, window_size: int) -> list[float]:
    if not path.is_file():
        return []
    latest_by_step: dict[int, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        step = record.get("step")
        if isinstance(step, int) and step <= checkpoint_step:
            latest_by_step[step] = record
    valid: list[float] = []
    for step in sorted(latest_by_step):
        record = latest_by_step[step]
        loss = record.get("loss")
        if not record.get("skipped") and isinstance(loss, (int, float)) and math.isfinite(loss):
            valid.append(float(loss))
    return valid[-window_size:]


def loss_target_reached(losses: list[float], *, target: float, window_size: int) -> bool:
    return len(losses) >= window_size and sum(losses[-window_size:]) / window_size < target


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
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
    precision: str = "fp32",
    save_every_steps: int = 0,
    resume_checkpoint: str | Path | None = None,
    target_loss: float | None = None,
    target_loss_window: int = 100,
) -> TrainResult:
    if max_steps <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("max_steps and gradient_accumulation_steps must be positive")
    if save_every_steps < 0:
        raise ValueError("save_every_steps must be non-negative")
    if target_loss is not None and (not math.isfinite(target_loss) or target_loss <= 0):
        raise ValueError("target_loss must be finite and positive")
    if target_loss_window <= 0:
        raise ValueError("target_loss_window must be positive")
    resolved_device = select_device(str(device)) if not isinstance(device, torch.device) else device
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be 'fp32', 'fp16', or 'bf16'")
    if precision != "fp32" and resolved_device.type not in {"mps", "cuda"}:
        raise ValueError("mixed precision training requires MPS or CUDA")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.to(resolved_device)
    model.train()
    optimizer_eps = 1e-4 if precision != "fp32" else 1e-8
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=optimizer_eps
    )
    rng = random.Random(seed)
    torch.manual_seed(seed)
    start_step = 0
    if resume_checkpoint is not None:
        # Keep RNG state on CPU; torch.set_rng_state only accepts a CPU ByteTensor.
        payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        if payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload.get("step", 0))
        if payload.get("python_rng_state") is not None:
            rng.setstate(payload["python_rng_state"])
        if payload.get("torch_rng_state") is not None:
            torch.set_rng_state(payload["torch_rng_state"])
    losses: list[float] = []
    batch_iterator = iter(batches)
    optimizer.zero_grad(set_to_none=True)

    metrics_path = output / "metrics.jsonl"
    mode = "a" if resume_checkpoint is not None and metrics_path.exists() else "w"
    loss_window = deque(
        read_loss_window(metrics_path, checkpoint_step=start_step, window_size=target_loss_window),
        maxlen=target_loss_window,
    )
    autocast_enabled = precision != "fp32"
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    consecutive_skips = 0
    completed_step = start_step
    with metrics_path.open(mode, encoding="utf-8") as metrics:
        steps = count(start_step + 1) if target_loss is not None else range(start_step + 1, max_steps + 1)
        for step in steps:
            accumulated_loss = 0.0
            skip_step = False
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
                with torch.autocast(
                    device_type=resolved_device.type,
                    dtype=autocast_dtype,
                    enabled=autocast_enabled,
                ):
                    result = model(batch, labels=batch, num_recurrences=recurrences)
                if result.loss is None:
                    raise RuntimeError("model did not return a training loss")
                if not bool(torch.isfinite(result.loss).all()):
                    skip_step = True
                    break
                (result.loss / gradient_accumulation_steps).backward()
                accumulated_loss += float(result.loss.detach().cpu())
            if skip_step:
                optimizer.zero_grad(set_to_none=True)
                consecutive_skips += 1
                metrics.write(json.dumps({"step": step, "loss": None, "skipped": True, "reason": "non-finite loss"}) + "\n")
                metrics.flush()
                if consecutive_skips >= 8:
                    raise FloatingPointError(f"eight consecutive non-finite steps ending at step {step}")
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if not bool(torch.isfinite(torch.as_tensor(grad_norm)).all()):
                # Do not apply a corrupted update; clear it and continue safely.
                optimizer.zero_grad(set_to_none=True)
                metrics.write(json.dumps({"step": step, "loss": accumulated_loss / gradient_accumulation_steps, "skipped": True}) + "\n")
                metrics.flush()
                consecutive_skips += 1
                if consecutive_skips >= 8:
                    raise FloatingPointError(f"eight consecutive non-finite steps ending at step {step}")
                continue
            consecutive_skips = 0
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            mean_loss = accumulated_loss / gradient_accumulation_steps
            losses.append(mean_loss)
            loss_window.append(mean_loss)
            metrics.write(json.dumps({"step": step, "loss": mean_loss}) + "\n")
            metrics.flush()
            completed_step = step
            if target_loss is not None and loss_target_reached(
                list(loss_window), target=target_loss, window_size=target_loss_window
            ):
                window_mean = sum(loss_window) / target_loss_window
                metrics.write(json.dumps({
                    "event": "training_stopped",
                    "reason": "target_loss_reached",
                    "step": step,
                    "target_loss": target_loss,
                    "target_loss_window": target_loss_window,
                    "window_mean_loss": window_mean,
                }) + "\n")
                metrics.flush()
                checkpoint = save_checkpoint(
                    output / "checkpoint-latest.pt", model=model, optimizer=optimizer, step=step
                )
                return TrainResult(steps=step, losses=losses, checkpoint=checkpoint)
            if save_every_steps and step % save_every_steps == 0 and (
                target_loss is not None or step < max_steps
            ):
                save_checkpoint(output / "checkpoint-latest.pt", model=model, optimizer=optimizer, step=step)

    checkpoint_name = "checkpoint-latest.pt" if save_every_steps else f"checkpoint-{max_steps:06d}.pt"
    checkpoint = save_checkpoint(
        output / checkpoint_name,
        model=model,
        optimizer=optimizer,
        step=completed_step,
    )
    return TrainResult(steps=completed_step, losses=losses, checkpoint=checkpoint)

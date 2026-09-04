from __future__ import annotations

import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import ModelConfig
from .model import RecurrentTransformer


def save_checkpoint(
    path: str | Path,
    model: RecurrentTransformer,
    step: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    tokenizer_path: str | Path | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    payload: dict[str, Any] = {
        "config": asdict(model.config),
        "model": model.state_dict(),
        "step": step,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "tokenizer_path": str(tokenizer_path) if tokenizer_path is not None else None,
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
    }
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _config_differences(expected: ModelConfig, actual: ModelConfig) -> list[str]:
    expected_values = asdict(expected)
    actual_values = asdict(actual)
    return [
        f"{key}: expected {expected_values[key]!r}, checkpoint has {actual_values[key]!r}"
        for key in expected_values
        if expected_values[key] != actual_values[key]
    ]


def load_checkpoint(
    path: str | Path,
    expected_config: ModelConfig | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[RecurrentTransformer, dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint not found: {source}")
    payload = torch.load(source, map_location=map_location, weights_only=False)
    actual_config = ModelConfig(**payload["config"])
    if expected_config is not None:
        differences = _config_differences(expected_config, actual_config)
        if differences:
            raise ValueError("incompatible checkpoint configuration: " + "; ".join(differences))
    model = RecurrentTransformer(actual_config)
    model.load_state_dict(payload["model"])
    metadata = {key: value for key, value in payload.items() if key != "model"}
    return model, metadata

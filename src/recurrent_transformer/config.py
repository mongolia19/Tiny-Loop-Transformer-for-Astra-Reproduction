from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 16_000
    d_model: int = 768
    n_heads: int = 12
    mlp_hidden_size: int = 2_048
    prelude_layers: int = 2
    recurrent_core_layers: int = 8
    coda_layers: int = 2
    max_seq_len: int = 512
    min_recurrences: int = 1
    max_recurrences: int = 4
    rope_base: float = 10_000.0
    rms_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        integer_fields = (
            "vocab_size", "d_model", "n_heads", "mlp_hidden_size",
            "prelude_layers", "recurrent_core_layers", "coda_layers",
            "max_seq_len", "min_recurrences", "max_recurrences",
        )
        for name in integer_fields:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.min_recurrences > self.max_recurrences:
            raise ValueError("min_recurrences must not exceed max_recurrences")


@dataclass(frozen=True)
class DataConfig:
    max_bytes_per_language: int = 20 * 1024 * 1024
    sequence_length: int = 512
    tokenizer_vocab_size: int = 16_000
    validation_fraction: float = 0.01
    seed: int = 42

    def __post_init__(self) -> None:
        if self.max_bytes_per_language <= 0:
            raise ValueError("max_bytes_per_language must be positive")
        if self.sequence_length <= 1:
            raise ValueError("sequence_length must exceed one")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between zero and one")


@dataclass(frozen=True)
class TrainConfig:
    max_steps: int = 20
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    device: str = "auto"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.max_steps <= 0 or self.micro_batch_size <= 0:
            raise ValueError("training steps and batch size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        if self.device not in {"auto", "mps", "cpu", "cuda"}:
            raise ValueError(f"unsupported device: {self.device}")


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    data: DataConfig
    train: TrainConfig


ConfigT = TypeVar("ConfigT", ModelConfig, DataConfig, TrainConfig)


def _from_mapping(cls: type[ConfigT], values: dict[str, Any]) -> ConfigT:
    known = {field.name for field in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    return ExperimentConfig(
        model=_from_mapping(ModelConfig, raw.get("model", {})),
        data=_from_mapping(DataConfig, raw.get("data", {})),
        train=_from_mapping(TrainConfig, raw.get("train", {})),
    )

import pytest

from recurrent_transformer.config import ModelConfig, load_config


def test_model_config_rejects_width_not_divisible_by_heads():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(d_model=767, n_heads=12)


def test_smoke_yaml_has_bounded_local_defaults():
    cfg = load_config("configs/smoke.yaml")
    assert cfg.model.vocab_size == 16_000
    assert cfg.model.recurrent_core_layers == 8
    assert cfg.train.max_steps == 20
    assert cfg.data.max_bytes_per_language <= 20 * 1024 * 1024

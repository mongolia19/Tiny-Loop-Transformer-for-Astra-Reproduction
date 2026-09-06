import torch

from recurrent_transformer.model import RecurrentTransformer
from recurrent_transformer.train import loss_target_reached, read_loss_window, train_steps
from tests.test_model import tiny_config


def test_target_stop_saves_latest_checkpoint_and_event(tmp_path):
    import json

    model = RecurrentTransformer(tiny_config())
    result = train_steps(
        model,
        [torch.randint(0, 64, (2, 8))],
        device="cpu",
        max_steps=100,
        min_recurrences=1,
        max_recurrences=1,
        output_dir=tmp_path,
        seed=7,
        target_loss=100.0,
        target_loss_window=1,
    )
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert result.steps == 1
    assert result.checkpoint == tmp_path / "checkpoint-latest.pt"
    assert records[-1]["reason"] == "target_loss_reached"
    assert records[-1]["window_mean_loss"] < 100.0


def test_target_cli_arguments_are_typed():
    from recurrent_transformer.cli import build_parser

    args = build_parser().parse_args([
        "train", "--english", "en.txt", "--chinese", "zh.txt", "--output", "out",
        "--target-loss", "1.0", "--target-loss-window", "100",
    ])
    assert args.target_loss == 1.0
    assert args.target_loss_window == 100


def test_generate_cli_accepts_cuda():
    from recurrent_transformer.cli import build_parser

    args = build_parser().parse_args([
        "generate", "--checkpoint", "model.pt", "--prompt", "hello", "--device", "cuda",
    ])
    assert args.device == "cuda"


def test_one_step_updates_shared_core_and_writes_checkpoint(tmp_path):
    model = RecurrentTransformer(tiny_config())
    batches = [torch.randint(0, 64, (2, 8))]
    before = next(model.recurrent_core.parameters()).detach().clone()
    result = train_steps(
        model,
        batches,
        device="cpu",
        max_steps=1,
        min_recurrences=1,
        max_recurrences=2,
        output_dir=tmp_path,
        seed=7,
    )
    after = next(model.recurrent_core.parameters()).detach()
    assert result.steps == 1 and result.losses[0] > 0
    assert not torch.equal(before, after)
    assert (tmp_path / "checkpoint-000001.pt").exists()


def test_gradient_checkpointing_can_be_enabled():
    model = RecurrentTransformer(tiny_config())
    assert model.gradient_checkpointing is False
    model.set_gradient_checkpointing(True)
    assert model.gradient_checkpointing is True


def test_periodic_saves_keep_one_latest_checkpoint(tmp_path):
    model = RecurrentTransformer(tiny_config())
    batches = [torch.randint(0, 64, (2, 8))]
    result = train_steps(
        model,
        batches,
        device="cpu",
        max_steps=2,
        min_recurrences=1,
        max_recurrences=1,
        output_dir=tmp_path,
        seed=7,
        save_every_steps=1,
    )
    assert result.checkpoint == tmp_path / "checkpoint-latest.pt"
    assert result.checkpoint.exists()
    assert list(tmp_path.glob("checkpoint-*.pt")) == [result.checkpoint]
    assert not (tmp_path / "checkpoint-000002.pt").exists()


def test_nonfinite_loss_stops_training():
    import pytest
    from recurrent_transformer.train import ensure_finite

    with pytest.raises(FloatingPointError, match="non-finite loss"):
        ensure_finite(float("nan"), what="loss", step=1)


def test_read_loss_window_deduplicates_and_stops_at_checkpoint(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        '\n'.join([
            '{"step": 1, "loss": 2.0}',
            '{"step": 1, "loss": 1.5}',
            '{"step": 2, "loss": null, "skipped": true}',
            '{"step": 3, "loss": 0.5}',
        ]) + '\n'
    )
    assert read_loss_window(metrics, checkpoint_step=2, window_size=100) == [1.5]


def test_loss_target_requires_full_strict_window():
    assert not loss_target_reached([0.5] * 99, target=1.0, window_size=100)
    assert not loss_target_reached([1.0] * 100, target=1.0, window_size=100)
    assert loss_target_reached([0.99] * 100, target=1.0, window_size=100)

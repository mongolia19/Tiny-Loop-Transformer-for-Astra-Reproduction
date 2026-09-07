from pathlib import Path

from scripts.monitor_mps_training import continuation_command, should_launch, target_completed


def test_target_completed_recognizes_terminal_event(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        '{"step": 9767, "loss": 0.9}\n'
        '{"event": "training_stopped", "reason": "target_loss_reached", "step": 9767}\n'
    )
    assert target_completed(metrics)


def test_monitor_launch_decision_is_idempotent():
    assert not should_launch(running=True, completed=False, checkpoint_exists=True)
    assert not should_launch(running=False, completed=True, checkpoint_exists=True)
    assert should_launch(running=False, completed=False, checkpoint_exists=True)


def test_continuation_command_resumes_bf16_with_target():
    command = continuation_command(Path(__file__).resolve().parents[1] / "artifacts/mps-safe-10m/checkpoint-latest.pt")
    joined = " ".join(command)
    assert "--resume artifacts/mps-safe-10m/checkpoint-latest.pt" in joined
    assert "--precision bf16" in joined
    assert "--gradient-accumulation 4" in joined
    assert "--target-loss 1.0" in joined
    assert "--target-loss-window 100" in joined
    assert "--steps 9766" not in joined

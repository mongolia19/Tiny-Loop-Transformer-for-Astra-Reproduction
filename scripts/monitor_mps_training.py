"""Keep target-loss MPS training running from its latest checkpoint."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
TEN_M_OUTPUT = PROJECT / "artifacts/mps-safe-10m"
TARGET_LOSS = 1.0
TARGET_LOSS_WINDOW = 100


def training_is_running(output_name: str) -> bool:
    result = subprocess.run(
        ["pgrep", "-f", f"recurrent_transformer.cli train.*--output artifacts/{output_name}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def target_completed(path: Path) -> bool:
    if not path.is_file():
        return False
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if record.get("event") == "training_stopped":
            return record.get("reason") == "target_loss_reached"
    return False


def continuation_command(checkpoint: Path) -> list[str]:
    return [
        str(PROJECT / ".venv-mps/bin/python"),
        "-m", "recurrent_transformer.cli", "train",
        "--config", "configs/phase1-safe.yaml",
        "--english", "data/phase1/english.txt",
        "--chinese", "data/phase1/chinese.txt",
        "--output", "artifacts/mps-safe-10m",
        "--sequence-length", "128",
        "--gradient-accumulation", "4",
        "--device", "mps",
        "--precision", "bf16",
        "--save-every", "500",
        "--resume", str(checkpoint.relative_to(PROJECT)),
        "--target-loss", str(TARGET_LOSS),
        "--target-loss-window", str(TARGET_LOSS_WINDOW),
    ]


def should_launch(*, running: bool, completed: bool, checkpoint_exists: bool) -> bool:
    return not running and not completed and checkpoint_exists


def main() -> int:
    running = training_is_running("mps-safe-10m")
    metrics = TEN_M_OUTPUT / "metrics.jsonl"
    checkpoint = TEN_M_OUTPUT / "checkpoint-latest.pt"
    completed = target_completed(metrics)
    print(f"10M running={running}; target completed={completed}; checkpoint={checkpoint.is_file()}")
    if not should_launch(running=running, completed=completed, checkpoint_exists=checkpoint.is_file()):
        if completed:
            print("target loss already reached; no relaunch")
        elif running:
            print("target-loss training already running; no duplicate start")
        else:
            print("latest checkpoint missing; cannot resume")
            return 1
        return 0
    command = continuation_command(checkpoint)
    print("resuming target-loss training from latest checkpoint", flush=True)
    return subprocess.run(command, cwd=PROJECT, env={**os.environ, "PYTHONPATH": "src"}).returncode


if __name__ == "__main__":
    raise SystemExit(main())

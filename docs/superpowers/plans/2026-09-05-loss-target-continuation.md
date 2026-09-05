# Loss-Target Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resume the 10M-token training lineage from step 9766 and keep it recoverably running until 100 consecutive valid losses average below 1.0.

**Architecture:** The trainer owns loss-window evaluation and atomic completion checkpointing. The CLI exposes the target options, while the existing Harness command monitor owns idempotent relaunch from the latest checkpoint and recognizes a durable completion record.

**Tech Stack:** Python 3.14, PyTorch, argparse, pytest, Avibe Harness tasks

---

## File map

- Modify `src/recurrent_transformer/train.py`: target validation, historical-window recovery, open-ended step iteration, completion metric, and checkpoint save.
- Modify `src/recurrent_transformer/cli.py`: target CLI arguments and safe result reporting.
- Modify `tests/test_train.py`: trainer and parser acceptance tests.
- Modify `scripts/monitor_mps_training.py`: idempotent target-aware launch and completion detection.
- Create `tests/test_monitor_mps_training.py`: monitor unit tests without launching real MPS training.

### Task 1: Loss-window primitives

**Files:**
- Modify: `src/recurrent_transformer/train.py`
- Test: `tests/test_train.py`

- [ ] **Step 1: Write failing tests for historical recovery and threshold evaluation**

Add tests that write duplicate, skipped, post-checkpoint, and completion records to `metrics.jsonl`, then assert the helper returns only the latest valid losses through the checkpoint. Add a second test asserting a 100-value window with mean `0.99` passes while 99 values or mean exactly `1.0` do not.

```python
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
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `.venv-mps/bin/pytest tests/test_train.py -k 'loss_window or loss_target' -v`

Expected: collection failure because `read_loss_window` and `loss_target_reached` do not exist.

- [ ] **Step 3: Implement the minimal pure helpers**

Implement `read_loss_window(path, checkpoint_step, window_size)` using JSONL parsing, latest-record-per-step de-duplication, finite/non-skipped filtering, and `step <= checkpoint_step`. Implement `loss_target_reached(losses, target, window_size)` as a full-window strict arithmetic-mean comparison.

- [ ] **Step 4: Run focused tests**

Run: `.venv-mps/bin/pytest tests/test_train.py -k 'loss_window or loss_target' -v`

Expected: both tests pass.

- [ ] **Step 5: Commit the isolated primitives**

```bash
git add src/recurrent_transformer/train.py tests/test_train.py
git commit -m "feat: add resumable loss target window"
```

### Task 2: Trainer-native target stop

**Files:**
- Modify: `src/recurrent_transformer/train.py`
- Test: `tests/test_train.py`

- [ ] **Step 1: Write failing trainer tests**

Use a tiny deterministic model or monkeypatched loss sequence to assert: training does not stop on one sub-target loss; skipped values are excluded; reaching the full window writes `checkpoint-latest.pt`; and the terminal JSON line is:

```json
{"event":"training_stopped","reason":"target_loss_reached","step":9767,"target_loss":1.0,"target_loss_window":100,"window_mean_loss":0.99}
```

Also assert the returned `TrainResult.steps` equals the actual stopping step rather than the requested ceiling.

- [ ] **Step 2: Run the trainer target tests and verify failure**

Run: `.venv-mps/bin/pytest tests/test_train.py -k 'target_stop or skipped_loss' -v`

Expected: failures because `train_steps` has no target parameters or early-stop event.

- [ ] **Step 3: Add target parameters and validation**

Extend `train_steps` with:

```python
target_loss: float | None = None,
target_loss_window: int = 100,
```

Reject non-positive targets/windows and require a finite target. Seed a bounded `deque` from `read_loss_window` after checkpoint loading.

- [ ] **Step 4: Implement open-ended iteration and atomic target completion**

Use `itertools.count(start_step + 1)` when a target is configured, otherwise preserve `range(start_step + 1, max_steps + 1)`. After each successful optimizer step, append the mean loss; on target satisfaction, append and flush the terminal event, save `checkpoint-latest.pt`, and return `TrainResult` with the actual step.

- [ ] **Step 5: Run all trainer tests**

Run: `.venv-mps/bin/pytest tests/test_train.py -v`

Expected: all tests pass.

- [ ] **Step 6: Commit trainer behavior**

```bash
git add src/recurrent_transformer/train.py tests/test_train.py
git commit -m "feat: stop training on sustained loss target"
```

### Task 3: CLI contract

**Files:**
- Modify: `src/recurrent_transformer/cli.py`
- Test: `tests/test_train.py`

- [ ] **Step 1: Write a failing parser test**

Parse a `train` command containing `--target-loss 1.0 --target-loss-window 100` and assert both typed values are present. Add a validation test for using `--target-loss-window` without `--target-loss` if parser-level validation is chosen.

- [ ] **Step 2: Run the parser test and verify failure**

Run: `.venv-mps/bin/pytest tests/test_train.py -k target_cli -v`

Expected: argparse rejects the unknown arguments.

- [ ] **Step 3: Wire CLI options into the trainer**

Add:

```python
train.add_argument("--target-loss", type=float)
train.add_argument("--target-loss-window", type=int, default=100)
```

Pass both values to `train_steps`. Report the actual completion step and guard final-loss formatting when a resumed historical window satisfies the target before a new step.

- [ ] **Step 4: Run CLI and trainer tests**

Run: `.venv-mps/bin/pytest tests/test_train.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit CLI support**

```bash
git add src/recurrent_transformer/cli.py tests/test_train.py
git commit -m "feat: expose sustained loss target options"
```

### Task 4: Target-aware Harness monitor

**Files:**
- Modify: `scripts/monitor_mps_training.py`
- Create: `tests/test_monitor_mps_training.py`

- [ ] **Step 1: Write failing monitor tests**

Cover three states with temporary paths and mocked subprocess calls: an active matching process causes no launch; a terminal `target_loss_reached` event causes no launch; an absent process with a valid checkpoint builds one command containing `--resume`, `--precision bf16`, `--target-loss 1.0`, and `--target-loss-window 100`.

- [ ] **Step 2: Run monitor tests and verify failure**

Run: `.venv-mps/bin/pytest tests/test_monitor_mps_training.py -v`

Expected: failures because the monitor still implements one-time 10M launch behavior.

- [ ] **Step 3: Refactor monitor into testable decisions**

Add pure helpers `target_completed(metrics_path)`, `continuation_command(checkpoint)`, and `should_launch(running, completed, checkpoint_exists)`. Remove `.launch-record.json` as a launch gate. The command must resume `artifacts/mps-safe-10m/checkpoint-latest.pt`, omit the completed `--steps 9766` ceiling, use BF16, and include the target arguments.

- [ ] **Step 4: Run monitor tests**

Run: `.venv-mps/bin/pytest tests/test_monitor_mps_training.py -v`

Expected: all tests pass without starting MPS training.

- [ ] **Step 5: Run the complete suite**

Run: `.venv-mps/bin/pytest -q`

Expected: all tests pass.

- [ ] **Step 6: Commit monitor behavior**

```bash
git add scripts/monitor_mps_training.py tests/test_monitor_mps_training.py
git commit -m "feat: recover loss-target training from latest checkpoint"
```

### Task 5: Activate and verify durable continuation

**Files:**
- Modify through CLI state: Harness task `6598cd0917af`

- [ ] **Step 1: Inspect task update syntax and current task**

Run: `vibe task update --help` and `vibe task show 6598cd0917af`

Expected: the task remains a healthy 10-minute command task rooted at this project.

- [ ] **Step 2: Trigger an immediate monitor cycle**

Run: `vibe task run 6598cd0917af`

Expected: the monitor launches exactly one continuation process from `checkpoint-latest.pt`.

- [ ] **Step 3: Verify live state without interrupting training**

Run: `pgrep -af 'recurrent_transformer.cli train.*mps-safe-10m'`

Run: `tail -n 5 artifacts/mps-safe-10m/metrics.jsonl`

Expected: one process, steps greater than `9766`, finite loss records, BF16 and target arguments in the process command.

- [ ] **Step 4: Verify the next monitor cycle is idempotent**

Run: `vibe task run 6598cd0917af`

Expected: it reports an active process and does not launch a duplicate.

- [ ] **Step 5: Report activation**

Report the starting checkpoint step, current step/loss, process uniqueness, Harness task ID, stop criterion, and the fact that training will keep running recoverably until the 100-step mean is below 1.0.

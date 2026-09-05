# Loss-Target Continuation Design

## Goal

Continue the existing 10M-token model from `artifacts/mps-safe-10m/checkpoint-latest.pt` until the mean of 100 consecutive valid training losses is below `1.0`. A single low-loss step must not stop training.

## Training behavior

The training CLI will accept `--target-loss 1.0` and `--target-loss-window 100`. When both options are present, training has no ordinary step-based completion condition: it keeps producing optimizer steps until the target window is satisfied. Existing BF16 autocast, FP32 parameters and loss, learning rate `1e-4`, gradient clipping, skipped-step handling, and periodic checkpoints remain active.

Only finite, non-skipped optimizer steps enter the target window. After each valid step, the trainer computes the arithmetic mean of the latest 100 valid losses. When that mean is strictly below `1.0`, it writes the final `checkpoint-latest.pt`, records a structured stop event in `metrics.jsonl`, and exits successfully.

## Resume semantics

Training resumes from the last saved checkpoint, currently step `9766`. On startup, the trainer reads `metrics.jsonl`, de-duplicates repeated steps by keeping the latest valid record for each step, and seeds the loss window with valid records at or before the checkpoint step. This preserves the 100-step criterion across restarts without allowing post-checkpoint metrics to influence resumed model state.

The monotonically increasing step number continues beyond the original 10M-token budget. “10M-token model” identifies the existing training lineage; the new stopping condition intentionally permits additional tokens.

## Durable operation

The existing Harness command monitor will become the owner of process recovery. It will:

1. Inspect the latest checkpoint, metrics, and active process.
2. Do nothing while the correct training process is running.
3. Permanently stop relaunching after a recorded target-loss completion event.
4. Otherwise launch the BF16 trainer from `checkpoint-latest.pt` with the target-loss options.

Recoverable process failures restart from the latest checkpoint. Hardware unavailability, corrupt checkpoints, exhausted disk, or repeated non-finite failures are reported as operational failures; they do not count as successful completion.

## Observability

Normal metrics retain `step` and `loss`. The successful terminal record includes a stop reason, target, window size, and observed window mean. Periodic checkpoints continue every 500 steps. Existing reporting can show step, latest loss, 100-step mean, checkpoint freshness, process state, and whether the target has been reached.

## Tests

Tests will cover:

- no stop before 100 valid losses;
- stop when the 100-loss mean is strictly below `1.0`;
- no stop for a single sub-1 loss;
- skipped and non-finite steps do not enter the window;
- resume seeds the window only through the checkpoint step;
- target completion saves the checkpoint and terminal metric;
- the monitor does not duplicate a running process or relaunch a completed run.

## Non-goals

This change does not alter the dataset, tokenizer, model architecture, optimizer hyperparameters, or precision policy. It does not guarantee that loss `1.0` is statistically attainable; it guarantees continued recoverable training until that criterion is observed or an external resource failure requires intervention.

# Tiny Loop Transformer for Astra Reproduction

[中文文档 / Chinese README](README.zh-CN.md)

This is a tiny, local-first loop Transformer reproduction inspired by the recurrent-depth ideas discussed around Astra. It reuses a fixed shared core at inference time, increasing effective computation without increasing physical parameter count.

A small, auditable reproduction of recurrent-depth language modeling. The physical model has **97,241,856 trainable parameters**. It applies two prelude blocks, repeatedly applies the same eight-block core, and finishes with two coda blocks.

With `r` recurrences, effective depth is `2 + 8r + 2`: 12, 20, 28, or 36 blocks for the supported smoke settings, while the physical parameter count remains unchanged.

This repository demonstrates architecture and training mechanics. A one-step or 20-step smoke checkpoint is not a useful pretrained language model.

## Research basis

- Jonas Geiping et al., [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171), and the public [seal-rg/recurrent-pretraining](https://github.com/seal-rg/recurrent-pretraining) implementation. This Huginn work is the primary implementation reference.
- Nikunj Saunshi et al., [Reasoning with Latent Thoughts: On the Power of Looped Transformers](https://arxiv.org/abs/2502.17416), ICLR 2025.
- Reference video: Bilibili `BV1gBto6QEqa`, *Astra架构曝光：OpenAI的循环Transformer到底有多炸？*

The video's statements about an OpenAI model called “Astra” are media claims, not the technical specification reproduced here. This project makes no claim that OpenAI has confirmed that name, architecture, or release status.

## Differences from Huginn

- 97.24M rather than 3.5B parameters.
- A fixed 2/8/2 prelude/core/coda split and recurrence range 1–4.
- Standard PyTorch causal attention, RoPE, RMSNorm, and SwiGLU.
- Single-machine MPS/CPU training rather than distributed Frontier training.
- A tiny bounded bilingual corpus and locally trained SentencePiece tokenizer.
- Fixed recurrence selection per micro-batch; no claim of learned latent reasoning.

## 立即运行（无需联网）

```bash
cd /Users/a58/prjs/recurrent-transformer-100m
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python -m recurrent_transformer.cli tiny-pipeline \\
  --english tests/fixtures/en.txt \\
  --chinese tests/fixtures/zh.txt \\
  --output artifacts/tiny \\
  --steps 1
```

成功标准：测试显示 `19 passed`，并生成 `artifacts/tiny/checkpoint-000001.pt` 与非空 `generated:` 输出。该命令不需要 Hugging Face、账号或远程语料。

## Setup

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

实现已在 Apple Silicon + PyTorch 2.10.0 验证。若安装依赖时网络不可用，可使用已有的 PyTorch、SentencePiece、PyYAML、pytest 环境，并将 `python -m recurrent_transformer.cli` 作为统一入口。

## Corpus

Configured remote sources are deliberately bounded to 20 MiB per language:

- English: `HuggingFaceFW/fineweb-edu`, `sample-10BT`, ODC-By 1.0.
- Chinese: `0xDing/wikipedia-cn-20230720-filtered`, CC-BY-SA-3.0.

可选：下载受限远程语料（立即运行流程不需要）：

```bash
recurrent-transformer prepare-data --config configs/smoke.yaml
```

The command streams rows and stops before the configured UTF-8 byte cap. It checks disk space with a 2 GiB reserve. If Hugging Face or `datasets` is unavailable, use local UTF-8 `.txt` files instead:

```bash
recurrent-transformer train \
  --config configs/smoke.yaml \
  --english path/to/english.txt \
  --chinese path/to/chinese.txt \
  --output artifacts/smoke
```

Local `.jsonl` ingestion is available through the corpus module with an explicit text key. Normalization uses Unicode NFKC, collapsed whitespace, exact normalized-text deduplication, and a hard byte limit.

## Offline acceptance pipeline

This exercises corpus loading, a 128-token test tokenizer, packed blocks, a structurally identical tiny model, one optimizer step, atomic checkpoint reload, and generation without network access:

```bash
recurrent-transformer tiny-pipeline \
  --english tests/fixtures/en.txt \
  --chinese tests/fixtures/zh.txt \
  --output artifacts/tiny \
  --steps 1
```

## 100M smoke training

### Apple Silicon MPS (verified)

Use a virtual environment that reuses the system PyTorch, then install the small runtime dependencies:

```bash
cd /Users/a58/prjs/recurrent-transformer-100m
python3 -m venv --system-site-packages .venv-mps
.venv-mps/bin/python -m pip install -U pip
.venv-mps/bin/python -m pip install sentencepiece pyyaml
```

Run the short MPS probe first:

```bash
PYTHONPATH=src .venv-mps/bin/python -m recurrent_transformer.cli train \
  --config configs/smoke.yaml \
  --english tests/fixtures/en.txt --chinese tests/fixtures/zh.txt \
  --output artifacts/mps-probe --memory-probe \
  --sequence-length 32 --gradient-accumulation 1 --device mps
```

After it reports `selected device: mps`, run 20-step smoke training:

```bash
PYTHONPATH=src .venv-mps/bin/python -m recurrent_transformer.cli train \
  --config configs/smoke.yaml \
  --english tests/fixtures/en.txt --chinese tests/fixtures/zh.txt \
  --output artifacts/mps-smoke-20 --steps 20 \
  --sequence-length 32 --gradient-accumulation 1 --device mps
```

On 16GB Apple Silicon, keep these conservative settings initially and close other apps using unified memory before increasing sequence length.

The default configuration uses a sequence length of 512, micro-batch size 1, gradient accumulation 4, random recurrence count 1–4, activation checkpointing on the shared core, and 20 optimizer steps.

Start with a memory probe:

```bash
recurrent-transformer train \
  --config configs/smoke.yaml \
  --english data/smoke/english.txt \
  --chinese data/smoke/chinese.txt \
  --output artifacts/probe \
  --memory-probe
```

On a memory-constrained machine, reduce sequence length and accumulation explicitly:

```bash
recurrent-transformer train \
  --config configs/smoke.yaml \
  --english data/smoke/english.txt \
  --chinese data/smoke/chinese.txt \
  --output artifacts/smoke \
  --memory-probe \
  --sequence-length 32 \
  --gradient-accumulation 1 \
  --device cpu
```

Artifacts include `tokenizer/tokenizer.model`, `metrics.jsonl`, and an atomic `checkpoint-NNNNNN.pt`. The checkpoint stores model configuration, model and optimizer state, step, and random states.

## Generation with test-time recurrence

```bash
recurrent-transformer generate \
  --checkpoint artifacts/smoke/checkpoint-000020.pt \
  --prompt '循环 Transformer can' \
  --recurrences 1,2,4 \
  --max-new-tokens 24
```

Different outputs are observations, not evidence of quality improvement. Meaningful comparison requires a trained model and controlled evaluation.

## Recorded local verification

Date: 2026-09-04. Machine: Apple M1 MacBook Pro, 16 GB unified memory.

- Automated suite: 19 tests passed before final documentation verification.
- Physical parameters: 97,241,856.
- MPS result: both float32 and float16 processes were terminated while moving the full model to MPS under high system memory pressure, before forward execution. No silent fallback was used.
- CPU fallback: one optimizer step completed with sequence length 32, gradient accumulation 1, and sampled recurrence range 1–4.
- Recorded loss: 9.89547061920166.
- Checkpoint: approximately 1.1 GiB including AdamW state.
- Reload/generation: completed at recurrence counts 1, 2, and 4.
- Remote corpus: not downloaded during this run because system DNS could not resolve Hugging Face and the available environment lacked `datasets`; the command failed with an actionable message. Local bounded text fallback was used to verify training.

The fallback text was an English implementation-plan document plus an existing Chinese public-video transcript. It is adequate only for pipeline validation, not model-quality training.

## Tests

The suite covers configuration validation, physical parameter count, shared recurrent weights and gradients, future-token isolation, backward pass, text caps and deduplication, tokenizer IDs, packing, atomic checkpoint equivalence, incompatible checkpoint diagnostics, one-step optimization, recurrence-controlled generation, and the offline end-to-end pipeline.

```bash
python -m pytest -q
python -m compileall -q src
git diff --check
```

## Windows + NVIDIA CUDA

Install a CUDA-enabled PyTorch wheel matching the installed NVIDIA driver, then install this project with `pip install -e .`. Use PowerShell with `$env:PYTHONPATH="src"` and pass `--device cuda`; `--precision bf16` is recommended on Ampere or newer GPUs, while `--precision fp16` supports older CUDA GPUs.

```powershell
$env:PYTHONPATH="src"
python -m recurrent_transformer.cli train `
  --config configs/phase1-safe.yaml `
  --english data/phase1/english.txt `
  --chinese data/phase1/chinese.txt `
  --output artifacts/cuda-run `
  --device cuda --precision bf16 --steps 10
```

Run a short smoke test first, then remove `--steps 10` and add `--save-every 500` for long training. The `auto` device mode prefers CUDA when available, then MPS, then CPU.

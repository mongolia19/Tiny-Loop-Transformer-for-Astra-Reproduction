# 100M Recurrent-Depth Transformer Design

Date: 2026-09-04

## Goal

Build a small, auditable recurrent-depth causal language model that runs end to end on an Apple M1 MacBook Pro with 16 GB unified memory. The acceptance target is a local smoke pretraining run, not useful convergence: bounded bilingual corpus acquisition, tokenizer training, model initialization, training, checkpoint reload, and generation must all complete locally.

The model should contain approximately 100 million physical parameters while allowing additional test-time computation through repeated application of a weight-shared Transformer core.

## Research Basis and Claims

The implementation is grounded in these public papers:

1. Jonas Geiping et al., *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach*, arXiv:2502.05171. This paper and its public `seal-rg/recurrent-pretraining` repository are the primary architectural references for the Huginn recurrent-depth language model.
2. Nikunj Saunshi et al., *Reasoning with Latent Thoughts: On the Power of Looped Transformers*, arXiv:2502.17416 / ICLR 2025. This paper supplies theoretical and controlled-experiment context for looped Transformers.

The referenced Bilibili video is `BV1gBto6QEqa`, *Astra架构曝光：OpenAI的循环Transformer到底有多炸？* by 杜雨说AI. Its transcript discusses recurrent depth, latent-space computation, the two research directions above, and an alleged OpenAI model called “Astra.” The project will not present Astra, its architecture, or its release status as an established OpenAI fact. “Astra” is contextual motivation only; the reproducible target is the public Huginn-style architecture.

## Scope

### Included

- A PyTorch decoder-only causal LM with a recurrent, weight-shared core.
- Apple MPS execution with CPU fallback.
- Bounded acquisition and preprocessing of English FineWeb-Edu and cleaned Chinese Wikipedia samples.
- A locally trained 16,000-token bilingual tokenizer.
- Smoke and extended configuration files.
- Checkpointing, reload, loss logging, and recurrence-controlled generation.
- Unit tests and a tiny offline end-to-end test fixture.
- Documentation of parameter count, references, simplifications, and commands.

### Excluded

- Training to convergence or claiming competitive language-model quality.
- Reproducing Huginn's 3.5B scale, 800B-token dataset, or distributed training stack.
- Adaptive computation or learned halting.
- Distributed, CUDA, or production-serving optimization in the initial version.
- Treating media reports about OpenAI Astra as verified technical documentation.

## Model Architecture

The model is a pre-norm, decoder-only causal Transformer with three stages:

1. `prelude`: two distinct Transformer blocks.
2. `recurrent_core`: eight distinct Transformer blocks forming one shared core. The entire eight-block core is reapplied for each recurrence; parameters are shared across recurrences.
3. `coda`: two distinct Transformer blocks.

The initial target configuration is:

- vocabulary size: 16,000
- model width: 768
- attention heads: 12
- key/value heads: 12 unless memory profiling justifies grouped-query attention
- MLP: SwiGLU with intermediate size 2,048; the implementation test remains authoritative that the complete physical parameter count lies between 95M and 105M
- normalization: RMSNorm
- positions: RoPE
- context length: 512
- tied token embedding and language-model head
- dropout: zero by default for a deterministic smoke run

With recurrence count `r`, effective block depth is `2 + 8r + 2`. Thus recurrence counts 1, 2, 3, and 4 produce effective depths 12, 20, 28, and 36 without changing the physical parameter count.

Training samples `r` from a configurable discrete range, initially 1–4. Evaluation and generation accept an explicit recurrence count. All residual streams retain the same shape across stage and recurrence boundaries.

## Data and Tokenizer

The default online data sources are:

- English: a streamed sample from FineWeb-Edu.
- Chinese: a streamed sample from a cleaned Chinese Wikipedia dataset.

Acquisition must be bounded before downloading. The smoke defaults cap each language at approximately 10–20 MB of normalized text and stop as soon as the cap is reached. The exact Hugging Face dataset identifiers will be pinned in configuration after availability and licensing are verified during implementation.

Processing performs Unicode normalization, whitespace cleanup, empty-document removal, and exact-content deduplication. It avoids aggressive language-specific rewriting. Chinese and English documents are mixed to approximate a 1:1 token ratio; deterministic shuffling uses a configured seed.

A 16,000-token SentencePiece BPE tokenizer is trained locally from the bounded corpus. BOS, EOS, PAD, and UNK identifiers are explicit and validated. The tokenized training stream is concatenated and divided into fixed 512-token next-token-prediction blocks. A small deterministic validation split is held out by document before tokenizer-to-block conversion.

The pipeline also accepts local `.txt` and `.jsonl` inputs. Network failure must not prevent offline tests or use with user-supplied data.

## Training and Generation

The smoke configuration uses:

- MPS when available, otherwise CPU
- micro-batch size 1
- gradient accumulation to a small effective batch
- 20 optimizer steps by default
- AdamW, gradient clipping, and a short warmup/decay schedule
- recurrence sampled uniformly from 1–4 unless configured otherwise
- deterministic seeds where PyTorch permits
- checkpoint and log output beneath a configurable artifacts directory

The trainer detects non-finite loss or gradients and stops with an actionable error. Checkpoints contain the model configuration, model state, optimizer state, scheduler state, training step, tokenizer reference, and random-state metadata needed for a smoke-level resume.

Generation loads a checkpoint and tokenizer, accepts prompt text and recurrence count, and supports greedy or temperature/top-k sampling. A comparison command runs the same prompt and seed at several recurrence counts. Output differences are observations only and are not presented as quality improvement without evaluation evidence.

## Components and Interfaces

- `model.py`: configuration, attention/MLP/block primitives, recurrent-depth model, parameter-count reporting, and `forward(input_ids, labels=None, num_recurrences=None)`.
- `data.py`: bounded source streaming, local-file ingestion, normalization, deduplication, bilingual mixing, and token-block dataset creation.
- `tokenizer.py`: tokenizer training, special-token validation, save, and load.
- `train.py`: device selection, dataloaders, optimization loop, recurrence sampling, logging, checkpoint save/resume, and failure checks.
- `generate.py`: checkpoint loading and recurrence-controlled text generation/comparison.
- configuration files: model/data/training values for smoke and optional extended runs.
- tests: offline fixtures plus focused behavioral and integration coverage.

Public interfaces must use configuration objects rather than module-level mutable globals. Paths are resolved explicitly and output directories are created narrowly.

## Error Handling

The command-line tools fail early with a nonzero exit and a clear message for:

- unavailable or empty corpora
- corpus byte/token caps yielding too little tokenizer input
- insufficient free disk space before data acquisition
- missing or inconsistent special tokens
- unsupported device/dtype combinations
- non-finite training state
- incompatible model configuration and checkpoint
- missing checkpoint or tokenizer files

MPS selection is reported. If a required operation cannot run on MPS, the program either uses an explicitly documented CPU fallback or stops with remediation; it does not silently move arbitrary tensors between devices.

## Testing Strategy

Implementation follows test-driven development. Tests are written and observed failing before production behavior is added.

Required tests cover:

1. Reapplying the core does not create recurrence-specific parameters and gradients reach shared core weights.
2. Logit and residual shapes remain correct for multiple recurrence counts.
3. The causal mask prevents a future token from changing earlier-token logits in evaluation mode.
4. The target configuration's physical parameter count is between 95M and 105M.
5. A tiny batch completes forward, causal-LM loss, backward, and optimizer update.
6. A saved checkpoint reloads to equivalent logits for the same input and recurrence count.
7. Invalid recurrence counts and incompatible checkpoints fail clearly.
8. Tiny local bilingual fixtures can train a small tokenizer, form token blocks, run at least one optimization step, save/reload, and generate tokens without network access.

The full 100M smoke run is a manual acceptance test because it is too expensive for the routine unit suite. Unit tests use a structurally identical tiny configuration.

## Acceptance Criteria

The project is complete when all of the following evidence exists:

- The automated test suite passes locally.
- The configured model reports 95M–105M trainable physical parameters.
- The bounded bilingual data command completes, or its documented local-file fallback is demonstrated if the remote source is unavailable.
- The tokenizer and token blocks are produced.
- At least one optimizer step of the 100M model completes on the selected local device; the default target is the full 20-step smoke run when runtime permits.
- A checkpoint is saved and reloaded.
- Generation completes from the reloaded checkpoint at two or more recurrence counts.
- README instructions identify the papers, distinguish verified research from the Astra media claim, and state that a smoke-trained checkpoint is not a useful pretrained model.

## Resource Guardrails

The current machine has 16 GB unified memory and approximately 26 GB free disk space at design time. Data downloads are capped, intermediate artifacts are kept below a configurable budget, and training starts with a one-step memory probe. Activation checkpointing is enabled for the recurrent core if needed. A failure during the memory probe results in smaller sequence length or explicit CPU execution, not an unbounded retry loop.

## Implementation Order

After this design is approved in writing, a separate implementation plan will sequence environment setup, tiny-model TDD, parameter-budget calibration, offline data/tokenizer integration, trainer/checkpoint work, remote corpus acquisition, and final MPS smoke verification.

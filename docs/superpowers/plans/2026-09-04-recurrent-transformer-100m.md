# 100M Recurrent-Depth Transformer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and locally smoke-pretrain a roughly 100M-parameter bilingual Huginn-style recurrent-depth causal language model on Apple M1 using PyTorch MPS.

**Architecture:** A decoder-only model applies two unique prelude blocks, an eight-block core one or more times with shared weights, and two unique coda blocks. Offline-first corpus, tokenizer, trainer, checkpoint, and generation modules are kept independently testable; the full-size smoke run reuses the same paths exercised by tiny fixtures.

**Tech Stack:** Python 3.11+, PyTorch 2.x, SentencePiece, Hugging Face `datasets`, PyYAML, pytest, Apple MPS with CPU fallback.

---

## File Map

- `pyproject.toml`: package metadata, dependencies, pytest configuration, and CLI entry points.
- `.gitignore`: local environments, caches, downloaded data, and training artifacts.
- `src/recurrent_transformer/config.py`: validated model/data/train configuration dataclasses and YAML loading.
- `src/recurrent_transformer/model.py`: RMSNorm, RoPE attention, SwiGLU block, recurrent-depth causal LM, and parameter reporting.
- `src/recurrent_transformer/corpus.py`: bounded remote streaming, local ingestion, normalization, deduplication, and bilingual mixing.
- `src/recurrent_transformer/tokenizer.py`: SentencePiece training/loading and special-token validation.
- `src/recurrent_transformer/dataset.py`: deterministic document split and packed causal-LM blocks.
- `src/recurrent_transformer/checkpoint.py`: atomic checkpoint save, compatibility validation, and restore.
- `src/recurrent_transformer/train.py`: device selection, recurrence sampling, optimizer loop, logging, resume, and memory probe.
- `src/recurrent_transformer/generate.py`: checkpoint-based autoregressive generation and recurrence comparison.
- `src/recurrent_transformer/cli.py`: narrow command-line adapters for data, tokenizer, train, and generation operations.
- `configs/smoke.yaml`: 100M, 20-step local acceptance configuration.
- `configs/extended.yaml`: explicit opt-in longer-run configuration.
- `tests/fixtures/{zh,en}.txt`: tiny offline bilingual fixtures.
- `tests/test_config.py`: configuration validation.
- `tests/test_model.py`: recurrence sharing, shapes, causality, parameter count, and backward pass.
- `tests/test_corpus.py`: corpus caps, normalization, deduplication, and mixing.
- `tests/test_tokenizer_dataset.py`: tokenizer and packed-block behavior.
- `tests/test_checkpoint.py`: checkpoint round-trip and compatibility errors.
- `tests/test_train.py`: one-step offline train and resume.
- `tests/test_generate.py`: deterministic generation and recurrence selection.
- `tests/test_e2e.py`: complete tiny local pipeline.
- `README.md`: research grounding, setup, commands, limitations, and observed smoke evidence.

### Task 1: Package Skeleton and Validated Configuration

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/recurrent_transformer/__init__.py`
- Create: `src/recurrent_transformer/config.py`
- Create: `configs/smoke.yaml`
- Create: `configs/extended.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing configuration tests**

```python
# tests/test_config.py
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
```

- [ ] **Step 2: Run the tests and verify the import failure**

Run: `python -m pytest tests/test_config.py -v`

Expected: FAIL because `recurrent_transformer.config` does not exist.

- [ ] **Step 3: Add package metadata, dataclasses, and exact YAML values**

Implement frozen dataclasses `ModelConfig`, `DataConfig`, `TrainConfig`, and `ExperimentConfig`. Validate positive dimensions, `d_model % n_heads == 0`, positive recurrence bounds, `min_recurrences <= max_recurrences`, and explicit byte caps. Implement `load_config(path) -> ExperimentConfig` with `yaml.safe_load`. The smoke model values are vocabulary 16000, width 768, heads 12, MLP 2048, prelude 2, core 8, coda 2, sequence length 512, recurrence range 1–4, batch 1, accumulation 4, and 20 steps. Add dependencies and CLI scripts in `pyproject.toml`.

- [ ] **Step 4: Run the focused test and package import check**

Run: `python -m pytest tests/test_config.py -v && python -c 'from recurrent_transformer.config import load_config; print(load_config("configs/smoke.yaml").model)'`

Expected: 2 tests PASS and a printed `ModelConfig`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore src configs tests/test_config.py
git commit -m "chore: scaffold recurrent transformer package"
```

### Task 2: Recurrent-Depth Model Core

**Files:**
- Create: `src/recurrent_transformer/model.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Write failing tests for recurrence sharing, shape, causality, and gradients**

```python
# tests/test_model.py
import torch
from recurrent_transformer.config import ModelConfig
from recurrent_transformer.model import RecurrentTransformer


def tiny_config():
    return ModelConfig(vocab_size=64, d_model=32, n_heads=4, mlp_hidden_size=64,
                       prelude_layers=1, recurrent_core_layers=2, coda_layers=1,
                       max_seq_len=16, min_recurrences=1, max_recurrences=3)


def test_recurrence_reuses_the_same_parameters_and_preserves_shape():
    model = RecurrentTransformer(tiny_config())
    ids = torch.randint(0, 64, (2, 8))
    parameter_ids = {id(p) for p in model.recurrent_core.parameters()}
    assert model(ids, num_recurrences=1).logits.shape == (2, 8, 64)
    assert model(ids, num_recurrences=3).logits.shape == (2, 8, 64)
    assert parameter_ids == {id(p) for p in model.recurrent_core.parameters()}


def test_future_tokens_do_not_change_earlier_logits():
    torch.manual_seed(1)
    model = RecurrentTransformer(tiny_config()).eval()
    a = torch.tensor([[1, 2, 3, 4]])
    b = torch.tensor([[1, 2, 9, 10]])
    with torch.no_grad():
        logits_a = model(a, num_recurrences=2).logits[:, :2]
        logits_b = model(b, num_recurrences=2).logits[:, :2]
    torch.testing.assert_close(logits_a, logits_b)


def test_loss_backward_reaches_shared_core():
    model = RecurrentTransformer(tiny_config())
    ids = torch.randint(0, 64, (2, 8))
    output = model(ids, labels=ids, num_recurrences=2)
    output.loss.backward()
    assert output.loss.isfinite()
    assert all(p.grad is not None for p in model.recurrent_core.parameters())
```

- [ ] **Step 2: Run and verify the missing-model failure**

Run: `python -m pytest tests/test_model.py -v`

Expected: FAIL because `RecurrentTransformer` is not defined.

- [ ] **Step 3: Implement the minimal causal recurrent model**

Implement `RMSNorm`, RoPE cache/application, causal scaled-dot-product attention, bias-free SwiGLU, pre-norm `TransformerBlock`, and `RecurrentTransformer`. Store the core once in `nn.ModuleList`; loop over that same list `num_recurrences` times. Tie `lm_head.weight` to `token_embedding.weight`. Return a `ModelOutput(logits, loss)` dataclass. Shift logits/labels by one for cross-entropy and reject recurrences outside the configured positive range.

- [ ] **Step 4: Run model tests on CPU**

Run: `python -m pytest tests/test_model.py -v`

Expected: 3 tests PASS with no warnings.

- [ ] **Step 5: Commit**

```bash
git add src/recurrent_transformer/model.py tests/test_model.py
git commit -m "feat: implement shared recurrent transformer core"
```

### Task 3: Calibrate and Guard the 100M Configuration

**Files:**
- Modify: `src/recurrent_transformer/model.py`
- Modify: `configs/smoke.yaml`
- Modify: `tests/test_model.py`

- [ ] **Step 1: Add a failing physical-parameter budget test**

```python
def test_smoke_model_has_about_100m_physical_parameters():
    from recurrent_transformer.config import load_config
    cfg = load_config("configs/smoke.yaml").model
    model = RecurrentTransformer(cfg)
    count = model.num_parameters()
    assert 95_000_000 <= count <= 105_000_000, count
```

- [ ] **Step 2: Run only the budget test and record the actual count**

Run: `python -m pytest tests/test_model.py::test_smoke_model_has_about_100m_physical_parameters -v`

Expected: FAIL if the initial exact count is outside the interval, proving the guard is active.

- [ ] **Step 3: Add `num_parameters` and calibrate only the MLP width if required**

Implement `num_parameters(trainable_only=True)` by summing unique parameter objects so tied embeddings are counted once. Keep vocabulary 16000, width 768, 12 heads, and the 2/8/2 block split fixed. If 2048 misses the interval, choose the nearest multiple of 256 that lands inside it and update both YAML files plus the design document's numeric field.

- [ ] **Step 4: Run all model and configuration tests**

Run: `python -m pytest tests/test_model.py tests/test_config.py -v`

Expected: all tests PASS and the failure output reports a count within 95M–105M.

- [ ] **Step 5: Commit**

```bash
git add src/recurrent_transformer/model.py configs docs/superpowers/specs tests/test_model.py
git commit -m "test: enforce 100m physical parameter budget"
```

### Task 4: Offline Corpus, Tokenizer, and Packed Dataset

**Files:**
- Create: `src/recurrent_transformer/corpus.py`
- Create: `src/recurrent_transformer/tokenizer.py`
- Create: `src/recurrent_transformer/dataset.py`
- Create: `tests/fixtures/en.txt`
- Create: `tests/fixtures/zh.txt`
- Test: `tests/test_corpus.py`
- Test: `tests/test_tokenizer_dataset.py`

- [ ] **Step 1: Write failing bounded-corpus tests**

```python
# tests/test_corpus.py
from recurrent_transformer.corpus import normalize_text, read_local_documents


def test_normalize_and_deduplicate_with_byte_cap(tmp_path):
    path = tmp_path / "docs.txt"
    path.write_text("  Hello   world  \n\nHello world\n独立 文档\n", encoding="utf-8")
    docs = read_local_documents([path], max_bytes=40)
    assert docs == ["Hello world", "独立 文档"]
    assert sum(len(x.encode("utf-8")) for x in docs) <= 40


def test_normalize_text_uses_nfkc_and_collapses_whitespace():
    assert normalize_text("Ａ  B\tC") == "A B C"
```

- [ ] **Step 2: Write failing tokenizer and packing tests**

```python
# tests/test_tokenizer_dataset.py
from recurrent_transformer.dataset import pack_documents
from recurrent_transformer.tokenizer import train_tokenizer


def test_tiny_tokenizer_and_fixed_blocks(tmp_path):
    docs = ["hello recurrent world " * 40, "你好 循环 世界 " * 40]
    tokenizer = train_tokenizer(docs, tmp_path, vocab_size=128)
    assert tokenizer.pad_id >= 0 and tokenizer.bos_id >= 0 and tokenizer.eos_id >= 0
    blocks = pack_documents(docs, tokenizer, seq_len=16)
    assert blocks.ndim == 2
    assert blocks.shape[1] == 16
```

- [ ] **Step 3: Run both tests and verify missing-module failures**

Run: `python -m pytest tests/test_corpus.py tests/test_tokenizer_dataset.py -v`

Expected: FAIL because corpus/tokenizer/dataset modules do not exist.

- [ ] **Step 4: Implement deterministic offline processing**

Use NFKC, collapsed internal whitespace, exact normalized-string deduplication, and UTF-8 byte accounting. Support `.txt` lines and `.jsonl` records with configurable text key. Train SentencePiece BPE with fixed IDs `pad=0, unk=1, bos=2, eos=3`, `hard_vocab_limit=false`, and save model/vocabulary. Pack BOS-document-EOS streams into non-overlapping `seq_len` tensors, dropping only the final short remainder. Add deterministic document-level split and alternating bilingual mix helpers.

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest tests/test_corpus.py tests/test_tokenizer_dataset.py -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/recurrent_transformer/corpus.py src/recurrent_transformer/tokenizer.py src/recurrent_transformer/dataset.py tests
git commit -m "feat: add bounded bilingual text pipeline"
```

### Task 5: Checkpointing and One-Step Trainer

**Files:**
- Create: `src/recurrent_transformer/checkpoint.py`
- Create: `src/recurrent_transformer/train.py`
- Test: `tests/test_checkpoint.py`
- Test: `tests/test_train.py`

- [ ] **Step 1: Write failing checkpoint equivalence test**

```python
# tests/test_checkpoint.py
import torch
from recurrent_transformer.checkpoint import load_checkpoint, save_checkpoint
from recurrent_transformer.model import RecurrentTransformer
from tests.test_model import tiny_config


def test_checkpoint_round_trip_preserves_logits(tmp_path):
    torch.manual_seed(4)
    model = RecurrentTransformer(tiny_config()).eval()
    ids = torch.randint(0, 64, (1, 8))
    expected = model(ids, num_recurrences=2).logits.detach()
    save_checkpoint(tmp_path / "step.pt", model=model, step=1)
    restored, metadata = load_checkpoint(tmp_path / "step.pt", expected_config=tiny_config())
    torch.testing.assert_close(restored.eval()(ids, num_recurrences=2).logits, expected)
    assert metadata["step"] == 1
```

- [ ] **Step 2: Write a failing real one-step optimization test**

```python
# tests/test_train.py
import torch
from recurrent_transformer.model import RecurrentTransformer
from recurrent_transformer.train import train_steps
from tests.test_model import tiny_config


def test_one_step_updates_shared_core_and_writes_checkpoint(tmp_path):
    model = RecurrentTransformer(tiny_config())
    batches = [torch.randint(0, 64, (2, 8))]
    before = next(model.recurrent_core.parameters()).detach().clone()
    result = train_steps(model, batches, device="cpu", max_steps=1,
                         min_recurrences=1, max_recurrences=2,
                         output_dir=tmp_path, seed=7)
    after = next(model.recurrent_core.parameters()).detach()
    assert result.steps == 1 and result.losses[0] > 0
    assert not torch.equal(before, after)
    assert (tmp_path / "checkpoint-000001.pt").exists()
```

- [ ] **Step 3: Run tests and verify missing API failures**

Run: `python -m pytest tests/test_checkpoint.py tests/test_train.py -v`

Expected: FAIL because checkpoint and training APIs do not exist.

- [ ] **Step 4: Implement atomic checkpointing and the bounded trainer**

Save to a sibling temporary file and replace the target only after `torch.save` succeeds. Persist serialized config, model, optional optimizer/scheduler, step, tokenizer path, and RNG states. Reject configuration mismatches with field-level messages. In `train_steps`, use AdamW, shifted causal loss from the model, seeded integer recurrence sampling, gradient accumulation, clipping, finite-loss/gradient checks, JSONL metrics, and exact `max_steps` termination. Add `select_device("auto")` returning MPS when available, else CPU, with an explicit log message.

- [ ] **Step 5: Run checkpoint and trainer tests**

Run: `python -m pytest tests/test_checkpoint.py tests/test_train.py -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/recurrent_transformer/checkpoint.py src/recurrent_transformer/train.py tests/test_checkpoint.py tests/test_train.py
git commit -m "feat: add finite smoke trainer and checkpoints"
```

### Task 6: Generation and Tiny End-to-End Pipeline

**Files:**
- Create: `src/recurrent_transformer/generate.py`
- Create: `src/recurrent_transformer/cli.py`
- Test: `tests/test_generate.py`
- Test: `tests/test_e2e.py`

- [ ] **Step 1: Write failing generation test**

```python
# tests/test_generate.py
import torch
from recurrent_transformer.generate import generate_ids
from recurrent_transformer.model import RecurrentTransformer
from tests.test_model import tiny_config


def test_greedy_generation_honors_length_and_recurrence():
    model = RecurrentTransformer(tiny_config()).eval()
    prompt = torch.tensor([[2, 5, 6]])
    output = generate_ids(model, prompt, max_new_tokens=4, num_recurrences=3,
                          temperature=0.0)
    assert output.shape == (1, 7)
    torch.testing.assert_close(output[:, :3], prompt)
```

- [ ] **Step 2: Write failing offline end-to-end test**

```python
# tests/test_e2e.py
from recurrent_transformer.cli import run_tiny_pipeline


def test_offline_bilingual_pipeline(tmp_path):
    result = run_tiny_pipeline(
        english_path="tests/fixtures/en.txt",
        chinese_path="tests/fixtures/zh.txt",
        output_dir=tmp_path,
        steps=1,
    )
    assert result.checkpoint.exists()
    assert result.generated_text.strip()
    assert result.steps == 1
```

- [ ] **Step 3: Run and verify missing-generation failures**

Run: `python -m pytest tests/test_generate.py tests/test_e2e.py -v`

Expected: FAIL because generation and CLI orchestration do not exist.

- [ ] **Step 4: Implement bounded generation and CLI adapters**

Implement greedy decoding when temperature is zero; otherwise apply temperature, optional top-k filtering, and `torch.multinomial` with a seeded generator. Stop on EOS or `max_new_tokens`. Add CLI subcommands `prepare-data`, `train-tokenizer`, `train`, `generate`, and `tiny-pipeline`; orchestration calls module APIs rather than duplicating logic. Ensure `tiny-pipeline` forces the tiny structural configuration, local fixtures, CPU, one step, and no network.

- [ ] **Step 5: Run generation, end-to-end, then the complete test suite**

Run: `python -m pytest tests/test_generate.py tests/test_e2e.py -v && python -m pytest -q`

Expected: focused tests and complete suite PASS.

- [ ] **Step 6: Commit**

```bash
git add src/recurrent_transformer/generate.py src/recurrent_transformer/cli.py tests/test_generate.py tests/test_e2e.py pyproject.toml
git commit -m "feat: add recurrence-controlled generation pipeline"
```

### Task 7: Bounded Remote Corpus Acquisition

**Files:**
- Modify: `src/recurrent_transformer/corpus.py`
- Modify: `src/recurrent_transformer/cli.py`
- Modify: `configs/smoke.yaml`
- Modify: `tests/test_corpus.py`

- [ ] **Step 1: Add failing tests using an in-memory streamed-row provider**

```python
def test_stream_rows_stops_before_exceeding_cap():
    from recurrent_transformer.corpus import collect_streamed_documents
    rows = ({"text": f"document {i} 中文"} for i in range(1000))
    docs = collect_streamed_documents(rows, text_key="text", max_bytes=100)
    assert docs
    assert sum(len(x.encode("utf-8")) for x in docs) <= 100


def test_empty_stream_has_actionable_error():
    import pytest
    from recurrent_transformer.corpus import collect_streamed_documents
    with pytest.raises(ValueError, match="no usable documents"):
        collect_streamed_documents(iter(()), text_key="text", max_bytes=100)
```

- [ ] **Step 2: Run and verify missing collector failure**

Run: `python -m pytest tests/test_corpus.py -v`

Expected: FAIL because `collect_streamed_documents` does not exist.

- [ ] **Step 3: Implement provider-independent caps, then wire pinned sources**

Implement the collector before importing `datasets`. Add a lazy `load_dataset(..., streaming=True)` adapter with descriptive network errors. Verify current dataset cards and licenses, then pin the exact dataset ID/config/revision and text key for FineWeb-Edu and cleaned Chinese Wikipedia in `configs/smoke.yaml`. Record those identifiers and licenses in README. Before collection, check that configured artifact budget is below available disk space with a 2 GB safety reserve.

- [ ] **Step 4: Run offline tests, then acquire bounded samples**

Run: `python -m pytest tests/test_corpus.py -v && recurrent-transformer prepare-data --config configs/smoke.yaml`

Expected: tests PASS; command writes bounded English and Chinese normalized files and prints bytes/documents per language. If external DNS is unavailable, capture the actionable error and run the documented local fixture command instead; do not weaken the offline tests.

- [ ] **Step 5: Commit**

```bash
git add src/recurrent_transformer/corpus.py src/recurrent_transformer/cli.py configs/smoke.yaml tests/test_corpus.py
git commit -m "feat: add capped streaming corpus acquisition"
```

### Task 8: MPS Probe, Full Smoke Run, Documentation, and Evidence

**Files:**
- Modify: `src/recurrent_transformer/train.py`
- Modify: `src/recurrent_transformer/cli.py`
- Create: `README.md`
- Modify: `tests/test_train.py`

- [ ] **Step 1: Add failing device and non-finite-state tests**

```python
def test_auto_device_is_explicit():
    from recurrent_transformer.train import select_device
    assert select_device("auto").type in {"mps", "cpu"}


def test_nonfinite_loss_stops_training(tmp_path):
    import pytest
    from recurrent_transformer.train import ensure_finite
    with pytest.raises(FloatingPointError, match="non-finite loss"):
        ensure_finite(float("nan"), what="loss", step=1)
```

- [ ] **Step 2: Run and verify the finite-check failure**

Run: `python -m pytest tests/test_train.py -v`

Expected: FAIL because `ensure_finite` is not implemented.

- [ ] **Step 3: Implement preflight and one-step memory probe**

Add device/dtype validation, disk preflight, one-step forward/backward probe, peak-memory logging when supported, and activation checkpointing around each recurrent-core pass. Never silently fall back after training begins; log the selected device before model allocation and provide a command-line `--device cpu` remedy on MPS failure.

- [ ] **Step 4: Write README with exact reproducibility boundaries**

Document environment creation, test commands, bounded data acquisition, offline fallback, tokenizer training, 100M smoke training, resume, recurrence comparison, artifact layout, the two paper citations, the public Huginn repository, verified simplifications, and the unverified nature of the video's Astra claim. State plainly that 20 steps do not produce a useful language model.

- [ ] **Step 5: Run static and automated verification**

Run: `python -m pytest -q && python -m compileall -q src && git diff --check`

Expected: tests PASS, compilation succeeds, and `git diff --check` is silent.

- [ ] **Step 6: Run the offline tiny acceptance pipeline**

Run: `recurrent-transformer tiny-pipeline --english tests/fixtures/en.txt --chinese tests/fixtures/zh.txt --output artifacts/tiny --steps 1`

Expected: tokenizer, checkpoint, metrics, and non-empty generated text are produced under `artifacts/tiny`.

- [ ] **Step 7: Run the 100M MPS memory probe and smoke training**

Run: `recurrent-transformer train --config configs/smoke.yaml --memory-probe && recurrent-transformer train --config configs/smoke.yaml`

Expected: model reports 95M–105M unique trainable parameters; the probe finishes; training reaches step 20 and writes its checkpoint. If MPS rejects an operation, run the documented CPU command and record the limitation accurately.

- [ ] **Step 8: Reload and compare recurrence counts**

Run: `recurrent-transformer generate --checkpoint artifacts/smoke/checkpoint-000020.pt --prompt '循环 Transformer can' --recurrences 1,2,4 --max-new-tokens 24`

Expected: the checkpoint reloads and produces labeled outputs for recurrence counts 1, 2, and 4.

- [ ] **Step 9: Record actual evidence and commit**

Add measured parameter count, device, elapsed time, peak memory when available, completed steps, final smoke loss, artifact sizes, and any corpus-network fallback to README. Do not claim quality gains from generated samples.

```bash
git add README.md src/recurrent_transformer/train.py src/recurrent_transformer/cli.py tests/test_train.py configs
git commit -m "docs: verify local recurrent transformer smoke run"
```

## Final Verification

- [ ] Run `python -m pytest -q` and retain the pass count.
- [ ] Run `git diff --check` and verify no whitespace errors.
- [ ] Run `git status --short` and inspect every remaining path.
- [ ] Confirm README evidence matches command output and does not describe Astra as verified.
- [ ] Confirm downloaded corpora and checkpoints remain ignored by Git.

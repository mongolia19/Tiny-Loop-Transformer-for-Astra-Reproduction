# 33M disk streaming training

Goal: replace the stopped 97M physical-parameter run with a fresh 33,104,384-parameter model, fixed loop=2, sequence length 2048, and at least 662,087,680 training tokens.

Architecture: retain tied 16K embeddings, use width 512 / 8 heads / SwiGLU 1344 and 2 prelude + 4 shared core + 2 coda blocks. Reuse the existing bilingual tokenizer. Stream normalized source documents to text and uint16 binary files, with disk-backed document hashes for deduplication. Mix English and Chinese blocks on demand using bounded disk reads. Use BF16, microbatch 1, accumulation 4, gradient checkpointing, and atomic latest checkpoints. The new finite token budget supersedes the old loss-target continuation.

- [ ] Disable old monitor with a user-stop marker; verify no live old trainer or restart watcher.
- [ ] Add bounded text iteration and disk token batches with cursor state; test lazy consumption, packing, alternating language blocks, truncation validation, and resume equivalence.
- [ ] Add a streaming corpus preparation script. Pin remote revisions, count actual encoded tokens, retain source metadata and tokenizer checksum. Fail if source exhaustion leaves the target unmet. Only a complete manifest authorizes training.
- [ ] Make train CLI stream local text or prepared tokens. Reuse tokenizer on resume; restore batch cursor and recurrent RNG. Validate model/tokenizer/sequence compatibility.
- [ ] Add configs/33m-loop2-2048.yaml and run all existing plus new tests.
- [ ] Build full corpus, verify token budget and disk size. Run representative 2048/loop2 BF16 MPS steps and measure memory. Launch detached training in artifacts/mps-33m-loop2-2048 and verify advancing finite-loss steps plus checkpoint.

Validation commands: PYTHONPATH=src .venv-mps/bin/python -m pytest -q; git diff --check. Preserve pre-existing untracked scripts/wait_next_checkpoint.sh and uv.lock.

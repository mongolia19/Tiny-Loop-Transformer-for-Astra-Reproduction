import numpy as np
import pytest
import torch

from recurrent_transformer.dataset import DiskTokenBatches, pack_documents, pack_documents_to_disk
from recurrent_transformer.corpus import iter_local_documents


class FakeTokenizer:
    bos_id, eos_id, vocab_size = 2, 3, 256

    def encode(self, text):
        return [ord(c) for c in text]


def test_disk_packing_matches_legacy_and_resume(tmp_path):
    docs = ["hello", "world", "streaming"]
    tok = FakeTokenizer()
    expected = pack_documents(docs, tok, 4)
    path = tmp_path / "tokens.bin"
    batches = pack_documents_to_disk(iter(docs), tok, 4, path)
    iterator = iter(batches)
    assert torch.equal(next(iterator), expected[:1])
    saved = batches.state_dict()
    restored = DiskTokenBatches(path, seq_len=4, batch_size=1)
    restored.load_state_dict(saved)
    assert torch.equal(torch.cat(list(restored)), expected[1:])
    assert torch.equal(torch.cat(list(batches)), expected[1:])
    assert torch.equal(torch.cat(list(batches)), expected)


def test_round_robin_unequal_files(tmp_path):
    paths = [tmp_path / "en.bin", tmp_path / "zh.bin"]
    np.array([10]*4 + [11]*4 + [12]*4, dtype="<i4").tofile(paths[0])
    np.array([20]*4, dtype="<i4").tofile(paths[1])
    assert [x[0, 0].item() for x in DiskTokenBatches(paths, seq_len=4, batch_size=1)] == [10, 20, 11, 12]


def test_rejects_partial_tokens_and_changed_resume_layout(tmp_path):
    path = tmp_path / "bad.bin"
    path.write_bytes(b"123")
    with pytest.raises(ValueError, match="partial"):
        DiskTokenBatches(path, seq_len=4, batch_size=1)
    path.write_bytes(bytes(64))
    state = DiskTokenBatches(path, seq_len=4, batch_size=1).state_dict()
    with pytest.raises(ValueError, match="seq_len"):
        DiskTokenBatches(path, seq_len=8, batch_size=1).load_state_dict(state)


def test_local_text_reader_is_lazy(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text('{"text": " hello "}\ninvalid json\n')
    iterator = iter(iter_local_documents([path], 100))
    assert next(iterator) == "hello"
    with pytest.raises(ValueError):
        next(iterator)

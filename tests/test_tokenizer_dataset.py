from recurrent_transformer.dataset import pack_documents
from recurrent_transformer.tokenizer import train_tokenizer


def test_tiny_tokenizer_and_fixed_blocks(tmp_path):
    docs = ["hello recurrent world " * 40, "你好 循环 世界 " * 40]
    tokenizer = train_tokenizer(docs, tmp_path, vocab_size=128)
    assert tokenizer.pad_id >= 0 and tokenizer.bos_id >= 0 and tokenizer.eos_id >= 0
    blocks = pack_documents(docs, tokenizer, seq_len=16)
    assert blocks.ndim == 2
    assert blocks.shape[1] == 16

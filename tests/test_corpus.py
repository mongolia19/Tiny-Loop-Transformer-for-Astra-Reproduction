from recurrent_transformer.corpus import normalize_text, read_local_documents


def test_normalize_and_deduplicate_with_byte_cap(tmp_path):
    path = tmp_path / "docs.txt"
    path.write_text("  Hello   world  \n\nHello world\n独立 文档\n", encoding="utf-8")
    docs = read_local_documents([path], max_bytes=40)
    assert docs == ["Hello world", "独立 文档"]
    assert sum(len(x.encode("utf-8")) for x in docs) <= 40


def test_normalize_text_uses_nfkc_and_collapses_whitespace():
    assert normalize_text("Ａ  B\tC") == "A B C"


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

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

from recurrent_transformer.cli import build_parser, run_tiny_pipeline


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


def test_parser_exposes_train_and_generate_commands():
    parser = build_parser()
    train = parser.parse_args([
        "train", "--config", "configs/smoke.yaml", "--english", "en.txt",
        "--chinese", "zh.txt", "--output", "artifacts/smoke",
    ])
    generate = parser.parse_args([
        "generate", "--checkpoint", "model.pt", "--prompt", "hello",
        "--recurrences", "1,2,4",
    ])
    assert train.handler.__name__ == "_train_command"
    assert generate.handler.__name__ == "_generate_command"

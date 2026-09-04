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


def test_incompatible_checkpoint_reports_the_field(tmp_path):
    import pytest

    model = RecurrentTransformer(tiny_config())
    path = save_checkpoint(tmp_path / "step.pt", model=model, step=1)
    incompatible = tiny_config().__class__(**{**tiny_config().__dict__, "vocab_size": 65})
    with pytest.raises(ValueError, match="vocab_size"):
        load_checkpoint(path, expected_config=incompatible)

import torch

from recurrent_transformer.model import RecurrentTransformer
from recurrent_transformer.train import train_steps
from tests.test_model import tiny_config


def test_one_step_updates_shared_core_and_writes_checkpoint(tmp_path):
    model = RecurrentTransformer(tiny_config())
    batches = [torch.randint(0, 64, (2, 8))]
    before = next(model.recurrent_core.parameters()).detach().clone()
    result = train_steps(
        model,
        batches,
        device="cpu",
        max_steps=1,
        min_recurrences=1,
        max_recurrences=2,
        output_dir=tmp_path,
        seed=7,
    )
    after = next(model.recurrent_core.parameters()).detach()
    assert result.steps == 1 and result.losses[0] > 0
    assert not torch.equal(before, after)
    assert (tmp_path / "checkpoint-000001.pt").exists()

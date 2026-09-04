import torch

from recurrent_transformer.config import ModelConfig
from recurrent_transformer.model import RecurrentTransformer


def tiny_config():
    return ModelConfig(
        vocab_size=64,
        d_model=32,
        n_heads=4,
        mlp_hidden_size=64,
        prelude_layers=1,
        recurrent_core_layers=2,
        coda_layers=1,
        max_seq_len=16,
        min_recurrences=1,
        max_recurrences=3,
    )


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

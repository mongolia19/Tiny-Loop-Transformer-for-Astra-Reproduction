import torch

from recurrent_transformer.generate import generate_ids
from recurrent_transformer.model import RecurrentTransformer
from tests.test_model import tiny_config


def test_greedy_generation_honors_length_and_recurrence():
    model = RecurrentTransformer(tiny_config()).eval()
    prompt = torch.tensor([[2, 5, 6]])
    output = generate_ids(
        model,
        prompt,
        max_new_tokens=4,
        num_recurrences=3,
        temperature=0.0,
    )
    assert output.shape == (1, 7)
    torch.testing.assert_close(output[:, :3], prompt)

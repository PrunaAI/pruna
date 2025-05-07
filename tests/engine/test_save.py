import os
import pytest
from pruna.engine.pruna_model import PrunaModel

@pytest.mark.skipif("HF_TOKEN" not in os.environ, reason="HF_TOKEN environment variable is not set, skipping tests.")
@pytest.mark.slow
@pytest.mark.cpu
@pytest.mark.parametrize(
    "repo_id",
    [
        ("PrunaAI/tiny-random-llama4-smashed"),
        ("PrunaAI/tiny-stable-diffusion-pipe-smashed"),
    ],
)
def test_save_to_hub(repo_id: str) -> None:
    """Test PrunaModel.save_to_hub with different models."""

    model = PrunaModel.from_hub(repo_id)
    model.save_to_hub(repo_id, private=False)
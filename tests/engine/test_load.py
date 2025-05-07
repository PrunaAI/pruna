import pytest

from pruna.engine.pruna_model import PrunaModel

@pytest.mark.parametrize(
    "model_name, expected_output, should_raise",
    [
        ("PrunaAI/tiny-random-llama4-smashed", "PrunaModel", False),
        ("PrunaAI/tiny-stable-diffusion-pipe-smashed", "PrunaModel", False),
        ("NonExistentRepo/model", None, True),
    ],
)
@pytest.mark.cpu
def test_pruna_model_from_hub(model_name: str, expected_output: str, should_raise: bool) -> None:
    """Test PrunaModel.from_hub."""
    if should_raise:
        with pytest.raises(Exception):
            PrunaModel.from_hub(model_name)
    else:
        model = PrunaModel.from_hub(model_name)
        assert model.__class__.__name__ == expected_output


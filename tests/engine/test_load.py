import pytest
import torch
from pathlib import Path

from pruna.engine.pruna_model import PrunaModel
from pruna.engine.load import LOAD_FUNCTIONS
from pruna.engine.load import load_pruna_model
from pruna.config.smash_config import SmashConfig


@pytest.mark.parametrize(
    "model_name, expected_output, should_raise",
    [
        ("PrunaAI/test-load-tiny-random-llama4-smashed", "PrunaModel", False),
        ("PrunaAI/test-load-tiny-stable-diffusion-pipe-smashed", "PrunaModel", False),
        ("NonExistentRepo/model", None, True),
    ],
)
@pytest.mark.cpu
def test_pruna_model_from_hub(model_name: str, expected_output: str, should_raise: bool) -> None:
    """Test PrunaModel.from_hub."""
    if should_raise:
        with pytest.raises(Exception):
            PrunaModel.from_hub(model_name, force_download=True)
    else:
        model = PrunaModel.from_hub(model_name, force_download=True)
        assert model.__class__.__name__ == expected_output


@pytest.mark.parametrize(
    "path_type",
    ["string", "pathlib"],
)
@pytest.mark.cpu
def test_load_pruna_model_path_types(tmp_path, path_type: str) -> None:
    """Test loading PrunaModel with different path types (str vs Path)."""
    model_path = tmp_path / "test_model"
    model_path.mkdir()

    config = SmashConfig()
    config.load_fns = ["pickled"]
    config.save_to_json(model_path)

    dummy_model = torch.nn.Linear(10, 5)
    torch.save(dummy_model, model_path / "optimized_model.pt")

    if path_type == "string":
        test_path = str(model_path)
    else:
        test_path = Path(model_path)

    loaded_model, loaded_config = load_pruna_model(test_path)

    assert isinstance(loaded_model, torch.nn.Linear)
    assert loaded_model.in_features == 10
    assert loaded_model.out_features == 5


@pytest.mark.parametrize(
    "path_type",
    ["string", "pathlib"],
)
@pytest.mark.cpu
def test_load_functions_path_types(tmp_path, path_type: str) -> None:
    """Test individual load functions with different path types."""
    model_path = tmp_path / "pickled_test"
    model_path.mkdir()

    dummy_model = torch.nn.Linear(5, 3)
    torch.save(dummy_model, model_path / "optimized_model.pt")

    if path_type == "string":
        test_path = str(model_path)
    else:
        test_path = Path(model_path)

    loaded_model = LOAD_FUNCTIONS.pickled(test_path)
    assert isinstance(loaded_model, torch.nn.Linear)
    assert loaded_model.in_features == 5
    assert loaded_model.out_features == 3

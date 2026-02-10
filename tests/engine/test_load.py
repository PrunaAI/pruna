from pathlib import Path
from typing import Any
from unittest.mock import patch
import tempfile

import pytest
import torch
from huggingface_hub import snapshot_download

from pruna.engine.pruna_model import PrunaModel
from pruna.engine.load import LOAD_FUNCTIONS, filter_load_kwargs, load_diffusers_model, load_transformers_model
from pruna.config.smash_config import SmashConfig
from pruna.algorithms.base.registry import AlgorithmRegistry
from pruna.engine.save import SAVE_FUNCTIONS
from pruna import smash

@pytest.mark.parametrize(
    "model_name, expected_output, should_raise",
    [
        ("pruna-test/test-load-tiny-stable-diffusion-pipe-smashed", "PrunaModel", False),
        ("NonExistentRepo/model", None, True),
    ],
)
@pytest.mark.cpu
def test_pruna_model_from_pretrained(model_name: str, expected_output: str, should_raise: bool) -> None:
    """Test PrunaModel.from_pretrained."""
    if should_raise:
        with pytest.raises(Exception):
            PrunaModel.from_pretrained(model_name, force_download=True)
    else:
        model = PrunaModel.from_pretrained(model_name, force_download=True)
        assert model.__class__.__name__ == expected_output


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
    loaded_model = LOAD_FUNCTIONS.pickled(test_path, SmashConfig())
    assert isinstance(loaded_model, torch.nn.Linear)
    assert loaded_model.in_features == 5
    assert loaded_model.out_features == 3


@pytest.mark.parametrize(
    "func_def, kwargs, expected_output, test_name",
    [
        # Test with function that accepts **kwargs
        (
            lambda: lambda a, b, **kwargs: None,
            {"a": 1, "b": "test", "c": 3, "d": 4},
            {"a": 1, "b": "test", "c": 3, "d": 4},
            "with_kwargs"
        ),
        # Test with function that doesn't accept **kwargs
        (
            lambda: lambda a, b: None,
            {"a": 1, "b": "test", "c": 3, "d": 4},
            {"a": 1, "b": "test"},
            "without_kwargs"
        ),
        # Test with only valid parameters
        (
            lambda: lambda a, b: None,
            {"a": 1, "b": "test"},
            {"a": 1, "b": "test"},
            "no_invalid"
        ),
        # Test with empty kwargs
        (
            lambda: lambda a, b: None,
            {},
            {},
            "empty"
        ),
        # Test with all invalid parameters
        (
            lambda: lambda a, b: None,
            {"c": 3, "d": 4},
            {},
            "all_invalid"
        ),
        # Test with default parameters
        (
            lambda: lambda a=1, b="default": None,
            {"a": 2, "c": 3},
            {"a": 2},
            "with_defaults"
        ),
    ],
)
def test_filter_load_kwargs(func_def, kwargs, expected_output, test_name):
    """Test filter_load_kwargs with various function signatures and kwargs combinations."""
    func = func_def()
    filtered = filter_load_kwargs(func, kwargs)
    assert filtered == expected_output, f"Test {test_name} failed"

@pytest.mark.cpu
@pytest.mark.parametrize("model_id", ["katuni4ka/tiny-random-flux"])
def test_load_diffusers_model_without_smash_config(model_id: str) -> None:
    """Test loading a diffusers model without a SmashConfig."""
    download_directory = snapshot_download(model_id)
    model = load_diffusers_model(download_directory)
    assert model is not None


@pytest.mark.cpu
@pytest.mark.parametrize("model_id", ["yujiepan/llama-3-tiny-random"])
def test_load_transformers_model_without_smash_config(model_id: str) -> None:
    """Test loading a diffusers model without a SmashConfig."""
    download_directory = snapshot_download(model_id)
    model = load_transformers_model(download_directory)
    assert model is not None

@pytest.mark.cpu
@pytest.mark.parametrize(
    "model_fixture", ("resnet_18",),
    indirect=["model_fixture"],
)
def test_resmash(model_fixture: tuple[Any, SmashConfig]) -> None:
    model, smash_config = model_fixture

    # this test assumes torch_structured is defining a custom save function and torch_compile is reapplied to trigger partial resmash
    assert AlgorithmRegistry["torch_structured"].save_fn not in [SAVE_FUNCTIONS.reapply, SAVE_FUNCTIONS.save_before_apply]
    assert AlgorithmRegistry["torch_compile"].save_fn is SAVE_FUNCTIONS.save_before_apply
    smash_config.add(["torch_structured", "torch_compile"])

    smashed_model = smash(model, smash_config=smash_config)

    with tempfile.TemporaryDirectory() as temp_dir:
        smashed_model.save_pretrained(temp_dir)

        def assert_torch_compile_no_torch_structured(model: Any, smash_config: SmashConfig) -> Any:
            """Monkey patch smash function at the end of resmash to compare the subset of algorithms that are reapplied"""
            is_torch_structured_skipped = isinstance(smash_config["torch_structured"], bool) and not smash_config["torch_structured"]
            is_torch_compile_reapplied = isinstance(smash_config["torch_compile"], bool) and smash_config["torch_compile"]
            assert is_torch_structured_skipped, "torch_structured was expected to be skipped when resmashing, but was activated in the smash config."
            assert is_torch_compile_reapplied, "torch_compile was expected to be reapplied when resmashing, but was not activated in the smash config."
            model.__has_been_monkeypatch_loaded = True
            return model

        with patch("pruna.smash.smash", assert_torch_compile_no_torch_structured):
            # load model with monkey patch tests and confirm that the monkey patch was applied
            loaded_model = PrunaModel.from_pretrained(temp_dir)
            is_test_monkeypatch_applied = getattr(loaded_model, "__has_been_monkeypatch_loaded", False)
            assert is_test_monkeypatch_applied, "Monkey patch was not applied, test did not run as expected."

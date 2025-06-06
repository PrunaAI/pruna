import pytest

from pruna.engine.pruna_model import PrunaModel
from pruna.engine.load import filter_load_kwargs

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


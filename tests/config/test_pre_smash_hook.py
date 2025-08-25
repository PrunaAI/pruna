import torch
import pytest

from pruna import SmashConfig, smash
from pruna.algorithms.quantization.huggingface_llm_int8 import LLMInt8Quantizer
from pruna.config.smash_config import SmashConfigPrefixWrapper

from typing import Any


@pytest.mark.cpu
@pytest.mark.parametrize(
    "model_fixture",
    [
        pytest.param("opt_tiny_random", marks=pytest.mark.cpu),
    ],
    indirect=["model_fixture"],
)
def test_pre_smash_hook(monkeypatch: pytest.MonkeyPatch, model_fixture: tuple[Any, SmashConfig]) -> None:
    """Test the pre_smash_hook method."""
    model, smash_config = model_fixture

    pre_smash_hook_called = False
    def mock_pre_smash_hook(self: LLMInt8Quantizer, model: Any, smash_config: SmashConfigPrefixWrapper) -> None:
        nonlocal pre_smash_hook_called
        pre_smash_hook_called = True

    monkeypatch.setattr(LLMInt8Quantizer, "_pre_smash_hook", mock_pre_smash_hook)

    # use any oss algorithm to test the pre_smash_hook
    smash_config["quantizer"] = "llm_int8"

    smash(model, smash_config)

    assert pre_smash_hook_called

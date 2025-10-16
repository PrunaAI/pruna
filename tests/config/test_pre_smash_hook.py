import torch
import pytest

from pruna import SmashConfig, smash
from pruna.algorithms.huggingface_llm_int8 import LLMInt8
from pruna.config.smash_config import SmashConfigPrefixWrapper

from typing import Any


@pytest.mark.cuda
@pytest.mark.parametrize("model_fixture", ["opt_tiny_random"], indirect=["model_fixture"])
def test_pre_smash_hook(monkeypatch: pytest.MonkeyPatch, model_fixture: tuple[Any, SmashConfig]) -> None:
    """Test the pre_smash_hook method."""
    model, smash_config = model_fixture

    pre_smash_hook_called = False
    def mock_pre_smash_hook(self: LLMInt8, model: Any, smash_config: SmashConfigPrefixWrapper) -> None:
        nonlocal pre_smash_hook_called
        pre_smash_hook_called = True

    monkeypatch.setattr(LLMInt8, "_pre_smash_hook", mock_pre_smash_hook)

    # use any oss algorithm to test the pre_smash_hook
    smash_config.add("llm_int8")

    smash(model, smash_config)

    assert pre_smash_hook_called

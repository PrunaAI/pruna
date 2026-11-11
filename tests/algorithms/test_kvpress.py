from __future__ import annotations

import pytest
import torch
from ConfigSpace.exceptions import IllegalValueError
from transformers import LlamaConfig, LlamaForCausalLM

from pruna import SmashConfig
from pruna.algorithms.kvpress import PRESS_TYPES, KVPress


@pytest.mark.requires_kvpress
@pytest.mark.cpu
def test_press_types_exclude_exit_time_presses() -> None:
    """Verify that PRESS_TYPES excludes presses that only compress on context manager exit."""
    import kvpress

    assert "KVzipPress" not in PRESS_TYPES
    assert "FastKVzipPress" not in PRESS_TYPES

    for press_name in PRESS_TYPES:
        assert hasattr(kvpress, press_name), f"{press_name} is in PRESS_TYPES but missing from kvpress"
        press_cls = getattr(kvpress, press_name)
        assert not hasattr(press_cls, "compress_post"), (
            f"{press_name} implements compress_post and should not be in PRESS_TYPES"
        )


@pytest.mark.requires_kvpress
@pytest.mark.cpu
@pytest.mark.parametrize("press_type", ["KVzipPress", "FastKVzipPress"])
def test_exit_time_presses_rejected_at_config_time(press_type: str) -> None:
    """Verify that configuring KVzipPress or FastKVzipPress in SmashConfig raises IllegalValueError."""
    smash_config = SmashConfig()
    smash_config.add("kvpress")
    with pytest.raises(IllegalValueError):
        smash_config.add({"kvpress_press_type": press_type})


@pytest.mark.requires_kvpress
@pytest.mark.cpu
@pytest.mark.parametrize("press_type", ["KVzipPress", "FastKVzipPress"])
def test_exit_time_presses_raise_value_error(press_type: str) -> None:
    """Verify that attempting to apply KVzipPress or FastKVzipPress raises a clear ValueError."""
    config = LlamaConfig(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).eval()

    algorithm = KVPress()
    fake_config = {
        "press_type": press_type,
        "compression_ratio": 0.5,
        "press_kwargs": None,
    }

    with pytest.raises(ValueError, match="is not supported because it compresses only on context manager exit"):
        algorithm._apply(model, fake_config)


@pytest.mark.requires_kvpress
@pytest.mark.cpu
def test_unknown_press_type_raises_value_error() -> None:
    """Verify that an unknown press type raises ValueError."""
    config = LlamaConfig(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).eval()

    algorithm = KVPress()
    fake_config = {
        "press_type": "NonExistentPress",
        "compression_ratio": 0.5,
        "press_kwargs": None,
    }

    with pytest.raises(ValueError, match="Unknown or unsupported press type"):
        algorithm._apply(model, fake_config)


@pytest.mark.requires_kvpress
@pytest.mark.cpu
def test_supported_press_generate_execution() -> None:
    """Verify that a supported press is applied to model.generate and generates tokens."""
    config = LlamaConfig(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).eval()

    algorithm = KVPress()
    fake_config = {
        "press_type": "ExpectedAttentionPress",
        "compression_ratio": 0.5,
        "press_kwargs": None,
    }

    smashed = algorithm._apply(model, fake_config)
    assert hasattr(smashed, "_kvpress_press")
    assert hasattr(smashed, "_kvpress_original_generate")

    input_ids = torch.randint(0, 100, (1, 16))
    attention_mask = torch.ones_like(input_ids)
    output = smashed.generate(input_ids, attention_mask=attention_mask, max_new_tokens=4, pad_token_id=0)

    assert output.shape == (1, 20)

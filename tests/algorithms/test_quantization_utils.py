from __future__ import annotations

from typing import Any

import pytest
import torch

from pruna.algorithms.global_utils.quantization.dtypes import parse_float8_dtype
from pruna.algorithms.global_utils.quantization.swap_linear import swap_linear
from pruna.algorithms.global_utils.quantization.symmetric_scale import amax_to_scale, scale_and_clamp


class _IdentityReplacement(torch.nn.Module):
    """Stand-in module used to detect that a linear layer was swapped."""

    def __init__(self, tag: str) -> None:
        super().__init__()
        self.tag = tag


def _replace_linear_with_identity(parent: torch.nn.Module, child_name: str, tag: str = "swapped") -> None:
    """Quantize callback that replaces a child with ``_IdentityReplacement``."""
    setattr(parent, child_name, _IdentityReplacement(tag))


@pytest.mark.cpu
def test_swap_linear_replaces_nested_linear() -> None:
    """Test that ``swap_linear`` follows a dotted path and applies the callback in place."""
    model = torch.nn.Sequential(torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.ReLU()))

    swap_linear(model, _replace_linear_with_identity, path="0.0", kwargs={"tag": "inner"})

    replacement = model[0][0]
    assert isinstance(replacement, _IdentityReplacement)
    assert replacement.tag == "inner"
    assert isinstance(model[0][1], torch.nn.ReLU)


@pytest.mark.cpu
def test_swap_linear_skips_non_linear_child() -> None:
    """Test that ``swap_linear`` does not call the callback when the path is not an ``nn.Linear``."""
    model = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(2, 2))
    calls: list[tuple[Any, ...]] = []

    def record_call(parent: torch.nn.Module, child_name: str, **kwargs: Any) -> None:
        calls.append((parent, child_name, kwargs))

    swap_linear(model, record_call, path="0")

    assert calls == []
    assert isinstance(model[0], torch.nn.ReLU)


@pytest.mark.cpu
def test_swap_linear_empty_path_raises() -> None:
    """Test that ``swap_linear`` rejects an empty path."""
    model = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="empty path"):
        swap_linear(model, _replace_linear_with_identity, path="")


@pytest.mark.cpu
def test_amax_to_scale_non_zero_amax() -> None:
    """Test the nominal per-tensor scale ``max_val / amax``."""
    amax = torch.tensor(2.0)
    max_val = 448.0
    scale = amax_to_scale(amax, max_val)
    torch.testing.assert_close(scale, torch.tensor(224.0))


@pytest.mark.cpu
def test_amax_to_scale_zero_amax() -> None:
    """Test that a zero amax is floored so the scale stays finite."""
    amax = torch.tensor(0.0)
    max_val = 448.0
    scale = amax_to_scale(amax, max_val)
    torch.testing.assert_close(scale, torch.tensor(max_val / 1e-12))
    assert torch.isfinite(scale)


@pytest.mark.cpu
def test_scale_and_clamp_maps_and_clips() -> None:
    """Test that values are scaled then clipped to ``[-max_val, max_val]``."""
    x = torch.tensor([-3.0, 0.5, 3.0])
    scale = torch.tensor(2.0)
    max_val = 4.0
    out = scale_and_clamp(x, scale, max_val)
    torch.testing.assert_close(out, torch.tensor([-4.0, 1.0, 4.0]))


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("torch.float8_e4m3fn", torch.float8_e4m3fn),
        ("torch.float8_e5m2", torch.float8_e5m2),
    ],
)
def test_parse_float8_dtype_known_names(name: str, expected: torch.dtype) -> None:
    """Test that known float8 dtype names resolve to the matching torch dtype."""
    assert parse_float8_dtype(name) is expected


@pytest.mark.cpu
def test_parse_float8_dtype_unknown_name_raises() -> None:
    """Test that an unsupported dtype name raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported float8 dtype"):
        parse_float8_dtype("torch.float16")

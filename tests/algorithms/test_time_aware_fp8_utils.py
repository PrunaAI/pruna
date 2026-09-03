from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from pruna.algorithms.global_utils.quantization.symmetric_scale import amax_to_scale
from pruna.algorithms.time_aware_fp8_diffusers import TimeAwareFp8Diffusers
from pruna.algorithms.time_aware_fp8_diffusers.utils import (
    TimeAwareFp8Linear,
    TimeAwareScaleHelper,
)


class _Denoiser(torch.nn.Module):
    """Minimal denoiser whose forward takes a ``timestep`` argument."""

    def forward(self, x: torch.Tensor, timestep: torch.Tensor | None = None) -> torch.Tensor:
        """Return the input unchanged."""
        return x


@pytest.mark.cpu
def test_model_check_rejects_denoiser_without_timestep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a diffusers-like backbone whose forward has no timestep is rejected."""
    monkeypatch.setattr(
        "pruna.algorithms.time_aware_fp8_diffusers.time_aware_fp8_diffusers.is_diffusers_model",
        lambda model: True,
    )

    class NoTimestepDenoiser(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return the input unchanged."""
            return x

    model = SimpleNamespace(transformer=NoTimestepDenoiser())
    assert TimeAwareFp8Diffusers().model_check_fn(model) is False


@pytest.mark.cpu
def test_model_check_accepts_denoiser_with_timestep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a diffusers-like backbone whose forward takes timestep is accepted."""
    monkeypatch.setattr(
        "pruna.algorithms.time_aware_fp8_diffusers.time_aware_fp8_diffusers.is_diffusers_model",
        lambda model: True,
    )
    model = SimpleNamespace(transformer=_Denoiser())
    assert TimeAwareFp8Diffusers().model_check_fn(model) is True


@pytest.mark.cpu
def test_scale_helper_captures_timestep_from_forward() -> None:
    """Test that the denoiser pre-hook copies the timestep into the helper."""
    helper = TimeAwareScaleHelper()
    denoiser = _Denoiser()
    helper.register_on_denoiser(denoiser)

    denoiser(torch.zeros(1, 2), timestep=torch.tensor([3.5]))

    assert helper.current_value == 3.5
    torch.testing.assert_close(helper.buffer, torch.tensor([3.5]))


def _prepared_helper_and_layer() -> tuple[TimeAwareScaleHelper, TimeAwareFp8Linear]:
    """Build a single frozen layer and run ``prepare_for_inference`` on its helper."""
    helper = TimeAwareScaleHelper()
    layer = TimeAwareFp8Linear.from_linear(torch.nn.Linear(2, 2), scale_helper=helper)
    layer.input_running_amax_by_timestep[0.0] = torch.tensor(1.0, dtype=torch.float32)
    layer.input_running_amax_by_timestep[1.0] = torch.tensor(2.0, dtype=torch.float32)
    layer.freeze_input_scales()
    helper.prepare_for_inference([layer])
    return helper, layer


@pytest.mark.cpu
def test_prepare_for_inference_stops_calibration_tracking() -> None:
    """Test that the inference table is live and the Python timestep mirror is off."""
    helper, _ = _prepared_helper_and_layer()
    assert helper.scales_initialized is True
    assert helper.track_python_value is False


@pytest.mark.cpu
def test_prepare_for_inference_sets_bin_edges() -> None:
    """Test that the bin edge is the midpoint between calibration timesteps 0.0 and 1.0."""
    helper, _ = _prepared_helper_and_layer()
    torch.testing.assert_close(helper.bin_edges, torch.tensor([0.5]))


@pytest.mark.cpu
def test_prepare_for_inference_builds_all_scales() -> None:
    """Test per-layer, per-bin scales from the recorded amaxes (one layer, two timesteps)."""
    helper, layer = _prepared_helper_and_layer()
    expected_scales = amax_to_scale(torch.tensor([1.0, 2.0]), layer.input_max_value).unsqueeze(0)
    torch.testing.assert_close(helper.all_scales, expected_scales)
    torch.testing.assert_close(helper.all_scales_reciprocal, expected_scales.reciprocal())


@pytest.mark.cpu
def test_prepare_for_inference_links_layer_views() -> None:
    """Test that current-scale buffers start at zero and layer views alias column 0."""
    helper, layer = _prepared_helper_and_layer()
    assert helper.current_scales is not None
    assert helper.current_scales_reciprocal is not None
    assert helper.current_scales.shape == (1, 4)
    torch.testing.assert_close(helper.current_scales[:, 0], torch.zeros(1))
    torch.testing.assert_close(helper.current_scales_reciprocal[:, 0], torch.zeros(1))
    helper.current_scales.fill_(3.0)
    helper.current_scales_reciprocal.fill_(4.0)
    torch.testing.assert_close(layer.input_current_scale, torch.tensor(3.0))
    torch.testing.assert_close(layer.input_current_scale_reciprocal, torch.tensor(4.0))


def _frozen_layer(helper: TimeAwareScaleHelper) -> TimeAwareFp8Linear:
    """Build a frozen layer on ``helper`` with two calibration timesteps."""
    layer = TimeAwareFp8Linear.from_linear(torch.nn.Linear(2, 2), scale_helper=helper)
    layer.input_running_amax_by_timestep[0.0] = torch.tensor(1.0, dtype=torch.float32)
    layer.input_running_amax_by_timestep[1.0] = torch.tensor(2.0, dtype=torch.float32)
    layer.freeze_input_scales()
    return layer


@pytest.mark.cpu
def test_layer_scale_views_are_16_byte_strided() -> None:
    """Each layer view must land on a 16-byte boundary (4 float32s) for cuBLAS."""
    helper = TimeAwareScaleHelper()
    layers = [_frozen_layer(helper) for _ in range(5)]
    helper.prepare_for_inference(layers)
    for slot, layer in enumerate(layers):
        assert layer.input_current_scale is not None
        assert layer.input_current_scale.storage_offset() == slot * 4
        assert layer.input_current_scale_reciprocal.storage_offset() == slot * 4

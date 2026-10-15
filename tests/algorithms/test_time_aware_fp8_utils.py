from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from pruna.algorithms.global_utils.quantization.symmetric_scale import amax_to_scale
from pruna.algorithms.time_aware_fp8_diffusers import TimeAwareFp8Diffusers
from pruna.algorithms.time_aware_fp8_diffusers.utils import (
    TimeAwareFp8Linear,
    TimeAwareScaleHelper,
)


def _denoiser_with_time_arg(arg_name: str) -> torch.nn.Module:
    """Build a denoiser whose ``forward`` exposes ``arg_name`` as the timestep argument."""
    namespace: dict[str, Any] = {}
    exec(f"def forward(self, x, {arg_name}=None):\n    return x\n", namespace)
    return type("Denoiser", (torch.nn.Module,), {"forward": namespace["forward"]})()


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
def test_register_on_denoiser_lists_capturable_parameter_names_on_failure() -> None:
    """The missing-timestep error names every capturable forward argument."""

    class NoTimestepDenoiser(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return the input unchanged."""
            return x

    with pytest.raises(ValueError, match="cannot capture the denoising timestep") as exc_info:
        TimeAwareScaleHelper().register_on_denoiser(NoTimestepDenoiser())

    message = str(exc_info.value)
    for name in TimeAwareScaleHelper.timestep_arg_names:
        assert repr(name) in message


@pytest.mark.cpu
@pytest.mark.parametrize("arg_name", TimeAwareScaleHelper.timestep_arg_names)
def test_model_check_accepts_capturable_timestep_arg(monkeypatch: pytest.MonkeyPatch, arg_name: str) -> None:
    """Test that a denoiser whose time argument is any capturable name is accepted."""
    monkeypatch.setattr(
        "pruna.algorithms.time_aware_fp8_diffusers.time_aware_fp8_diffusers.is_diffusers_model",
        lambda model: True,
    )
    model = SimpleNamespace(transformer=_denoiser_with_time_arg(arg_name))
    assert TimeAwareFp8Diffusers().model_check_fn(model) is True


@pytest.mark.cpu
@pytest.mark.parametrize("arg_name", TimeAwareScaleHelper.timestep_arg_names)
def test_scale_helper_captures_timestep_from_kwargs(arg_name: str) -> None:
    """Test that the denoiser pre-hook copies a named timestep kwarg into the helper."""
    helper = TimeAwareScaleHelper()
    denoiser = _denoiser_with_time_arg(arg_name)
    helper.register_on_denoiser(denoiser)

    denoiser(torch.zeros(1, 2), **{arg_name: torch.tensor([3.5])})

    assert helper.current_value == 3.5
    torch.testing.assert_close(helper.buffer, torch.tensor([3.5]))


@pytest.mark.cpu
@pytest.mark.parametrize("arg_name", TimeAwareScaleHelper.timestep_arg_names)
def test_scale_helper_captures_timestep_from_positional_args(arg_name: str) -> None:
    """Test that the denoiser pre-hook copies a positional timestep into the helper."""
    helper = TimeAwareScaleHelper()
    denoiser = _denoiser_with_time_arg(arg_name)
    helper.register_on_denoiser(denoiser)

    denoiser(torch.zeros(1, 2), torch.tensor([2.25]))

    assert helper.current_value == 2.25
    torch.testing.assert_close(helper.buffer, torch.tensor([2.25]))


@pytest.mark.cpu
def test_finalize_calibration_raises_when_a_layer_was_not_visited() -> None:
    """Any unvisited swapped Linear is a targeting error, even if other layers calibrated."""
    helper = TimeAwareScaleHelper()
    observed = TimeAwareFp8Linear.from_linear(torch.nn.Linear(2, 2), scale_helper=helper)
    unused = TimeAwareFp8Linear.from_linear(torch.nn.Linear(2, 2), scale_helper=helper)
    observed.input_running_amax_by_timestep[0.0] = torch.tensor(1.0, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="not visited during calibration"):
        TimeAwareFp8Diffusers._finalize_calibration([unused, observed], helper)


@pytest.mark.cpu
def test_finalize_calibration_raises_when_no_layer_was_visited() -> None:
    """Freeze fails when every swapped Linear missed calibration."""
    helper = TimeAwareScaleHelper()
    unused = TimeAwareFp8Linear.from_linear(torch.nn.Linear(2, 2), scale_helper=helper)
    with pytest.raises(RuntimeError, match="not visited during calibration"):
        TimeAwareFp8Diffusers._finalize_calibration([unused], helper)


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

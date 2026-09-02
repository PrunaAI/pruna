# Copyright 2025 - Pruna AI GmbH. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import inspect
from typing import Any, Optional

import torch
import torch.nn.functional as f

from pruna.algorithms.global_utils.quantization.symmetric_scale import amax_to_scale, scale_and_clamp


class TimeAwareScaleHelper:
    """
    Shared holder of the current denoising timestep and per-layer activation scales.

    A ``forward_pre_hook`` on the denoiser copies the timestep tensor into :attr:`buffer`.

    During calibration the helper is used to provide the current timestep to the layers.
    It mirrors the timestep as a Python float in :attr:`current_value`,
    allowing layers to key their amax statistics by exact timestep value.
    This mirror incurs a GPU sync, so it is disabled post-calibration.

    Post-calibration, the helper provides the timestep-indexed quantization scales to the layers.
    On every denoiser forward, the helper uses the current timestep to provide a view into the current scales.
    """

    def __init__(self) -> None:
        self.buffer: torch.Tensor | None = None
        self.current_value: float = 0.0
        self.track_python_value: bool = True
        self.scales_initialized: bool = False
        self.bin_edges: torch.Tensor | None = None
        self.all_scales: torch.Tensor | None = None
        self.all_scales_reciprocal: torch.Tensor | None = None
        self.current_scales: torch.Tensor | None = None
        self.current_scales_reciprocal: torch.Tensor | None = None

    def update(self, timestep: torch.Tensor | float | int) -> None:
        """
        Update the helper state using the current timestep.

        When :attr:`track_python_value` is True, the helper mirrors the timestep as a Python float.
        This allows layers to key their amax statistics by exact timestep value - useful during calibration.

        When :attr:`scales_initialized` is True, the helper selects the current scales for all layers.
        This allows the layers to read the current scales for quantization - necessary during inference.

        Parameters
        ----------
        timestep : torch.Tensor | float | int
            The timestep passed to the denoiser. Tensors may have any shape; the first
            element is used (diffusers pipelines pass a batch-homogeneous timestep).
        """
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([float(timestep)], dtype=torch.float32)
        t = timestep.detach().reshape(-1)[:1].to(torch.float32)
        if self.buffer is None or self.buffer.device != t.device:
            self.buffer = t.clone()
        else:
            self.buffer.copy_(t)

        if self.track_python_value:
            self.current_value = float(t[0])

        if self.scales_initialized:
            self._select_current_scales()

    def _select_current_scales(self) -> None:
        """Bucketize the current timestep once and gather every layer's scale for that bin."""
        # scales_initialized guarantees the scale-table tensors are set.
        idx = torch.bucketize(self.buffer, self.bin_edges)  # type: ignore[arg-type]
        self.current_scales.copy_(self.all_scales[:, idx].reshape(-1))  # type: ignore[union-attr]
        self.current_scales_reciprocal.copy_(self.all_scales_reciprocal[:, idx].reshape(-1))  # type: ignore[union-attr]

    def register_on_denoiser(self, denoiser: torch.nn.Module) -> torch.utils.hooks.RemovableHandle:
        """
        Register a pre-forward hook on the denoiser that feeds the timestep into this helper.

        The ``timestep`` argument is resolved by name from the denoiser's ``forward`` signature.

        Parameters
        ----------
        denoiser : torch.nn.Module
            The denoiser (transformer or unet) whose forward receives the timestep.

        Returns
        -------
        torch.utils.hooks.RemovableHandle
            The handle of the registered hook.
        """
        parameters = list(inspect.signature(denoiser.forward).parameters)
        if "timestep" not in parameters:
            raise ValueError(
                f"{type(denoiser).__name__}.forward has no 'timestep' parameter; "
                "cannot capture the denoising timestep for time-aware fp8 quantization."
            )
        timestep_position = parameters.index("timestep")

        def capture_timestep(module: torch.nn.Module, args: tuple, kwargs: dict[str, Any]) -> None:
            if "timestep" in kwargs:
                timestep = kwargs["timestep"]
            elif len(args) > timestep_position:
                timestep = args[timestep_position]
            else:
                timestep = None

            if timestep is not None:
                self.update(timestep)

        return denoiser.register_forward_pre_hook(capture_timestep, with_kwargs=True)

    def prepare_for_inference(self, layers: list[TimeAwareFp8Linear]) -> None:
        """
        Validate frozen layers, install the shared scale table, link layer views, and stop calibration tracking.

        Parameters
        ----------
        layers : list[TimeAwareFp8Linear]
            The finalized quantized layers, in a fixed order defining their table slots.
            Every layer must already have been initialized and must share
            the same `TimeAwareScaleHelper` and the same `calibration_timesteps` tensor.
        """
        self._validate_layers(layers)
        self._setup_scale_table(layers)
        self._link_layer_scale_views(layers)
        self.track_python_value = False

    def _validate_layers(self, layers: list[TimeAwareFp8Linear]) -> None:
        """
        Validate the layers.

        Every layer must
        - be initialized (`input_initialized` set to True post `freeze_input_scales` call,
        - share the same `TimeAwareScaleHelper` and
        - share the same `calibration_timesteps` tensor.

        Parameters
        ----------
        layers : list[TimeAwareFp8Linear]
            The finalized quantized layers to validate.
        """
        if not layers:
            raise ValueError("No TimeAwareFp8Linear layers to prepare for inference.")

        # The first layer's calibration timesteps are used as the reference.
        reference_timesteps = layers[0].calibration_timesteps

        for layer in layers:
            if not layer.input_initialized:
                raise RuntimeError("TimeAwareFp8Linear layer is not initialized (`input_initialized` is False).")

            if layer.scale_helper is not self:
                raise RuntimeError("TimeAwareFp8Linear layers do not share the same TimeAwareScaleHelper.")

            if not torch.equal(layer.calibration_timesteps, reference_timesteps):
                raise RuntimeError("TimeAwareFp8Linear layers observed different timestep grids during calibration.")

    def _setup_scale_table(self, layers: list[TimeAwareFp8Linear]) -> None:
        """
        Initialize the shared scale table and allocate the current-scale buffers.

        - The bin edges are initialized to the midpoints between the calibration timesteps.
        - The scales are initialized using `amax_to_scale` with the amax values recorded during calibration.
        - The current-scale buffers are initialized to zero.

        Important: The timesteps recorded by the layers must be sorted and the same across all layers.

        Parameters
        ----------
        layers : list[TimeAwareFp8Linear]
            Frozen quantized layers in table-slot order.
        """
        centers = layers[0].calibration_timesteps
        self.bin_edges = (centers[:-1] + centers[1:]) / 2

        all_scales = torch.stack([amax_to_scale(layer.calibration_amaxes, layer.input_max_value) for layer in layers])

        self.all_scales = all_scales
        self.all_scales_reciprocal = all_scales.reciprocal()
        self.current_scales = torch.zeros_like(all_scales[:, 0])
        self.current_scales_reciprocal = torch.zeros_like(all_scales[:, 0])
        self.scales_initialized = True

    def _link_layer_scale_views(self, layers: list[TimeAwareFp8Linear]) -> None:
        """
        Point each layer at a static view into the shared current-scale buffers.

        Parameters
        ----------
        layers : list[TimeAwareFp8Linear]
            Layers in the same order used when building the scale table.
        """
        # scales_initialized guarantees the current-scale buffers are set.
        for slot, layer in enumerate(layers):
            layer.input_current_scale = self.current_scales[slot]  # type: ignore[index]
            layer.input_current_scale_reciprocal = self.current_scales_reciprocal[slot]  # type: ignore[index]


class TimeAwareFp8Linear(torch.nn.Module):
    """
    Linear layer with static fp8 weight and activation scale selected by the current denoising timestep.

    Weights are statically quantized (fp8, per-tensor scale).
    The activation scale is a lookup table over timestep bins.

    There are two phases.

    - Calibration phase - While ``input_initialized`` is False, the input is not quantized.
      The layer computes a high-precision output and accumulates a running ``amax`` over the input activations,
      keyed by the exact timestep value via a :class:`TimeAwareScaleHelper`.
    - Inference - When ``freeze_input_scales`` is called, the calibration artifacts are created.
      On ``TimeAwareScaleHelper.prepare_for_inference``, the shared bin/scale table is initialized within the helper.
      During inference, the layer reads its scale through views over the helper's :attr:`input_current_scale` /
      :attr:`input_current_scale_reciprocal`, and uses ``torch._scaled_mm`` for fast fp8 matrix multiplication.

    Parameters
    ----------
    in_features : int
        The number of input features.
    out_features : int
        The number of output features.
    scale_helper : TimeAwareScaleHelper
        The shared helper that supplies the current timestep and per-bin activation scales.
    bias : bool, optional
        Whether to use bias.
    device : torch.device, optional
        The device to use.
    dtype : torch.dtype, optional
        The dtype to use for the weight and bias.
    weight_float8_dtype : torch.dtype, optional
        The float8 dtype to use for quantizing the weight.
    weight_float_data : torch.Tensor, optional
        The weight to use in ``dtype`` float. If None, a new, zero weight is initialized.
    bias_float_data : torch.Tensor, optional
        The bias to use in ``dtype`` float. If None, a new, zero bias is initialized.
    input_float8_dtype : torch.dtype, optional
        The float8 dtype to use for input quantization.
    """

    # round timestep keys to tolerate float noise between otherwise-identical timesteps
    _timestep_key_decimals = 6

    def __init__(
        self,
        in_features: int,
        out_features: int,
        scale_helper: TimeAwareScaleHelper,
        bias: bool = True,
        device=None,
        dtype=torch.float16,
        weight_float8_dtype=torch.float8_e4m3fn,
        weight_float_data: Optional[torch.Tensor] = None,
        bias_float_data: Optional[torch.Tensor] = None,
        input_float8_dtype=torch.float8_e4m3fn,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale_helper = scale_helper
        self.weight_float8_dtype = weight_float8_dtype
        self.input_float8_dtype = input_float8_dtype
        self.weight_initialized = False
        self.input_initialized = False
        self.weight_max_value = torch.finfo(self.weight_float8_dtype).max
        self.input_max_value = torch.finfo(self.input_float8_dtype).max
        factory_kwargs = {"dtype": dtype, "device": device}
        if weight_float_data is None:
            self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        else:
            self.weight = torch.nn.Parameter(weight_float_data, requires_grad=weight_float_data.requires_grad)
        if bias_float_data is None:
            if bias:
                self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
            else:
                self.register_parameter("bias", None)
        else:
            self.bias = torch.nn.Parameter(bias_float_data, requires_grad=bias_float_data.requires_grad)
        self.register_buffer("weight_scale", torch.tensor(1.0, device=device, dtype=torch.float32))
        self.register_buffer("weight_scale_reciprocal", torch.tensor(1.0, device=device, dtype=torch.float32))
        self.register_buffer("weight_float8_data", None)
        # Dtype-independent calibration state persisted in artifacts and finalized by freeze_input_scales.
        self.register_buffer("calibration_timesteps", None)
        self.register_buffer("calibration_amaxes", None)
        self.input_current_scale: torch.Tensor | None = None
        self.input_current_scale_reciprocal: torch.Tensor | None = None
        self.input_running_amax_by_timestep: dict[float, torch.Tensor] = {}

    def quantize_weight(self) -> None:
        """Quantize the weight of the linear layer (static, no calibration required)."""
        if self.weight_initialized:
            return
        amax = torch.max(torch.abs(self.weight.data)).float()

        self.weight_scale = amax_to_scale(amax, self.weight_max_value)
        self.weight_float8_data = scale_and_clamp(self.weight.data, self.weight_scale, self.weight_max_value).to(
            self.weight_float8_dtype
        )
        self.weight_scale_reciprocal = self.weight_scale.reciprocal()

        self.weight.data = torch.zeros(1, dtype=self.weight.dtype, device=self.weight.device, requires_grad=False)
        self.weight_initialized = True

    def _dequantized_weight(self, dtype: torch.dtype) -> torch.Tensor:
        """Reconstruct the (lossy) high-precision weight from its fp8 representation and cast to ``dtype``."""
        return (self.weight_float8_data.to(torch.float32) * self.weight_scale_reciprocal).to(dtype)

    def _calibration_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Record the input amax under the current timestep and return a high-precision output."""
        amax = torch.max(torch.abs(x)).to(torch.float32)
        key = round(self.scale_helper.current_value, self._timestep_key_decimals)
        previous = self.input_running_amax_by_timestep.get(key)
        self.input_running_amax_by_timestep[key] = (
            amax.detach() if previous is None else torch.maximum(previous, amax.detach())
        )
        weight = self._dequantized_weight(x.dtype)
        return f.linear(x, weight, self.bias)

    def freeze_input_scales(self) -> None:
        """
        Finalize dtype-independent calibration artifacts for this layer.

        Prefers the in-memory calibration dict when present (fresh smash).

        Otherwise requires the persisted `calibration_timesteps` / `calibration_amaxes` buffers (artifact load).
        """
        if self.input_initialized:
            return

        if self.input_running_amax_by_timestep:
            device = self.weight_float8_data.device
            timesteps = sorted(self.input_running_amax_by_timestep)
            self.calibration_timesteps = torch.tensor(timesteps, device=device, dtype=torch.float32)
            self.calibration_amaxes = torch.stack([self.input_running_amax_by_timestep[t] for t in timesteps]).to(
                device=device, dtype=torch.float32
            )
            self.input_running_amax_by_timestep.clear()
        elif self.calibration_timesteps is None or self.calibration_amaxes is None:
            raise RuntimeError("TimeAwareFp8Linear saw no calibration forwards. Cannot build per-timestep scales.")

        self.input_initialized = True

    def _quantized_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fast fp8 path using the pre-gathered scale of the current timestep bin."""
        if self.input_current_scale is None or self.input_current_scale_reciprocal is None:
            raise RuntimeError(
                "The time-aware scale helper must call `prepare_for_inference` post-calibration and prior-inference."
            )

        x = scale_and_clamp(x, self.input_current_scale, self.input_max_value).to(self.input_float8_dtype)

        prev_dims = x.shape[:-1]
        x = x.reshape(-1, self.in_features)

        # torch._scaled_mm requires column-major weight matrix ((1, K) stride),
        # which .T produces using a view (i.e., no memory changes)
        out = torch._scaled_mm(
            x,
            self.weight_float8_data.T,
            scale_a=self.input_current_scale_reciprocal,
            scale_b=self.weight_scale_reciprocal,
            bias=self.bias,
            out_dtype=self.weight.dtype,
            use_fast_accum=True,
        )
        out = out.reshape(*prev_dims, self.out_features)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the forward pass, dispatching on the calibration flag.

        Parameters
        ----------
        x : torch.Tensor
            The input to the linear layer.

        Returns
        -------
        torch.Tensor
            The output of the linear layer.
        """
        if not self.input_initialized:
            return self._calibration_forward(x)
        return self._quantized_forward(x)

    @classmethod
    def from_linear(
        cls,
        linear: torch.nn.Linear,
        scale_helper: TimeAwareScaleHelper,
        weight_float8_dtype=torch.float8_e4m3fn,
        input_float8_dtype=torch.float8_e4m3fn,
    ) -> "TimeAwareFp8Linear":
        """
        Create a new TimeAwareFp8Linear instance from a nn.Linear instance.

        Parameters
        ----------
        linear : torch.nn.Linear
            The linear layer to convert to TimeAwareFp8Linear.
        scale_helper : TimeAwareScaleHelper
            The shared helper that supplies the current timestep and per-bin activation scales.
        weight_float8_dtype : torch.dtype
            The float8 dtype to use for weight quantization.
        input_float8_dtype : torch.dtype
            The float8 dtype to use for input quantization.

        Returns
        -------
        TimeAwareFp8Linear
            The new TimeAwareFp8Linear instance.
        """
        f8_lin = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            scale_helper=scale_helper,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            weight_float8_dtype=weight_float8_dtype,
            weight_float_data=linear.weight.data,
            bias_float_data=(linear.bias.data if linear.bias is not None else None),
            input_float8_dtype=input_float8_dtype,
        )
        f8_lin.quantize_weight()
        return f8_lin


def quantize_linear_layer_time_aware_fp8(
    parent: torch.nn.Module,
    child_name: str,
    weight_float8_dtype: torch.dtype,
    input_float8_dtype: torch.dtype,
    scale_helper: TimeAwareScaleHelper,
) -> None:
    """
    Quantize a linear layer with time-aware fp8 quantization.

    Parameters
    ----------
    parent : torch.nn.Module
        The parent module of the linear layer.
    child_name : str
        The name of the linear layer.
    weight_float8_dtype : torch.dtype
        The float8 dtype to use for weight quantization.
    input_float8_dtype : torch.dtype
        The float8 dtype to use for input quantization.
    scale_helper : TimeAwareScaleHelper
        The shared scale helper the layer reads to select its per-bin scale.

    Returns
    -------
    None
        This function modifies the model in-place and does not return anything.
    """
    child = getattr(parent, child_name)
    quantized_linear = TimeAwareFp8Linear.from_linear(
        child,
        scale_helper=scale_helper,
        weight_float8_dtype=weight_float8_dtype,
        input_float8_dtype=input_float8_dtype,
    )
    setattr(parent, child_name, quantized_linear)
    del child

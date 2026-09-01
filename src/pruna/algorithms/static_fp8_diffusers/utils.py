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

from typing import Optional

import torch
import torch.nn.functional as f

from pruna.algorithms.global_utils.quantization.symmetric_scale import amax_to_scale, scale_and_clamp


class StaticFp8Linear(torch.nn.Module):
    """
    Linear layer with static fp8 weight and activation quantization.

    Based on the ``F8Linear`` class from https://github.com/aredden/flux-fp8-api, but with a
    two-phase design coordinated by a boolean flag and tailored to the pruna squash framework.

    - Calibration phase - While ``input_initialized`` is False, the input is not quantized.
      The layer computes a high-precision output and accumulates a running ``amax`` over the input activations.
    - Inference - When ``input_initialized`` is True (i.e., after a ``freeze_input_scale`` call), the input scale
      is frozen and the layer uses the fast ``torch._scaled_mm`` fp8 matrix multiplication.

    Parameters
    ----------
    in_features : int
        The number of input features.
    out_features : int
        The number of output features.
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
        The float8 dtype to use for quantizing the input.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
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
        self.register_buffer("input_running_amax", torch.tensor(0.0, device=device, dtype=torch.float32))
        self.register_buffer("weight_scale", torch.tensor(1.0, device=device, dtype=torch.float32))
        self.register_buffer("weight_scale_reciprocal", torch.tensor(1.0, device=device, dtype=torch.float32))
        self.register_buffer("weight_float8_data", None)
        self.register_buffer("input_scale", torch.tensor(1.0, device=device, dtype=torch.float32))
        self.register_buffer("input_scale_reciprocal", torch.tensor(1.0, device=device, dtype=torch.float32))

    def quantize_weight(self) -> None:
        """Quantize the weight of the linear layer (static, no calibration required)."""
        if self.weight_initialized:
            return
        amax = torch.max(torch.abs(self.weight.data)).float()

        self.weight_scale = amax_to_scale(amax, self.weight_max_value)
        self.weight_scale_reciprocal = self.weight_scale.reciprocal()
        self.weight_float8_data = (
            scale_and_clamp(self.weight.data, self.weight_scale, self.weight_max_value).to(
                self.weight_float8_dtype
            )
        )

        self.weight.data = torch.zeros(1, dtype=self.weight.dtype, device=self.weight.device, requires_grad=False)
        self.weight_initialized = True

    def _dequantized_weight(self, dtype: torch.dtype) -> torch.Tensor:
        """Reconstruct the (lossy) high-precision weight from its fp8 representation and cast to ``dtype``."""
        return (self.weight_float8_data.to(torch.float32) * self.weight_scale_reciprocal).to(dtype)

    def freeze_input_scale(self) -> None:
        """Freeze the input scale from the accumulated statistics and switch to quantized mode."""
        if self.input_initialized:
            return
        self.input_scale = amax_to_scale(self.input_running_amax, self.input_max_value)
        self.input_scale_reciprocal = self.input_scale.reciprocal()
        self.input_initialized = True

    def _calibration_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gather input statistics without quantizing the input and return a high-precision output."""
        amax = torch.max(torch.abs(x)).to(torch.float32)
        self.input_running_amax = torch.maximum(self.input_running_amax, amax)
        weight = self._dequantized_weight(x.dtype)
        return f.linear(x, weight, self.bias)

    def _quantized_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fast fp8 forward pass using the frozen input scale."""
        x = scale_and_clamp(x, self.input_scale, self.input_max_value).to(self.input_float8_dtype)

        prev_dims = x.shape[:-1]
        x = x.reshape(-1, self.in_features)

        # torch._scaled_mm requires column-major weight matrix ((1, K) stride),
        # which .T produces using a view (i.e., no memory changes)
        out = torch._scaled_mm(
            x,
            self.weight_float8_data.T,
            scale_a=self.input_scale_reciprocal,
            scale_b=self.weight_scale_reciprocal,
            bias=self.bias,
            out_dtype=self.weight.dtype,
            use_fast_accum=True,
        )
        out = out.reshape(*prev_dims, self.out_features)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the forward pass.

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
        *,
        weight_float8_dtype=torch.float8_e4m3fn,
        input_float8_dtype=torch.float8_e4m3fn,
    ) -> "StaticFp8Linear":
        """
        Create a new StaticFp8Linear instance from a nn.Linear instance.

        Parameters
        ----------
        linear : torch.nn.Linear
            The linear layer to convert to StaticFp8Linear.
        weight_float8_dtype : torch.dtype
            The float8 dtype to use for weight quantization.
        input_float8_dtype : torch.dtype
            The float8 dtype to use for input quantization.

        Returns
        -------
        StaticFp8Linear
            The new StaticFp8Linear instance.
        """
        f8_lin = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
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


def quantize_linear_layer_static_fp8(
    parent: torch.nn.Module,
    child_name: str,
    weight_float8_dtype: torch.dtype,
    input_float8_dtype: torch.dtype,
) -> None:
    """
    Quantize a linear layer with static fp8 quantization.

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

    Returns
    -------
    None
        This function modifies the model in-place and does not return anything.
    """
    child = getattr(parent, child_name)
    quantized_linear = StaticFp8Linear.from_linear(
        child,
        weight_float8_dtype=weight_float8_dtype,
        input_float8_dtype=input_float8_dtype,
    )
    setattr(parent, child_name, quantized_linear)
    del child

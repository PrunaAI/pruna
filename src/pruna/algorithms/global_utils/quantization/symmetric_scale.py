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

import torch


def amax_to_scale(amax: torch.Tensor, max_val: float) -> torch.Tensor:
    """
    Convert an absolute-max value to a per-tensor quantization scale.

    Parameters
    ----------
    amax : torch.Tensor
        The absolute max value of the tensor to quantize.
    max_val : float
        The maximum representable magnitude of the target float dtype.

    Returns
    -------
    torch.Tensor
        The scale to use for quantization.
    """
    return (max_val / torch.clamp(amax, min=1e-12))


def scale_and_clamp(x: torch.Tensor, scale: torch.Tensor, max_val: float) -> torch.Tensor:
    """
    Map the input to the range [-max_val, max_val] using the given scale.

    Parameters
    ----------
    x : torch.Tensor
        The input to quantize.
    scale : torch.Tensor
        The scale to use for quantization.
    max_val : float
        The maximum representable magnitude of the target float dtype.

    Returns
    -------
    torch.Tensor
        The scaled input, clamped to ``[-max_val, max_val]``.
    """
    return (x * scale).clamp(-max_val, max_val)

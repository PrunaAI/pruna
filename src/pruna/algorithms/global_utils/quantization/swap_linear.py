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

from typing import Any, Callable, Dict

import torch


def swap_linear(
    module: torch.nn.Module,
    quantize_linear_layer_fn: Callable,
    path: str = "",
    kwargs: Dict[str, Any] = {},
) -> None:
    """
    Swap one nn.Linear module in the given model with its quantized equivalent.

    This function follows the provided path to the nn.Linear layer and swaps it
    in place by calling ``quantize_linear_layer_fn``. That callback is responsible
    for replacing the module and may delete the original linear to free its weights.

    Parameters
    ----------
    module : nn.Module
        The PyTorch module containing the linear layer to swap.
    quantize_linear_layer_fn : Callable
        The function to use to quantize the linear layer.
        It must accept the following arguments in this order:
        - parent: torch.nn.Module
        - child_name: str
        - **kwargs: Any
    path : str
        The path in the model hierarchy to find the linear layer to swap.
    kwargs : Dict[str, Any]
        The keyword arguments to pass to the quantize_linear_layer_fn.

    Returns
    -------
    None
        This function applies the quantization callback to the linear layer in-place.
        It does not return anything.
    """
    if not path:
        raise ValueError(f"Unexpected empty path: {path}")

    attributes_to_linear = path.split(".")
    attributes_to_parent, child_name = attributes_to_linear[:-1], attributes_to_linear[-1]

    # get the submodule containing the linear layer
    parent = module
    for attr in attributes_to_parent:
        parent = getattr(parent, attr)

    # apply the quantization callback to the linear layer
    child = getattr(parent, child_name)
    if isinstance(child, torch.nn.Linear):
        quantize_linear_layer_fn(parent, child_name, **kwargs)

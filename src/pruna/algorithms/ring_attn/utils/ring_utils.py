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

import contextlib
from types import ModuleType
from typing import Union

import torch
import torch.distributed as dist
from torch.nn.functional import scaled_dot_product_attention
from torch.overrides import TorchFunctionMode

from pruna.config.smash_config import SmashConfig, SmashConfigPrefixWrapper

ring_attention: ModuleType | None = None

with contextlib.suppress(ImportError):
    # see "import_algorithm_packages" for further explanation
    import torch.distributed.tensor.experimental._attention as ring_attention


class LocalFunc(torch.autograd.Function):
    """
    Local dummy function to mark the ring attention forwarding as an autograd function.

    Parameters
    ----------
    *args : Any
        The arguments to the autograd class construction.
    **kwargs : Any
        The keyword arguments to the autograd class construction.
    """

    @staticmethod
    def forward(cls, *args, **kwargs):
        """
        Forward pass for the ring attention implementation.

        Parameters
        ----------
        *args : Any
            The arguments to the forward pass.
        **kwargs : Any
            The keyword arguments to the forward pass.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        # when distributing manually, it seems this is overwritten but we have to ensure it is False
        ring_attention._cp_options.enable_load_balance = False
        # FUTURE: investigate if we can use the efficient implementation here and if it makes sense
        return ring_attention._scaled_dot_product_ring_flash_attention(*args, **kwargs)[:2]

    @staticmethod
    def backward(cls, *args, **kwargs):
        """
        Backward pass for ring attention implementation of flash attention.

        Parameters
        ----------
        *args : Any
            The arguments to the backward pass.
        **kwargs : Any
            The keyword arguments to the backward pass.

        Returns
        -------
        torch.Tensor
            The gradient of the output tensor.
        """
        return ring_attention._scaled_dot_product_ring_flash_attention_backward(*args, **kwargs)


class RingDistributedContext(TorchFunctionMode):
    """
    Intercept *every* call to F.scaled_dot_product_attention and routes it through the ring implementation.

    Parameters
    ----------
    device_mesh : dist.DeviceMesh
        The device mesh to use for the distributed attention.
    smash_config : Union[SmashConfig, SmashConfigPrefixWrapper]
        The SmashConfig to use.
    """

    def __init__(self, device_mesh: dist.DeviceMesh, smash_config: Union[SmashConfig, SmashConfigPrefixWrapper]):
        super().__init__()
        self.pg = device_mesh
        self.smash_config = smash_config

    def __torch_function__(self, func, types, args=(), kwargs=None):
        """
        Intercept the scaled_dot_product_attention function and route it through the ring implementation.

        Parameters
        ----------
        func : Callable
            The function to intercept.
        types : Tuple[type]
            The types of the arguments.
        args : Tuple
            The arguments to the function.
        kwargs : Dict
            The keyword arguments to the function.

        Returns
        -------
        Any
            The result of the function.
        """
        kwargs = {} if kwargs is None else kwargs

        if func is torch.Tensor.unflatten:
            return torch.unflatten(*args, **kwargs)

        if func is scaled_dot_product_attention:
            query, key, value, *extra = args
            attn_mask = kwargs.pop("attn_mask", None)
            if attn_mask is not None:
                raise ValueError("Ring attention path does not support `attn_mask`; use causal masking instead.")
            dropout_p = kwargs.get("dropout_p", 0.0)
            is_causal = kwargs.get("is_causal", False)
            scale = kwargs.get("scale", None)

            out, _ = LocalFunc.apply(self.pg, query, key, value, dropout_p, is_causal, scale)
            return out.to(query.dtype)

        # fall back to default behavior
        return func(*args, **kwargs)

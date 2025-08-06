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

import functools
from typing import Any, Dict, Optional, Tuple

import torch
from aenum import extend_enum
from diffusers import DiffusionPipeline
from diffusers import __version__ as diffusers_version
from kernels import get_kernel
from packaging.version import Version
from torch.overrides import TorchFunctionMode

from pruna.algorithms.kernels import PrunaKernel
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.logging.logger import pruna_logger


class FlashAttn3Kernel(PrunaKernel):
    """
    Replace torch.nn.functional.scaled_dot_product_attention with flash_attn3.

    Flash Attention 3 is a fast and memory-efficient attention mechanism. It uses a combination of tiling, streaming
    and fusing to speed up attention computations.
    """

    algorithm_name: str = "flash_attn3"
    references: dict[str, str] = {
        "GitHub": "https://github.com/Dao-AILab/flash-attention",
        "Kernel Hub": "https://huggingface.co/kernels-community/models",
    }
    tokenizer_required: bool = False
    processor_required: bool = False
    runs_on: list[str] = ["cuda", "accelerate"]
    dataset_required: bool = False
    compatible_algorithms: dict[str, list[str]] = dict(
        compiler=["torch_compile"], cacher=["fora"]
    )

    def model_check_fn(self, model: Any) -> bool:
        """
        Check if the model has an attention mechanism that can be replaced with flash_attn3.

        Parameters
        ----------
        model : Any
            The model to check.

        Returns
        -------
        bool
            True if the model is a valid model for the algorithm, False otherwise.
        """
        if Version(diffusers_version) >= Version("0.35.0.dev0"):
            if not isinstance(model, DiffusionPipeline) or not hasattr(
                model, "components"
            ):
                return False

            return any(
                hasattr(component, "set_attention_backend")
                and component.dtype in [torch.bfloat16, torch.float16]
                for component in model.components.values()
            )
        else:
            return isinstance(model, DiffusionPipeline) and hasattr(
                model, "transformer"
            )

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Wrap the model to use flash_attn3 where possible.

        Parameters
        ----------
        model : Any
            The model to wrap.
        smash_config : SmashConfigPrefixWrapper
            The configuration for the application of the algorithm.

        Returns
        -------
        Any
            The wrapped model.
        """
        imported_packages = self.import_algorithm_packages()

        # in the new version of diffusers, we can use the modular attention backend to inject flash_attn3
        if Version(diffusers_version) >= Version("0.35.0.dev0"):
            # register our "custom" attention function as a backend
            register_custom_backend(imported_packages)

            # replace in all compatible components
            for component in model.components.values():
                if hasattr(component, "set_attention_backend") and component.dtype in [
                    torch.bfloat16,
                    torch.float16,
                ]:
                    component.set_attention_backend("flash_attn3_pruna")

        else:
            # register the flash attention 3 operation with torch ops to make it compatible with fullgraph compilation
            register_pruna_flash_attn_op(imported_packages["flash_attention_3"])

            # wrap the model generate function to replace attention computations with flash_attn3 where possible
            wrap_pipeline_call(model, imported_packages)
        return model

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Import the algorithm packages.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        flash_attention_3 = get_kernel("kernels-community/flash-attn3")
        packages = {"flash_attention_3": flash_attention_3}

        if Version(diffusers_version) >= Version("0.35.0.dev0"):
            from diffusers.models.attention_dispatch import (
                AttentionBackendName,
                _AttentionBackendRegistry,
                _check_device,
                _check_qkv_dtype_bf16_or_fp16,
                _check_shape,
                _native_attention,
            )

            packages.update(
                {
                    "_AttentionBackendRegistry": _AttentionBackendRegistry,
                    "_check_device": _check_device,
                    "_check_qkv_dtype_bf16_or_fp16": _check_qkv_dtype_bf16_or_fp16,
                    "_check_shape": _check_shape,
                    "_native_attention": _native_attention,
                    "AttentionBackendName": AttentionBackendName,
                    "flash_attention_3": flash_attention_3,
                }
            )
        return packages


def register_custom_backend(imported_packages: Dict[str, Any]) -> None:
    """
    Register the attention backend for flash_attn3 by mimicing the native backend.

    Parameters
    ----------
    imported_packages : Dict[str, Any]
        The imported packages.
    """
    attention_backend_registry = imported_packages["_AttentionBackendRegistry"]
    _check_device = imported_packages["_check_device"]
    _check_shape = imported_packages["_check_shape"]
    _check_qkv_dtype_bf16_or_fp16 = imported_packages["_check_qkv_dtype_bf16_or_fp16"]
    _native_attention = imported_packages["_native_attention"]
    flash_attention_3 = imported_packages["flash_attention_3"]
    attention_backend_name = imported_packages["AttentionBackendName"]

    if attention_backend_registry.get_active_backend()[0].name != "NATIVE":
        pruna_logger.warning(
            "The current active attention backend is not native. This might lead to unexpected behavior."
        )

    if "FLASH_ATTN3_PRUNA" not in attention_backend_name.__members__:

        @attention_backend_registry.register(
            "flash_attn3_pruna",
            constraints=[_check_device, _check_shape],
        )
        def _flash_attention_3(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            scale: Optional[float] = None,
            is_causal: bool = False,
            # unsupported by flash_attn3 but we catch them to reroute to native attention if necessary
            attn_mask: Optional[torch.Tensor] = None,
            dropout_p: float = 0.0,
            enable_gqa: bool = False,
        ) -> torch.Tensor:
            # flash attention 3 only supports bfloat16 and fp16
            dtype_pass = True
            try:
                _check_qkv_dtype_bf16_or_fp16(query=query, key=key, value=value)
            except ValueError:
                dtype_pass = False

            # fa3 only supports attention with num_query_heads % num_kv_heads == 0
            num_heads_pass = all(query.shape[1] % t.shape[1] == 0 for t in (key, value))

            # test head dimension
            head_dim_pass = all(t.shape[3] <= 256 for t in (query, key, value))

            # if any constraints are not met or unsupported input arguments are being used, reroute to native attention
            if (
                attn_mask is not None
                or dropout_p != 0.0
                or not dtype_pass
                or not num_heads_pass
                or not head_dim_pass
            ):
                pruna_logger.debug(
                    "Rerouting to native attention... Check the following criteria: "
                    f"attn_mask_pass: {attn_mask is not None}, dropout_p_pass: {dropout_p != 0.0}, dtype_pass: {dtype_pass},"
                    f"num_heads_pass: {num_heads_pass}, head_dim_pass: {head_dim_pass}"
                )
                return _native_attention(
                    query=query,
                    key=key,
                    value=value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                    # GQA is anyway supported by flash attention 3
                    enable_gqa=enable_gqa,
                )
            else:
                pruna_logger.debug("Using FA3...")
                out, _, *_ = flash_attention_3.flash_attn_func(
                    q=query,
                    k=key,
                    v=value,
                    softmax_scale=scale,
                    causal=is_causal,
                    deterministic=False,
                )
                return out

        extend_enum(attention_backend_name, "FLASH_ATTN3_PRUNA", "flash_attn3_pruna")


class FlashAttention3Context(TorchFunctionMode):
    """
    Context manager to intercept calls to scaled_dot_product_attention and replace them with flash_attn3.

    Parameters
    ----------
    kernel : Any
        The kernel to use for the flash attention 3.
    """

    def __init__(self, kernel: Any):
        super().__init__()
        self.kernel = kernel

    def __torch_function__(self, func, types, args=(), kwargs=None):  # noqa: D105
        kwargs = {} if kwargs is None else kwargs
        if func == torch.nn.functional.scaled_dot_product_attention:
            # rename keyword arguments in case of naming mismatch
            if "q" in kwargs:
                kwargs["query"] = kwargs.pop("q")
            if "k" in kwargs:
                kwargs["key"] = kwargs.pop("k")
            if "v" in kwargs:
                kwargs["value"] = kwargs.pop("v")

            # parse arguments from kwargs or args
            query = kwargs["query"] if "query" in kwargs else args[0]
            key = kwargs["key"] if "key" in kwargs else args[1]
            value = kwargs["value"] if "value" in kwargs else args[2]

            # check that unsupported arguments are not being used
            attn_mask_pass = kwargs.get("attn_mask", None) is None
            dropout_p_pass = kwargs.get("dropout_p", 0.0) == 0.0

            # check that the number of query heads is divisible by the number of key/value heads (GQA constraint)
            shapes_pass = all(query.shape[1] % t.shape[1] == 0 for t in (key, value))
            # check that the dtype is bfloat16 or fp16
            dtype_pass = query.dtype in [torch.bfloat16, torch.float16]
            head_dim_pass = all(t.shape[3] <= 256 for t in (key, value, query))

            if (
                attn_mask_pass
                and dropout_p_pass
                and shapes_pass
                and dtype_pass
                and head_dim_pass
            ):
                kwargs.pop("attn_mask", None)
                kwargs.pop("dropout_p", None)
                kwargs.pop("enable_gqa", None)
                kwargs["softmax_scale"] = kwargs.pop("scale", None)
                pruna_logger.debug("Using FA3...")
                return _flash_attention3(*args, **kwargs, kernel=self.kernel)
            else:
                pruna_logger.debug(
                    "Rerouting to native attention... Check the following criteria: "
                    f"attn_mask_pass: {attn_mask_pass}, dropout_p_pass: {dropout_p_pass}, shapes_pass: {shapes_pass},"
                    f"dtype_pass: {dtype_pass}, head_dim_pass: {head_dim_pass}"
                )
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)


def _flash_attention3(
    query, key, value, *, is_causal=False, softmax_scale=None, kernel=None
):
    # convert (B, H, S, D) → (B, S, H, D)
    q, k, v = [x.transpose(1, 2).contiguous() for x in (query, key, value)]
    out, _ = torch.ops.flash_attn_pruna._flash_attn_forward(
        q, k, v, causal=is_causal, softmax_scale=softmax_scale
    )
    # back to (B, H, S, D) for the rest of the pipeline
    return out.transpose(1, 2)


def wrap_pipeline_call(model: Any, imported_packages: Dict[str, Any]) -> None:
    """
    Wrap the model generate function to replace attention computations with flash_attn3 where possible.

    Parameters
    ----------
    model : Any
        The model to wrap.
    imported_packages : Dict[str, Any]
        The imported packages.
    """
    original_forward = model.__call__

    @functools.wraps(original_forward)
    def new_forward(*args, original_forward=original_forward, **kwargs):
        with FlashAttention3Context(kernel=imported_packages["flash_attention_3"]):
            return original_forward(*args, **kwargs)

    model.__call__ = new_forward  # type: ignore


def register_pruna_flash_attn_op(kernel_mod: Any) -> None:
    """
    Register the flash attention 3 operation with torch ops to make it compatible with fullgraph compilation.

    Parameters
    ----------
    kernel_mod : Any
        The flash attention 3 kernel module.
    """
    flash_attn_cuda = kernel_mod.flash_attn_func

    @torch.library.custom_op(
        "flash_attn_pruna::_flash_attn_forward", mutates_args=(), device_types="cuda"
    )
    def _flash_attn_forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: float | None = None,
        causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out, lse = flash_attn_cuda(
            q, k, v, softmax_scale=softmax_scale or None, causal=causal
        )
        return out, lse.permute(0, 2, 1)  # (B,H,S) → (B,S,H)

    @torch.library.register_fake("flash_attn_pruna::_flash_attn_forward")
    def _flash_attn_forward_fake(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: float | None = None,
        causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, s, h, d = q.shape
        return torch.empty_like(q), q.new_empty((b, s, h))

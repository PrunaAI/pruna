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
import inspect
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple

import torch
from aenum import extend_enum
from diffusers import DiffusionPipeline
from kernels import get_kernel
from torch.overrides import TorchFunctionMode

from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag as tags
from pruna.config.hyperparameters import Boolean
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.config.target_modules import TargetModules, filter_targeted_modules, map_targeted_nn_roots
from pruna.engine.save import SAVE_FUNCTIONS
from pruna.logging.logger import pruna_logger


class FlashAttn3(PrunaAlgorithmBase):
    """
    Replace torch.nn.functional.scaled_dot_product_attention with flash_attn3.

    Flash Attention 3 is a fast and memory-efficient attention mechanism. It uses a combination of tiling, streaming
    and fusing to speed up attention computations.
    """

    algorithm_name: str = "flash_attn3"
    group_tags: list[tags] = [tags.KERNEL]
    save_fn = SAVE_FUNCTIONS.reapply
    references: dict[str, str] = {
        "GitHub": "https://github.com/Dao-AILab/flash-attention",
        "Kernel Hub": "https://huggingface.co/kernels-community/models",
    }
    tokenizer_required: bool = False
    processor_required: bool = False
    runs_on: list[str] = ["cuda", "accelerate"]
    dataset_required: bool = False
    compatible_before: Iterable[str] = ["torchao", "padding_pruning"]
    compatible_after: Iterable[str] = ["fora", "torch_compile"]

    def model_check_fn(self, model: Any) -> bool:
        """
        Check if the model fulfills necessary conditions to apply the fa3 algorithm.

        Parameters
        ----------
        model : Any
            The model to check.

        Returns
        -------
        bool
            True if the model is a valid model for the algorithm, False otherwise.
        """
        if isinstance(model, DiffusionPipeline):
            return any(
                isinstance(component, torch.nn.Module) and _is_fp16_or_bf16(component)
                for _, component in inspect.getmembers(model)
            )
        # Custom models: accept if the model is an nn.Module itself or contains at least one nn.Module attribute.
        # While this does not ensure that the model has at least one attention module,
        # it already filters out invalid models.
        if isinstance(model, torch.nn.Module):
            return True
        return any(isinstance(attr, torch.nn.Module) for _, attr in inspect.getmembers(model))

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Wrap the model to use flash_attn3 where possible.

        The algorithm follows the following logic:

        - If the model supports the diffusers backend API (``set_attention_backend``, diffusers >= 0.35),
          the algorithm sets the attention backend on targeted modules.
        - Otherwise (older diffusers pipelines, plain nn.Modules, or custom wrappers), the algorithm
          wraps the forward method of targeted modules with FlashAttention3Context.
        - Always register the standard (non-FP8) op.
        - If no fp8 is requested, apply the standard op to all targeted modules.
        - If fp8 is requested, apply the standard op to all compatible modules
          and apply fp8 to the target modules.

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
        kernel = self.import_algorithm_packages()["flash_attention_3"]
        use_fp8 = smash_config["fp8"]
        target_modules = smash_config["target_modules"]

        # Always register the standard (non-FP8) kernel with torch ops
        register_pruna_flash_attn_op(kernel, use_fp8=False)

        # Filter target modules to only include fp16/bf16 modules
        target_modules = filter_targeted_modules(_is_fp16_or_bf16, model, target_modules)

        # Determine apply strategy: backend API (diffusers >= 0.35) or forward wrapping (everything else)
        use_backend = self._has_backend_api(model)

        if use_backend:
            backend_packages = self._import_backend_packages()
            backend_name = register_custom_backend(backend_packages, use_fp8=False)
            apply_fn = functools.partial(_apply_via_backend, backend=backend_name)
        else:
            apply_fn = functools.partial(_apply_via_forward_wrap, kernel=kernel, use_fp8=False)

        # Apply the apply function to the model
        # If fp8 is used, the apply function for fp8 needs to be built as well
        if use_fp8:
            # FA3 fp16 on ALL compatible modules
            all_fp16 = filter_targeted_modules(_is_fp16_or_bf16, model, {"include": ["*"], "exclude": []})
            model = map_targeted_nn_roots(apply_fn, model, all_fp16)
            # FA3 fp8 overwrites targeted modules
            register_pruna_flash_attn_op(kernel, use_fp8=True)
            if use_backend:
                backend_name_fp8 = register_custom_backend(backend_packages, use_fp8=True)
                apply_fn_fp8 = functools.partial(_apply_via_backend, backend=backend_name_fp8)
            else:
                apply_fn_fp8 = functools.partial(_apply_via_forward_wrap, kernel=kernel, use_fp8=True)
            model = map_targeted_nn_roots(apply_fn_fp8, model, target_modules)
        else:
            # FA3 fp16 only on targeted modules
            model = map_targeted_nn_roots(apply_fn, model, target_modules)

        return model

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Import the algorithm packages.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        return {"flash_attention_3": get_kernel("kernels-community/flash-attn3")}

    def _has_backend_api(self, model: Any) -> bool:
        """Check if the model supports the diffusers attention backend API (>= 0.35)."""
        if not isinstance(model, DiffusionPipeline) or not hasattr(model, "components"):
            return False
        return any(hasattr(c, "set_attention_backend") for c in model.components.values())

    def _import_backend_packages(self) -> Dict[str, Any]:
        """Import diffusers backend packages. Only called when the backend API is available."""
        from diffusers.models.attention_dispatch import (
            AttentionBackendName,
            _AttentionBackendRegistry,
            _check_device,
            _check_qkv_dtype_bf16_or_fp16,
            _check_shape,
            _native_attention,
        )

        return {
            "_AttentionBackendRegistry": _AttentionBackendRegistry,
            "_check_device": _check_device,
            "_check_qkv_dtype_bf16_or_fp16": _check_qkv_dtype_bf16_or_fp16,
            "_check_shape": _check_shape,
            "_native_attention": _native_attention,
            "AttentionBackendName": AttentionBackendName,
        }

    def get_hyperparameters(self) -> list:
        """
        Get the list of configurable hyperparameters for this algorithm.

        Returns
        -------
        list
            A list of hyperparameter objects (e.g., Boolean, TargetModules) used by the
            configuration system.
        """
        return [
            # We do not set specific default target modules as FA3 is lossless if not used with FP8 quantization
            # and therefore can be applied to any attn module without any performance degradation.
            TargetModules(name="target_modules", default_value={"include": ["*"], "exclude": []}),
            Boolean("fp8", default=False, meta=dict(desc="Apply FlashAttention3 with FP8 quantization.")),
        ]


def register_custom_backend(imported_packages: Dict[str, Any], use_fp8: bool = False) -> str:
    """
    Register the attention backend for flash_attn3 by mimicing the native backend.

    Used for standard diffusers pipeline models.

    Parameters
    ----------
    imported_packages : Dict[str, Any]
        The imported packages.
    use_fp8 : bool
        Whether to use FP8 quantization in this backend instance.

    Returns
    -------
    str
        The registered backend name.
    """
    attention_backend_registry = imported_packages["_AttentionBackendRegistry"]
    _check_device = imported_packages["_check_device"]
    _check_shape = imported_packages["_check_shape"]
    _check_qkv_dtype_bf16_or_fp16 = imported_packages["_check_qkv_dtype_bf16_or_fp16"]
    _native_attention = imported_packages["_native_attention"]
    attention_backend_name = imported_packages["AttentionBackendName"]

    if attention_backend_registry.get_active_backend()[0].name != "NATIVE":
        pruna_logger.warning(
            "The current active attention backend is not native. This might lead to unexpected behavior."
        )

    backend_name = "flash_attn3_pruna_fp8" if use_fp8 else "flash_attn3_pruna"
    enum_key = backend_name.upper()

    if enum_key not in attention_backend_name.__members__:

        # Pick the right custom op based on use_fp8
        _ops = torch.ops.flash_attn_pruna  # ty: ignore[invalid-argument-type]
        _op_fn = _ops._flash_attn_forward_fp8 if use_fp8 else _ops._flash_attn_forward

        @attention_backend_registry.register(
            backend_name,
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
            if attn_mask is not None or dropout_p != 0.0 or not dtype_pass or not num_heads_pass or not head_dim_pass:
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
                out = _op_fn(
                    q=query, k=key, v=value, softmax_scale=scale, causal=is_causal  # ty: ignore[invalid-argument-type]
                )
                return out

        extend_enum(attention_backend_name, enum_key, backend_name)

    return backend_name


class FlashAttention3Context(TorchFunctionMode):
    """
    Context manager to intercept calls to scaled_dot_product_attention and replace them with flash_attn3.

    Used for custom (non-diffusers-pipeline) models.

    Parameters
    ----------
    kernel : Any
        The kernel to use for the flash attention 3.
    use_fp8 : bool
        Whether to quantize Q, K, V to FP8 before the attention computation.
    """

    def __init__(self, kernel: Any, use_fp8: bool = False):
        super().__init__()
        self.kernel = kernel
        self.use_fp8 = use_fp8

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

            if attn_mask_pass and dropout_p_pass and shapes_pass and dtype_pass and head_dim_pass:
                kwargs.pop("attn_mask", None)
                kwargs.pop("dropout_p", None)
                kwargs.pop("enable_gqa", None)
                kwargs["softmax_scale"] = kwargs.pop("scale", None)
                return _flash_attention3(*args, **kwargs, kernel=self.kernel, use_fp8=self.use_fp8)
            else:
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)


def _flash_attention3(query, key, value, *, is_causal=False, softmax_scale=None, kernel=None, use_fp8=False):
    # convert (B, H, S, D) → (B, S, H, D)
    q, k, v = [x.transpose(1, 2).contiguous() for x in (query, key, value)]
    _ops = torch.ops.flash_attn_pruna
    op_fn = _ops._flash_attn_forward_fp8 if use_fp8 else _ops._flash_attn_forward
    out = op_fn(q, k, v, causal=is_causal, softmax_scale=softmax_scale)  # ty: ignore[invalid-argument-type]
    # back to (B, H, S, D) for the rest of the pipeline
    return out.transpose(1, 2)


def _is_fp16_or_bf16(module: torch.nn.Module, path: str | None = None) -> bool:
    """Check if a module's dtype is float16 or bfloat16."""
    try:
        dtype = module.dtype
    except AttributeError:
        first_param = next(module.parameters(), None)
        dtype = first_param.dtype if first_param is not None else None
    return dtype in (torch.bfloat16, torch.float16)


def _apply_via_backend(
    root_name: str | None,
    root_nn_module: torch.nn.Module,
    relative_target_paths: List[str],
    backend: str,
) -> torch.nn.Module:
    """
    Apply FA3 by setting the attention backend on targeted submodules.

    Used for standard diffusers pipeline models.

    The model check function already ensures that the model has at least one attention module, which the algorithm
    will be applied to.

    Parameters
    ----------
    root_name : str | None
        The attribute name of the root in the model (None if model is an nn.Module).
    root_nn_module : torch.nn.Module
        The root nn.Module.
    relative_target_paths : List[str]
        Relative paths to targeted submodules within the root.
    backend : str
        The backend name to set.

    Returns
    -------
    torch.nn.Module
        The (modified) root module.
    """
    for rel_path in relative_target_paths:
        try:
            sub_module = root_nn_module.get_submodule(rel_path)
        except AttributeError:
            continue
        if hasattr(sub_module, "set_attention_backend"):
            sub_module.set_attention_backend(backend)
    return root_nn_module


def _apply_via_forward_wrap(
    root_name: str | None,
    root_nn_module: torch.nn.Module,
    relative_target_paths: List[str],
    kernel: Any,
    use_fp8: bool,
) -> torch.nn.Module:
    """
    Apply FA3 by wrapping individual attention module forwards with FlashAttention3Context.

    If the module is already wrapped by a previous pass, unwrap to the true original and wrap again.

    Used for custom (non-diffusers-pipeline) models.

    Since the generic model check cannot guarantee the model contains attention modules, it is
    the caller's responsibility to ensure the targeted submodules actually use ``scaled_dot_product_attention``.
    If there are no attention modules in the model, the algorithm just runs silently and does nothing.

    Parameters
    ----------
    root_name : str | None
        The attribute name of the root in the model (None if model is an nn.Module).
    root_nn_module : torch.nn.Module
        The root nn.Module.
    relative_target_paths : List[str]
        Relative paths to targeted submodules within the root.
    kernel : Any
        The flash attention 3 kernel module.
    use_fp8 : bool
        Whether to quantize Q, K, V to FP8 before the attention computation.

    Returns
    -------
    torch.nn.Module
        The (modified) root module.
    """
    for rel_path in relative_target_paths:
        try:
            sub_module = root_nn_module.get_submodule(rel_path)
        except AttributeError:
            continue
        original_forward = sub_module.forward

        # If already wrapped by a previous FA3 pass, unwrap to the true original
        # to avoid nested TorchFunctionMode contexts (inner would always win).
        while getattr(original_forward, "_is_fa3_pruna_wrap", False):
            original_forward = original_forward.__wrapped__

        @functools.wraps(original_forward)
        def new_forward(*args, _orig=original_forward, _kernel=kernel, _fp8=use_fp8, **kwargs):
            with FlashAttention3Context(kernel=_kernel, use_fp8=_fp8):
                return _orig(*args, **kwargs)

        # Add flag such that only the FA3 wrapper is removed when unwrapping.
        new_forward._is_fa3_pruna_wrap = True
        sub_module.forward = new_forward
    return root_nn_module


def _quantize_fp8(t: torch.Tensor, descale_shape: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-tensor absmax quantization to FP8 E4M3.

    Parameters
    ----------
    t : torch.Tensor
        The input tensor (BF16 or FP16), shape (B, S, H, D).
    descale_shape : Tuple[int, int]
        The required shape for the descale tensor, typically (batch_size, num_heads_k) -> per tensor quantization.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The FP8 tensor and the descale factor with the requested shape (float32).
    """
    amax = t.abs().amax()
    # E4M3 max representable value is 448.0
    scale = (448.0 / amax.clamp(min=1e-12))
    t_fp8 = (t * scale).to(torch.float8_e4m3fn)
    descale = torch.full(descale_shape, 1.0 / scale.item(), dtype=torch.float32, device=t.device)
    return t_fp8, descale


def register_pruna_flash_attn_op(kernel_mod: Any, use_fp8: bool = False) -> None:
    """
    Register the flash attention 3 operation with torch ops to make it compatible with fullgraph compilation.

    Parameters
    ----------
    kernel_mod : Any
        The flash attention 3 kernel module.
    use_fp8 : bool
        Whether to quantize Q, K, V to FP8 (E4M3) before the attention computation.
    """
    flash_attn_cuda = kernel_mod.flash_attn_func
    op_name = "flash_attn_pruna::_flash_attn_forward_fp8" if use_fp8 else "flash_attn_pruna::_flash_attn_forward"

    @torch.library.custom_op(op_name, mutates_args=(), device_types="cuda")
    def _flash_attn_forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: float | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        if use_fp8:
            # FA3 requires descale shape (batch_size, num_heads_k) for all three,
            # as kernel expects per tensor quantization
            # Quantize sequentially, reassigning to drop fp16 originals and reduce peak memory (otherwise risk of OOM)
            descale_shape = (q.shape[0], k.shape[2])  # (B, H) as input format = (B, N, H, D)
            q, descale_q = _quantize_fp8(q, descale_shape)
            k, descale_k = _quantize_fp8(k, descale_shape)
            v, descale_v = _quantize_fp8(v, descale_shape)
            result = flash_attn_cuda(
                q, k, v,
                softmax_scale=softmax_scale or None,
                causal=causal,
                deterministic=False,
                q_descale=descale_q,
                k_descale=descale_k,
                v_descale=descale_v,
            )
        else:
            result = flash_attn_cuda(q, k, v, softmax_scale=softmax_scale or None, causal=causal, deterministic=False)
        # Some kernel builds return (out, lse), others return just out, depending on torch and cuda version
        if isinstance(result, tuple):
            return result[0]
        return result

    @torch.library.register_fake(op_name)
    def _flash_attn_forward_fake(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: float | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        return torch.empty_like(q)

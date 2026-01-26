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

from collections.abc import Iterable
from typing import Any, Dict, Optional, List

import torch
from aenum import extend_enum
from diffusers import DiffusionPipeline
from diffusers import __version__ as diffusers_version
from packaging.version import Version

from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag as tags
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.engine.save import SAVE_FUNCTIONS
from pruna.logging.logger import pruna_logger
from pruna.config.target_modules import TARGET_MODULES_TYPE, TargetModules, map_targeted_nn_roots


class SpargeAttn(PrunaAlgorithmBase):
    """
    Replace torch.nn.functional.scaled_dot_product_attention with sparge_attn.

    SpargeAttention is a fast and memory-efficient attention mechanism. It applies the flash attention mechanism
    in combination with quantization, smoothing, and sparsity to speed up attention computations.
    """

    algorithm_name: str = "sparge_attn"
    group_tags: list[str] = [tags.KERNEL]
    save_fn = SAVE_FUNCTIONS.reapply
    references: dict[str, str] = {
        "Paper": "https://arxiv.org/pdf/2502.18137",
        "GitHub": "https://github.com/thu-ml/SpargeAttn",
    }
    tokenizer_required: bool = False
    processor_required: bool = False
    runs_on: list[str] = ["cuda", "accelerate"]
    dataset_required: bool = False
    compatible_before: Iterable[str] = [tags.QUANTIZER]
    compatible_after: Iterable[str] = ["torch_compile", tags.CACHER]

    @property
    def required_install(self) -> str | None:
        # Instruction string. Pruna will display it on import errors.
        return (
            "Requires the SpargeAttn CUDA package providing `spas_sage_attn`."
            "Please install the pruna extension pip install pruna[sparge-attn] or" 
            "install the sparge-attn package manually from the github repository:"
            "https://github.com/thu-ml/SpargeAttn"
        )

    def model_check_fn(self, model: Any) -> bool:
        """
        Check if the model has an self-attention mechanism that can be replaced with sparge_attn.

        Parameters
        ----------
        model : Any
            The model to check.

        Returns
        -------
        bool
            True if the model is a valid model for the algorithm, False otherwise.
        """

        if not isinstance(model, DiffusionPipeline) or not hasattr(model, "components"):
            return False

        for component in model.components.values():
            if not hasattr(component, "modules"):

                continue
            for m in component.modules():
                if not hasattr(m, "set_attention_backend"):
                    continue

                # self-attn check
                if hasattr(m, "is_cross_attention"):
                    if m.is_cross_attention is False:
                        return True
                    continue

                # fallback heuristic as a safety net
                if getattr(m, "cross_attention_dim_head", None) is None:
                    return True

        return False

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Wrap the model to use sparge_attn where possible.

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
        # register the sparge_attn operation with torch ops to make it compatible with full-graph compilation
        register_pruna_sparge_attn_op()

        register_custom_backend(self.import_algorithm_packages())

        target_modules = smash_config["target_modules"]

        if target_modules is None:
            target_modules = self.get_model_dependent_hyperparameter_defaults(
                model,
                smash_config
            )

        def apply_sparge_attn(
            root_name: str | None,
            root_nn_module: torch.nn.Module,
            relative_target_paths: List[str],
        ) -> torch.nn.Module:
            """
            Apply the SageAttention backend to targeted submodules of a root module.

            For each relative submodule path, this function retrieves the corresponding
            submodule from ``root_nn_module`` and applies
            ``set_attention_backend("sage_hub")`` if the method is available.

            Parameters
            ----------
            root_name : str or None
                The attribute name of the root module within the model (used for identification).
                May be ``None`` if the model itself is a ``torch.nn.Module``.
            root_nn_module : torch.nn.Module
                The root torch.nn.module containing the targeted submodules.
            relative_target_paths : List[str]
                Relative paths of submodules (with respect to ``root_nn_module``) to consider.

            Returns
            -------
            torch.nn.Module
                The root ntorch.nn.module with the SageAttention backend applied where supported.
            """
            for rel_path in relative_target_paths:
                try:
                    sub_module = root_nn_module.get_submodule(rel_path)
                except AttributeError:
                    continue

                if not hasattr(sub_module, "set_attention_backend"):
                    continue

                # SpargeAttn is only applicable to self-attention blocks
                # Applying spargeattn to other modules than self attn yields reduced output quality
                # or undefined behavior
                if hasattr(sub_module, "is_cross_attention"):
                    if sub_module.is_cross_attention:
                        continue
                elif hasattr(sub_module, "cross_attention_dim_head"):
                    if sub_module.cross_attention_dim_head is not None:
                        continue
                else:
                    pruna_logger.warning(
                        f"Skipping {root_name}.{rel_path} as it does not contain a self-attention module."
                        "SpargeAttn is intended for self-attention blocks only."
                    )
                    continue

                sub_module.set_attention_backend("sparge_attn_pruna")

            return root_nn_module

        return map_targeted_nn_roots(apply_sparge_attn, model, target_modules)

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Import the algorithm packages.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        if Version(diffusers_version) < Version("0.35.0.dev0"):
            raise ImportError(
                f"SpargeAttn requires diffusers>=0.35.0.dev0 for attention_dispatch backends; got {diffusers_version}."
            )

        from diffusers.models.attention_dispatch import (  # noqa: PLC0415
            AttentionBackendName,
            _AttentionBackendRegistry,
            _check_device,
            _check_shape,
            _native_attention,
        )

        packages = {
                "_AttentionBackendRegistry": _AttentionBackendRegistry,
                "_check_device": _check_device,
                "_check_shape": _check_shape,
                "_native_attention": _native_attention,
                "AttentionBackendName": AttentionBackendName,
            }
        return packages

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
            TargetModules(name="target_modules", default_value=None),
        ]

    def get_model_dependent_hyperparameter_defaults(
        self,
        model: Any,
        smash_config: SmashConfigPrefixWrapper,
    ) -> TARGET_MODULES_TYPE:
        """
        Get model-dependent default hyperparameters for this algorithm.

        Parameters
        ----------
        model : Any
            The model/pipeline instance for which defaults should be computed.
        smash_config : SmashConfigPrefixWrapper
            The configuration wrapper passed to the algorithm. It can be used to read other
            algorithm settings when selecting defaults.

        Returns
        -------
        TARGET_MODULES_TYPE
            A dictionary with keys "include" and "exclude" defining which modules should be
            targeted by default.
        """
        # We include all transformer blocks by default. Further filtering is done in _apply().
        include = ["transformer*"]
        exclude = []

        return {"include": include, "exclude": exclude}


def register_custom_backend(imported_packages: Dict[str, Any]) -> None:
    """
    Register the attention backend for sparge_attn by mimicing the native backend.

    Applies to diffusers >= 0.35.0.dev0.

    Parameters
    ----------
    imported_packages : Dict[str, Any]
        The imported packages.
    """

    attention_backend_registry = imported_packages["_AttentionBackendRegistry"]
    _check_device = imported_packages["_check_device"]
    _check_shape = imported_packages["_check_shape"]
    _native_attention = imported_packages["_native_attention"]
    attention_backend_name = imported_packages["AttentionBackendName"]

    if attention_backend_registry.get_active_backend()[0].name != "NATIVE":
        pruna_logger.warning(
            "The current active attention backend is not native. This might lead to unexpected behavior."
        )

    if "SPARGE_ATTN_PRUNA" not in attention_backend_name.__members__:
        @attention_backend_registry.register(
            "sparge_attn_pruna",
            # check that all tensors are on the same device
            # check that all tensors have the same shape
            constraints=[_check_device, _check_shape],
        )
        def _sparge_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            dropout_p: float = 0.0,
            is_causal: bool = False,
            scale: Optional[float] = None,
            attention_kwargs: Optional[Dict[str, Any]] = None,
            # unsupported by sparge_attn but we catch them to reroute to native attention if necessary
            enable_gqa: bool = False,
        ) -> torch.Tensor:
            attention_kwargs = attention_kwargs or {}

            # dtype handling: SpargeAttn supports all dtype as casting is handled in the actual implementation

            # Tensor layout handling: To be consistent with the huggingface backend we use the NHD layout and do
            # not allow HND
            if attention_kwargs.get("tensor_layout", "NHD") != "NHD":
                pruna_logger.warning("HND layout is not supported for SpargeAttn. Using NHD layout instead.")
                attention_kwargs["tensor_layout"] = "NHD"

            # Shape handling: 
            shape_pass = True

            # SpargeAttn is only applicable to self-attention blocks
            # Safety net: Should not happen since we already filter in _apply()
            if query.shape[1] != key.shape[1] or query.shape[1] != value.shape[1]:
                shape_pass = False

            # SpargeAttn requires sequence length to be not less than 128.  
            if query.shape[1] < 128:
                shape_pass = False

            # SpargeAttn requires head dimension to be 64 or 128.
            if query.shape[-1] not in [64, 128]:
                shape_pass = False

            # kwargs handling: Check for unsupported kwargs
            kwargs_pass = True
            allowed_kwargs = {
                "smooth_k",
                "simthreshd1",
                "cdfthreshd",
                "topk",
                "pvthreshd",
                "attention_sink",
                "tensor_layout",
            }
            for kw_name in attention_kwargs.keys():
                if kw_name not in allowed_kwargs:
                    kwargs_pass = False

            # if any constraints are not met or unsupported input arguments are being used, reroute to native attention
            # SpargeAttn kernels currently don't support masks or dropout; reroute in those cases.
            if attn_mask is not None or dropout_p != 0.0 or enable_gqa or not shape_pass or not kwargs_pass:
                pruna_logger.debug(
                    "Rerouting to native attention. Check the following criteria in algorithms/kernels/sparge_attn.py: "
                    f"attn_mask_pass: {attn_mask is None}, dropout_p_pass: {dropout_p == 0.0}, "
                    f"enable_gqa_pass: {not enable_gqa}, shape_pass: {shape_pass}, kwargs_pass: {kwargs_pass}"
                )
                return _native_attention(
                    query=query,
                    key=key,
                    value=value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                    enable_gqa=enable_gqa,
                )
            else:
                pruna_logger.debug("Using SpargeAttn...")
                out = torch.ops.sparge_attn_pruna._sparge_attn_forward(
                    q=query,
                    k=key,
                    v=value,
                    is_causal=is_causal,
                    scale=scale,
                    **attention_kwargs,
                )
                return out

        extend_enum(attention_backend_name, "SPARGE_ATTN_PRUNA", "sparge_attn_pruna")

def register_pruna_sparge_attn_op() -> None:
    """
    Register the sparge_attn operation with torch ops to make it compatible with fullgraph compilation.
    """
    @torch.library.custom_op("sparge_attn_pruna::_sparge_attn_forward", mutates_args=(), device_types="cuda")
    def _sparge_attn_forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
        smooth_k: bool = True,
        simthreshd1: float = -0.1,
        cdfthreshd: Optional[float] = None,
        topk: float = 0.5,
        pvthreshd: int = 50,
        attention_sink: bool = False,
        # We choose NHD to be consistent with the native backend
        tensor_layout: str = "NHD",
        # We ignore the output_dtype argument as it is not used in the actual implementation
        # The output dtype is float16 if q.dtype = float32 or float16, otherwise it is bfloat16
    ) -> torch.Tensor:
        # Local import to allow lazy package loading.
        # This allows the algorithm module to be importable without the spargeattn package to be installed.
        # Use importlib to avoid static import warnings when `spas_sage_attn` isn't installed.
        import importlib

        spas_sage_attn = importlib.import_module("spas_sage_attn")
        spas_sage2_attn_meansim_topk_cuda = getattr(spas_sage_attn, "spas_sage_attn_meansim_topk_cuda")

        out = spas_sage2_attn_meansim_topk_cuda(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            smooth_k=smooth_k,
            simthreshd1=simthreshd1,
            cdfthreshd=cdfthreshd,
            topk=topk,
            pvthreshd=pvthreshd,
            attention_sink=attention_sink,
            tensor_layout=tensor_layout,
        )
        return out

    @torch.library.register_fake("sparge_attn_pruna::_sparge_attn_forward")
    def _sparge_attn_forward_fake(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
        smooth_k: bool = True,
        simthreshd1: float = -0.1,
        cdfthreshd: Optional[float] = None,
        topk: float = 0.5,
        pvthreshd: int = 50,
        attention_sink: bool = False,
        tensor_layout: str = "NHD",
    ) -> torch.Tensor:
        return torch.empty_like(q)
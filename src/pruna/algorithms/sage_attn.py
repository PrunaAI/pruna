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
from typing import Any, List

import torch
from diffusers import DiffusionPipeline

from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag as tags
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.config.target_modules import TargetModules, map_targeted_nn_roots
from pruna.engine.save import SAVE_FUNCTIONS


class SageAttn(PrunaAlgorithmBase):
    """
    Replace torch.nn.functional.scaled_dot_product_attention with sage_attn.

    SageAttention is a fast and memory-efficient attention mechanism. It applies the flash attention mechanism
    in combination with quantization and smoothing to speed up attention computations.
    """

    algorithm_name: str = "sage_attn"
    group_tags: list[str] = [tags.KERNEL]
    save_fn = SAVE_FUNCTIONS.reapply
    references: dict[str, str] = {
        "Paper (SA2++)": "https://arxiv.org/pdf/2505.21136v3",
        "GitHub": "https://github.com/thu-ml/SageAttention",
        "Kernel Hub": "https://huggingface.co/kernels-community/sage_attention",
    }
    tokenizer_required: bool = False
    processor_required: bool = False
    runs_on: list[str] = ["cuda", "accelerate"]
    dataset_required: bool = False
    compatible_before: Iterable[str] = [tags.QUANTIZER]
    compatible_after: Iterable[str] = ["torch_compile", tags.CACHER]

    def model_check_fn(self, model: Any) -> bool:
        """
        Check if the model has an attention mechanism that can be replaced with sage_attn.

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

        return any(
            hasattr(component, "set_attention_backend") and component.dtype in (torch.bfloat16, torch.float16)
            for component in model.components.values()
        )

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Wrap the model to use SageAttention where possible.

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
        target_modules = smash_config["target_modules"]

        if target_modules is None:
            target_modules = self.get_model_dependent_hyperparameter_defaults(model, smash_config)

        def apply_sage_attn(
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
                    # safety net: should not happen,
                    # since the paths come from named_modules()
                    continue
                if hasattr(sub_module, "set_attention_backend"):
                    sub_module.set_attention_backend("sage_hub")
            return root_nn_module

        return map_targeted_nn_roots(apply_sage_attn, model, target_modules)

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
    ) -> dict[str, Any]:
        """
        Provide default `target_modules` targeting all transformer modules.

        SageAttn may also be applicable to other modules but could significantly
        decrease model quality, so only transformer modules are included by default.

        Parameters
        ----------
        model : Any
            The model to derive defaults from.
        smash_config : SmashConfigPrefixWrapper
            The algorithm-prefixed configuration.

        Returns
        -------
        TARGET_MODULES_TYPE
            A dictionary with "include" and "exclude" keys defining which modules
            should be targeted by default.
        """
        # We include all transformer modules by default.
        # SageAttn might also be applicable to other modules but could significantly decrease model quality.
        include = ["transformer*"]
        exclude = []

        return {"include": include, "exclude": exclude}

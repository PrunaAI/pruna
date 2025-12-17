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

import fnmatch
import re
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

import torch
from diffusers import DiffusionPipeline

from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag as tags
from pruna.config.hyperparameters import Boolean
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.config.target_modules import TARGET_MODULES_TYPE, TargetModules
from pruna.engine.save import SAVE_FUNCTIONS
from pruna.logging.logger import pruna_logger


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
        exclude_first_and_last_transformer_blocks = smash_config["exclude_first_and_last_transformer_blocks"]

        if target_modules is None:
            target_modules = self.get_model_dependent_hyperparameter_defaults(
                model,
                smash_config
            )  # for consistency, not used yet

        extra_excludes = (_get_transformer_sub_excludes(model) if exclude_first_and_last_transformer_blocks else [])

        include_patterns = target_modules.get("include", [])
        exclude_patterns = target_modules.get("exclude", [])
        exclude_patterns.extend(extra_excludes)

        # Heuristic: if any pattern contains a dot, there are nested rules
        def has_nested_rules(comp_name: str) -> bool:
            prefix = comp_name + "."
            return any(p.startswith(prefix) for p in (include_patterns + exclude_patterns))

        def is_relevant_component_by_include(comp_name: str) -> bool:
            if not include_patterns:
                return True
            if _matches_any(comp_name, include_patterns):   # "*", "transformer", ...
                return True
            prefix = comp_name + "."
            return any(p.startswith(prefix) for p in include_patterns)  # "transformer.*", "transformer.blocks.*"

        def should_apply(name: str) -> bool:
            if exclude_patterns and _matches_any(name, exclude_patterns):
                return False
            return not include_patterns or _matches_any(name, include_patterns)

        for comp_name, component in model.components.items():

            # --- Check component level ---

            # 1) Component-level filter (e.g. exclude "vae"), exclude if in exclude_patterns and not in include_patterns
            if exclude_patterns and _matches_any(comp_name, exclude_patterns):
                continue

            # 2) Pick only relevant components
            if not is_relevant_component_by_include(comp_name):
                continue

            # 3) If there are no nested rules for the current component,
            # make a faster global call otherwise go to submodule level
            if (hasattr(component, "set_attention_backend")
                and not has_nested_rules(comp_name)
                and should_apply(comp_name)):
                component.set_attention_backend("sage_hub")
                continue

            # --- Check submodule level ---

            # 1) Check for named_modules method for step 2) to work
            if component is None or not hasattr(component, "named_modules"):
                continue

            # 2) Nested rules: iterate over submodules and match full_name
            for sub_name, sub_module in component.named_modules():
                if not sub_name:
                    continue

                full_name = f"{comp_name}.{sub_name}"  # e.g., transformer.blocks.0.attn1

                if not should_apply(full_name):
                    continue

                if hasattr(sub_module, "set_attention_backend"):
                    sub_module.set_attention_backend("sage_hub")

        return model

    def get_hyperparameters(self) -> list:
        """Return hyperparameters for this algorithm."""
        return [
            Boolean(
                "exclude_first_and_last_transformer_blocks",
                default=False,
                meta=dict(desc="If True, do NOT apply SageAttention to the first and last"
                "transformer blocks for each transformer component."),
            ),
            TargetModules(name="target_modules", default_value=None),
        ]

    def get_model_dependent_hyperparameter_defaults(
        self,
        model: Any,
        smash_config: SmashConfigPrefixWrapper,
    ) -> TARGET_MODULES_TYPE:
        """Return model-dependent default target_modules."""
        # So far, everything is included and nothing is excluded
        # Filtering is done in the _apply method by the set_attention_backend method
        include = ["*"]
        exclude = []

        return {"include": include, "exclude": exclude}


def _get_transformer_sub_excludes(
    model: Any,
) -> list[str]:
    """
    Returns a flat list of glob patterns.

    Example:
    [
    "transformer.blocks.0*",
    "transformer.blocks.39*",
    "transformer_2.blocks.0*",
    "transformer_2.blocks.39*",
    ]
    """
    excludes: list[str] = []

    roots = _get_transformer_roots(model)

    for root in roots:
        # get the component
        comp = model.components.get(root, None)
        # if the component is None/missing (e.g. the case for transformer_2 in Wan2.2-TI2V-5B-Diffusers), skip it
        if comp is None:
            pruna_logger.warning("skip %s for excludes: component is None", root)
            continue
        # get the attention names
        attn_names = [
            name
            for name, module in model.components[root].named_modules()
            if name and hasattr(module, "set_attention_backend")
        ]
        # if there are no attention names, skip it
        if not attn_names:
            continue

        # get the block paths
        block_paths = _unique_in_order([n.rsplit(".", 1)[0] for n in attn_names])

        # if there are less than 3 block paths, skip it
        if len(block_paths) < 3:
            pruna_logger.warning(f"Root {root} has less than 3 transformer blocks."
            "Thus its first and last blocks are not excluded for sage_attn.")
            continue

        # We just want to exclude the first and last blocks of the transformer components
        excludes.extend([
            f"{root}.{block_paths[0]}*",
            f"{root}.{block_paths[-1]}*",
        ])

    return excludes


def _get_transformer_roots(model: Any) -> list[str]:
    """Get the roots of the transformer components."""
    roots = []
    for name, _ in model.components.items():
        if name == "transformer" or name.startswith("transformer_"):
            roots.append(name)

    # Sort the roots by the number of the transformer component, just to be sure
    def key(n: str) -> int:
        # transformer -> 0, transformer_10 -> 10
        if n == "transformer":
            return 0
        m = re.match(r"transformer_(\d+)$", n)
        return int(m.group(1)) if m else 10**9  # unknown suffix goes to end

    return sorted(roots, key=key)


def _unique_in_order(items: list[str]) -> list[str]:
    return list(OrderedDict.fromkeys(items))


def _matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in (patterns or []))

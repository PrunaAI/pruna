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
from typing import Any, Dict, cast

import torch
from ConfigSpace import CategoricalHyperparameter

from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag as tags
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.config.target_modules import (
    TARGET_MODULES_TYPE,
    TargetModules,
    get_skipped_submodules,
    is_leaf_module,
    map_targeted_nn_roots,
    target_backbone,
)
from pruna.engine.model_checks import is_causal_lm, is_transformers_pipeline_with_causal_lm


class LLMCompressor(PrunaAlgorithmBase):
    """
    Implement AWQ using llmcompressor.

    Activation Aware Quantization (AWQ) is a state-of-the-art technique to quantize the weights of
    large language models which involves using a small calibration dataset to calibrate the model.
    The AWQ algorithm utilizes calibration data to derive scaling factors which reduce the dynamic
    range of weights while minimizing accuracy loss to the most salient weight values.
    """

    algorithm_name: str = "awq"
    group_tags: list[str] = [tags.QUANTIZER]
    references: dict[str, str] = {"GitHub": "https://github.com/vllm-project/llm-compressor"}
    save_fn: None = None
    tokenizer_required: bool = True
    processor_required: bool = False
    dataset_required: bool = True
    runs_on: list[str] = ["cuda"]
    compatible_before: Iterable[str] = []
    compatible_after: Iterable[str] = []

    def get_hyperparameters(self) -> list:
        """
        Configure all algorithm-specific hyperparameters with ConfigSpace.

        Returns
        -------
        list
            The hyperparameters.
        """
        return [
            CategoricalHyperparameter(
                "quant_scheme",
                choices=["W4A16", "W4A16_ASYM"],
                default_value="W4A16",
                meta=dict(desc="Quantization scheme to use. Use symmetric quantization to avoid decompression issues."),
            ),
            TargetModules(
                "target_modules",
                default_value=None,
                meta=dict(
                    desc="Precise choices of which modules to quantize, "
                    "e.g. {include: ['model.*']} to quantize only the language model in a pipeline. "
                    f"See the {TargetModules.documentation_name_with_link} documentation for more details."
                ),
            ),
        ]

    def model_check_fn(self, model: Any) -> bool:
        """
        Check if the model is a causal language model.

        Parameters
        ----------
        model : Any
            The model to check.

        Returns
        -------
        bool
            True if the model is a causal language model, False otherwise.
        """
        return is_causal_lm(model) or is_transformers_pipeline_with_causal_lm(model)

    def get_model_dependent_hyperparameter_defaults(
        self, model: Any, smash_config: SmashConfigPrefixWrapper
    ) -> dict[str, Any]:
        """
        Get the default hyperparameters for the model.

        Parameters
        ----------
        model : Any
            The model to get the default hyperparameters from.
        smash_config : SmashConfigPrefixWrapper
            The configuration for the quantization.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the default target_modules for the algorithm.
        """
        target_modules: TARGET_MODULES_TYPE = target_backbone(model)
        return {"target_modules": target_modules}

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Quantize the model.

        Parameters
        ----------
        model : Any
            The model to quantize.
        smash_config : SmashConfigPrefixWrapper
            The configuration for the quantization.

        Returns
        -------
        Any
            The quantized model.
        """
        imported = self.import_algorithm_packages()
        dataset = smash_config.data.val_dataset

        # For text generation models, provide the tokenizer as processor to avoid AutoProcessor errors
        processor = smash_config.tokenizer if smash_config.tokenizer is not None else "bert-base-uncased"

        target_modules: None | TARGET_MODULES_TYPE = smash_config["target_modules"]
        if target_modules is None:
            defaults = self.get_model_dependent_hyperparameter_defaults(model, smash_config)
            target_modules = cast(TARGET_MODULES_TYPE, defaults["target_modules"])

        def quantize_language_model(
            attr_name: str | None, language_model: torch.nn.Module, subpaths: list[str]
        ) -> torch.nn.Module:
            """
            Quantize the language model.

            Parameters
            ----------
            attr_name : str | None
                The name of the attribute in the model pointing to the language model to quantize.
            language_model : torch.nn.Module
                The language model to quantize.
            subpaths : list[str]
                The subpaths of the language model to quantize.

            Returns
            -------
            torch.nn.Module
                The quantized language model.
            """
            ignore_modules = get_skipped_submodules(language_model, subpaths, filter_fn=is_leaf_module)
            recipe = [
                imported["AWQModifier"](
                    ignore=ignore_modules,
                    scheme=smash_config["quant_scheme"],
                    targets=["Linear"],
                )
            ]
            return imported["oneshot"](model=language_model, recipe=recipe, dataset=dataset, processor=processor)

        model = map_targeted_nn_roots(quantize_language_model, model, target_modules)
        return model

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Provide a algorithm packages for the algorithm.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        from llmcompressor import oneshot
        from llmcompressor.modifiers.awq import AWQModifier

        return {"oneshot": oneshot, "AWQModifier": AWQModifier}

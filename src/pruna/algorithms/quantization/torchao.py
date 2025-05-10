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

from typing import Any, Dict

import torch

from pruna.algorithms.quantization import PrunaQuantizer
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.config.smash_space import CategoricalHyperparameter
from pruna.engine.model_checks import get_diffusers_transformer_models
from pruna.engine.save import SAVE_FUNCTIONS


class TorchaoQuantizer(PrunaQuantizer):
    """
    Implement quantization using the torchao library.

    This algorithm quantizes the weights and activations of linear layers in a model.
    Combined with torch compile, it can give speedups compared to the base model.
    """

    algorithm_name = "torchao"
    references = {"GitHub": "https://huggingface.co/docs/diffusers/quantization/torchao"}
    save_fn = SAVE_FUNCTIONS.save_before_apply
    tokenizer_required = False
    processor_required = False
    run_on_cpu: bool = True
    run_on_cuda: bool = True
    dataset_required = False
    compatible_algorithms = dict(compiler=["torch_compile"])

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
                "quant_type",
                choices=["int4dq", "int4wo", "int8dq", "int8wo", "fp8wo", "fp8dq", "fp8dqrow"],
                default_value="int8dq",
                meta=dict(desc="Quantization type to use."),
            ),
        ]

    def model_check_fn(self, model: Any) -> bool:
        """
        Check if the model is a torch.nn.Module or a diffusers pipeline with a transformer model.

        Parameters
        ----------
        model : Any
            The model to check.

        Returns
        -------
        bool
            True if the model is suitable for torchao quantization, False otherwise.
        """
        transformer_models = get_diffusers_transformer_models()
        if isinstance(model, tuple(transformer_models)):
            return True
        if hasattr(model, "transformer") and isinstance(model.transformer, tuple(transformer_models)):
            return True
        return isinstance(model, torch.nn.Module)

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
        working_model = model.transformer if hasattr(model, "transformer") else model
        imported_modules = self.import_algorithm_packages()
        imported_modules["quantize"](working_model, imported_modules[smash_config["quant_type"]])
        return model

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Provide a algorithm packages for the algorithm.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        from torchao.quantization import (
            float8_dynamic_activation_float8_weight,
            float8_weight_only,
            int4_weight_only,
            int8_dynamic_activation_int4_weight,
            int8_dynamic_activation_int8_weight,
            int8_weight_only,
            quantize_,
        )
        from torchao.quantization.quant_api import PerRow

        return dict(quantize=quantize_,
                    int4dq=int8_dynamic_activation_int4_weight(),
                    int4wo=int4_weight_only(),
                    int8dq=int8_dynamic_activation_int8_weight(),
                    int8wo=int8_weight_only(),
                    fp8wo=float8_weight_only(),
                    fp8dq=float8_dynamic_activation_float8_weight(),
                    fp8dqrow=float8_dynamic_activation_float8_weight(PerRow()),
                )

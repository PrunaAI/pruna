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
from ConfigSpace import CategoricalHyperparameter, UniformFloatHyperparameter

from pruna.algorithms.pruning import PrunaPruner
from pruna.config.smash_config import SmashConfigPrefixWrapper


class TorchUnstructuredPruner(PrunaPruner):
    """
    Implement structured pruning using torch.

    Unstructured pruning sets individual weights to 0 based on criteria such as magnitude, resulting in sparse weight
    matrices that retain the overall model architecture but may require specialized sparse computation support to fully
    exploit the efficiency gains.
    """

    algorithm_name: str = "torch_unstructured"
    references: dict[str, str] = {"GitHub": "https://github.com/pytorch/pytorch"}
    # original model-saving can be retained as is, only parameter values are modified
    save_fn = None
    tokenizer_required: bool = False
    processor_required: bool = False
    runs_on: list[str] = ["cpu", "cuda"]
    dataset_required: bool = False
    compatible_algorithms: dict[str, list[str]] = dict(quantizer=["half"])

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
                "pruning_method",
                choices=[
                    "random",
                    "l1",
                ],
                default_value="l1",
                meta=dict(desc="Pruning method to use."),
            ),
            UniformFloatHyperparameter(
                "sparsity",
                lower=0.0,
                upper=1.0,
                log=False,
                default_value=0.1,
                meta=dict(desc="Sparsity level up to which to prune."),
            ),
        ]

    def model_check_fn(self, model: Any) -> bool:
        """
        Check if the model is a torch.nn.Module.

        Parameters
        ----------
        model : Any
            The model to check.

        Returns
        -------
        bool
            True if the model is a torch.nn.Module, False otherwise.
        """
        return isinstance(model, torch.nn.Module)

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Apply unstructured pruning to the model.

        Parameters
        ----------
        model : Any
            The model to be pruned
        smash_config : SmashConfigPrefixWrapper
            Configuration object containing pruning hyperparameters

        Returns
        -------
        Any
            The pruned model
        """
        for _, module in model.named_modules():
            if hasattr(module, "weight"):
                if smash_config["pruning_method"] == "random":
                    torch.nn.utils.prune.random_unstructured(module, name="weight", amount=smash_config["sparsity"])
                elif smash_config["pruning_method"] == "l1":
                    torch.nn.utils.prune.l1_unstructured(
                        module,
                        name="weight",
                        amount=smash_config["sparsity"],
                        importance_scores=None,
                    )
                torch.nn.utils.prune.remove(module, "weight")
        return model

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Import the algorithm packages.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        return dict()

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

import inspect
from typing import Any, Callable, Dict

import torch

from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag
from pruna.config.hyperparameters import Boolean
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.engine.model_checks import get_diffusers_transformer_models
from pruna.engine.save import SAVE_FUNCTIONS


class Autoquant(PrunaAlgorithmBase):
    """
    Implement autoquantization using the torchao library.

    This algorithm compiles, quantizes and sparsifies weights, gradients, and activations for inference.
    This algorithm is specifically adapted for Image-Gen models.
    """

    algorithm_name: str = "torchao_autoquant"
    group_tags: list[AlgorithmTag] = [AlgorithmTag.QUANTIZER]
    references: dict[str, str] = {"GitHub": "https://huggingface.co/docs/diffusers/quantization/torchao"}
    save_fn: SAVE_FUNCTIONS = SAVE_FUNCTIONS.save_before_apply
    tokenizer_required: bool = False
    processor_required: bool = False
    runs_on: list[str] = ["cpu", "cuda"]
    dataset_required: bool = False

    def get_hyperparameters(self) -> list:
        """
        Configure all algorithm-specific hyperparameters with ConfigSpace.

        Returns
        -------
        list
            The hyperparameters.
        """
        return [
            Boolean("compile", default=True, meta=dict(desc="Whether to compile the model after quantization or not.")),
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
            True if the model is a causal language model, False otherwise.
        """
        transformer_models = get_diffusers_transformer_models()

        if isinstance(model, tuple(transformer_models)):
            return True

        for _, attr_value in inspect.getmembers(model):
            if isinstance(attr_value, tuple(transformer_models)):
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
        transformer_models = get_diffusers_transformer_models()

        working_model = model.transformer if isinstance(model.transformer, tuple(transformer_models)) else model

        if smash_config["compile"]:
            working_model.torch_compiler = TorchCompiler(working_model)

            # Now we can compile the model
            working_model = working_model.torch_compiler.compile()
        working_model = self.import_algorithm_packages()["autoquant"](working_model, error_on_unseen=False)

        if isinstance(model.transformer, tuple(transformer_models)):
            model.transformer = working_model
        else:
            model = working_model

        return model

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Provide a algorithm packages for the algorithm.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        from torchao.quantization import autoquant

        return dict(autoquant=autoquant)


class TorchCompiler(object):
    """
    A class that compiles a PyTorch model using the pre-defined compilation options.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to be compiled.
    """

    def __init__(self, model: Callable[..., Any]) -> None:
        """
        Initialize the TorchCompiler with a model and a configuration.

        Parameters
        ----------
        model : Callable[..., Any]
            The PyTorch model to be compiled.
        smash_config : dict
            A configuration dictionary that contains the settings for the compilation process.
        """
        self.model = model

    def compile(self) -> Callable[..., Any]:
        """
        Compile the PyTorch model using options provided in the `smash_config`.

        Returns
        -------
        torch.nn.Module
            The compiled PyTorch model.
        """
        self.model = torch.compile(self.model, mode="max-autotune-no-cudagraphs", fullgraph=True)

        return self.model

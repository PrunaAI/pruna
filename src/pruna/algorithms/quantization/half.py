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
from pruna.engine.save import SAVE_FUNCTIONS


class HalfQuantizer(PrunaQuantizer):
    """
    Implement half precision quantization using torch.

    Converting model parameters to half precision (FP16) reduces memory usage and can accelerate computations on GPUs
    that support it.
    """

    algorithm_name: str = "half"
    references: dict[str, str] = {"GitHub": "https://github.com/pytorch/pytorch"}
    # the half-helper is not saved with the model but is fast to reattach
    save_fn: SAVE_FUNCTIONS = SAVE_FUNCTIONS.save_before_apply
    tokenizer_required: bool = False
    processor_required: bool = False
    runs_on: list[str] = ["cpu", "cuda", "accelerate"]
    dataset_required: bool = False
    compatible_algorithms: dict[str, list[str]] = dict(
        batcher=["ifw", "whisper_s2t"],
        cacher=["deepcache"],
        compiler=[
            "c_translate",
            "c_generate",
            "c_whisper",
            "stable_fast",
            "torch_compile",
        ],
        pruner=["torch_structured", "torch_unstructured"],
    )

    def get_hyperparameters(self) -> list:
        """
        Configure all algorithm-specific hyperparameters with ConfigSpace.

        Returns
        -------
        list
            The hyperparameters.
        """
        return []

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
        model.half()
        for param in model.parameters():
            param.requires_grad = False

        original_forward = model.forward

        def new_forward(*args: Any, **kwargs: Any) -> Any:
            args = tuple(arg.half() if hasattr(arg, "half") else arg for arg in args)
            kwargs = {k: v.half() if hasattr(v, "half") else v for k, v in kwargs.items()}
            return original_forward(*args, **kwargs)

        model.forward = new_forward

        return model

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Provide a algorithm packages for the algorithm.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        return dict()

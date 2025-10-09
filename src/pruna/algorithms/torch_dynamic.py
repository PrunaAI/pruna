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
from typing import Any

import torch
from ConfigSpace import OrdinalHyperparameter

from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag as tags
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.engine.save import SAVE_FUNCTIONS


class TorchDynamic(PrunaAlgorithmBase):
    """
    Implement dynamic quantization using torch.

    This technique converts model weights to lower precision (typically int8) dynamically at runtime,
    reducing model size and improving inference speed with minimal impact on accuracy and without the need
    for calibration data.
    """

    algorithm_name = "torch_dynamic"
    group_tags: list[str] = [tags.QUANTIZER]
    references: dict[str, str] = {"GitHub": "https://github.com/pytorch/pytorch"}
    save_fn: SAVE_FUNCTIONS = SAVE_FUNCTIONS.pickled
    tokenizer_required: bool = False
    processor_required: bool = False
    runs_on: list[str] = ["cpu", "cuda"]
    dataset_required: bool = False
    compatible_before: list[str] = []
    compatible_after: list[str] = []

    def get_hyperparameters(self) -> list:
        return [
            OrdinalHyperparameter(
                "weight_bits",
                sequence=["quint8", "qint8"],
                default_value="qint8",
                meta=dict(desc="Tensor type to use for quantization."),
            ),
        ]

    def model_check_fn(self, model: Any) -> bool:
        if isinstance(model, torch.nn.Module):
            return True

        return any(isinstance(attr_value, torch.nn.Module) for _, attr_value in inspect.getmembers(model))

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        if isinstance(model, torch.nn.Module):
            device = next(model.parameters()).device.type
            modules_to_quantize: set[type[torch.nn.Module]] = set()
            if device == "cuda":
                # Linear layers are not serializable by PyTorch when using CUDA
                modules_to_quantize = {torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d}
            else:
                modules_to_quantize = {torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear}

            quantized_model = torch.quantization.quantize_dynamic(
                model,
                modules_to_quantize,
                dtype=getattr(torch, smash_config["weight_bits"]),
                inplace=True,
            )
            return quantized_model
        else:
            for attribute_name, attribute_value in inspect.getmembers(model):
                if isinstance(attribute_value, torch.nn.Module):
                    quantized_attribute = TorchDynamicQuantizer()._apply(attribute_value, smash_config)
                    setattr(model, attribute_name, quantized_attribute)
            return model

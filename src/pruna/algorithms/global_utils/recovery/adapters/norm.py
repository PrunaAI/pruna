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

import torch

from pruna.algorithms.global_utils.recovery.adapters import PrunaAdapter, utils

# Normalization layers must be scraped by class name to match both torch, diffusers and other implementations.
# Matching by module names also does not work because it ends up matching e.g. AdaLayerNormZero which contain
# high dimensional linear layers.
NORM_CLASS_NAMES: tuple[str, ...] = (
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "LayerNorm",
    "GroupNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "RMSNorm",
    "LlamaRMSNorm",
)


class NormAdapter(PrunaAdapter):
    """Adapter for norm finetuning."""

    adapter_prefix = "norm"

    @classmethod
    def get_hyperparameters(cls, *args, **kwargs) -> list:
        """
        Configure all method-specific hyperparameters with ConfigSpace.

        Parameters
        ----------
        *args : tuple
            The arguments for the adapter.
        **kwargs : dict
            The keyword arguments for the adapter.

        Returns
        -------
        list
            The hyperparameters.
        """
        return []

    @classmethod
    def activate(cls, model: torch.nn.Module, *args, **kwargs) -> tuple[torch.nn.Module, int, int]:
        """
        Activate all normalization layers for training.

        Parameters
        ----------
        model : torch.nn.Module
            The model containing the normalization layers.
        *args : Any
            Unused arguments.
        **kwargs : Any
            Unused keyword arguments.

        Returns
        -------
        torch.nn.Module
            The model with the normalization layers activated.
        int
            The number of trainable normalization parameters.
        int
            The number of skipped normalization parameters.
        """
        num_activ_param, num_skip_param = utils.unfreeze_submodules_by_class_name(model, target_classes=NORM_CLASS_NAMES)

        return model, num_activ_param, num_skip_param

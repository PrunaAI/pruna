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


class BiasAdapter(PrunaAdapter):
    """Adapter for bias finetuning."""

    adapter_prefix = "bias"

    @classmethod
    def get_hyperparameters(cls, *args, **kwargs) -> list:
        """
        Configure all method-specific hyperparameters with ConfigSpace.

        Parameters
        ----------
        *args : Any
            Unused arguments.
        **kwargs : Any
            Unused keyword arguments.

        Returns
        -------
        list
            The hyperparameters.
        """
        return []

    @classmethod
    def activate(cls, model: torch.nn.Module, *args, **kwargs) -> tuple[torch.nn.Module, int, int]:
        """
        Activate all biases for training.

        Parameters
        ----------
        model : torch.nn.Module
            The model containing the biases.
        *args : Any
            Unused additional arguments.
        **kwargs : Any
            Unused additional keyword arguments.

        Returns
        -------
        torch.nn.Module
            The model with the biases activated.
        int
            The number of trainable bias parameters.
        int
            The number of skipped bias parameters.
        """
        num_activ_param, num_skip_param = utils.unfreeze_parameters_by_name(model, target_modules=("bias",))
        return model, num_activ_param, num_skip_param

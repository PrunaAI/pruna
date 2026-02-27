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

import torch

from pruna.algorithms.global_utils.recovery.adapters import PrunaAdapter, utils
from pruna.logging.logger import pruna_logger


class HeadAdapter(PrunaAdapter):
    """Adapter for finetuning the model's head while keeping the backbone as is."""

    adapter_prefix = "head"

    @classmethod
    def get_hyperparameters(cls, *args, **kwargs) -> list:
        """
        Configure all method-specific hyperparameters with ConfigSpace.

        Parameters
        ----------
        *args : tuple
            The arguments for the adapter.
        **kwargs : dict
            The hyperparameters for the adapter.

        Returns
        -------
        list
            The hyperparameters.
        """
        return []

    @classmethod
    def activate(cls, model: torch.nn.Module, *args, **kwargs) -> tuple[torch.nn.Module, int, int]:
        """
        Activate the model's head for training.

        Parameters
        ----------
        model : torch.nn.Module
            The model containing the head.
        *args : tuple
            The arguments for the adapter.
        **kwargs : dict
            The hyperparameters for the adapter.

        Returns
        -------
        torch.nn.Module
            The model with the head activated.
        int
            The number of trainable head parameters.
        int
            The number of skipped head parameters.
        """
        # find head from type and name
        model_heads = [
            component
            for comp_name, component in inspect.getmembers(model)
            if isinstance(component, torch.nn.Linear) and "head" in comp_name.lower()
        ]
        if len(model_heads) != 1:
            # = 0: model with no head, e.g. diffusers
            # > 1: model with multiple heads, e.g. for localization, not currently supported
            model_head_names = [
                comp_name
                for comp_name, component in inspect.getmembers(model)
                if isinstance(component, torch.nn.Linear) and "head" in comp_name.lower()
            ]
            pruna_logger.warning(
                f"Found multiple heads but expected only one: {model_head_names}. Skipping head finetuning."
            )
            return model, 0, 0
        model_head = model_heads[0]

        # unfreeze head parameters, recording the number of trainable and skipped parameters
        num_activ_param, num_skip_param = 0, 0
        for param in model_head.parameters():
            if utils.is_trainable(param):
                param.requires_grad = True
                num_activ_param += int(param.numel())
            else:
                num_skip_param += int(param.numel())

        return model, num_activ_param, num_skip_param

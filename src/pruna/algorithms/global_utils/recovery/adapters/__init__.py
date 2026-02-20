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

from abc import ABC, abstractmethod
from typing import Any

import torch

from pruna.config.smash_config import SmashConfigPrefixWrapper


class PrunaAdapter(ABC):
    """Base class for adapters, defining which parameters to finetune for recovery."""

    @property
    @abstractmethod
    def adapter_prefix(self) -> str:
        """The prefix of the adapter to use in the config."""
        pass

    @classmethod
    @abstractmethod
    def get_hyperparameters(cls, task_name: str, **override_defaults: Any) -> list:
        """
        Configure all algorithm-specific hyperparameters with ConfigSpace.

        Parameters
        ----------
        task_name : str
            The name of the task, e.g. "text-to-image" or "text-to-text".
        **override_defaults : Any
            Values used to override the default hyperparameters when using multiple finetuners together.

        Returns
        -------
        list
            The hyperparameters.
        """
        pass

    @classmethod
    @abstractmethod
    def activate(
        cls,
        model: torch.nn.Module,
        smash_config: SmashConfigPrefixWrapper,
        seed: int | None = None,
    ) -> tuple[torch.nn.Module, int, int]:
        """
        Activate or create the parameters in the model corresponding to the adapter.

        Parameters
        ----------
        model : torch.nn.Module
            The model to apply the component to.
        smash_config : SmashConfigPrefixWrapper
            The configuration for the component.
        seed : int
            The seed to use for the adapter if it requires initialization.

        Returns
        -------
        torch.nn.Module
            The model with the adapter activated.
        int
            The number of trainable parameters.
        int
            The number of skipped parameters.
        """
        pass

    @classmethod
    def pre_smash_hook(
        cls, model: torch.nn.Module, smash_config: SmashConfigPrefixWrapper, seed: int | None = None
    ) -> None:
        """
        Optional hook to prepare the model/config before smashing.

        Parameters
        ----------
        model : torch.nn.Module
            The model to prepare.
        smash_config : SmashConfigPrefixWrapper
            Configuration scoped to this adapter.
        seed : int | None
            Optional seed for deterministic initialization.
        """
        pass

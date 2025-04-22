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

import os
from abc import ABC, abstractmethod
from typing import Any, Dict

from pruna.config.smash_config import SmashConfig, SmashConfigPrefixWrapper
from pruna.config.smash_space import SMASH_SPACE
from pruna.engine.save import (
    SAVE_BEFORE_SMASH_CACHE_DIR,
    SAVE_FUNCTIONS,
    save_pruna_model,
)


class PrunaAlgorithmBase(ABC):
    """Base class for Pruna algorithms."""

    def __init__(self) -> None:
        self.hyperparameters = self.get_hyperparameters()

        # register algorithm in its config group
        SMASH_SPACE.register_algorithm(self.algorithm_group, self.algorithm_name)

        # register hyperparameters conditional on the algorithm
        hyperparameters = self.get_hyperparameters()
        SMASH_SPACE.register_algorithm_arguments(self.algorithm_name, hyperparameters, self.algorithm_group)

        # register allowed combinations
        SMASH_SPACE.register_allowed_combinations(self.algorithm_name, self.compatible_algorithms)

        # register argument requirements
        SMASH_SPACE.model_requirements[self.algorithm_name] = dict(
            dataset_required=self.dataset_required,
            tokenizer_required=self.tokenizer_required,
            processor_required=self.processor_required,
        )

    @classmethod
    def compatible_devices(cls) -> list[str]:
        """Return the compatible devices for the algorithm."""
        compatible_devices = []
        if cls.run_on_cpu:  # type: ignore
            compatible_devices.append("cpu")
        if cls.run_on_cuda:  # type: ignore
            compatible_devices.append("cuda")
        if not compatible_devices:
            raise ValueError(f"Algorithm {cls.algorithm_name} is not compatible with any device.")
        return compatible_devices

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """Subclasses need to provide a name for the algorithm."""
        pass

    @property
    def required_install(self) -> str | None:
        """Subclasses need to provide extra requirements for the algorithm."""
        return None

    @property
    @abstractmethod
    def references(self) -> None | dict[str, str]:
        """References likes papers or GitHub repository for the algorithm."""
        pass

    @property
    @abstractmethod
    def run_on_cpu(self) -> bool:
        """Subclasses need to provide a boolean indicating if the algorithm can be applied on cpu."""
        pass

    @property
    @abstractmethod
    def run_on_cuda(self) -> bool:
        """Subclasses need to provide a boolean indicating if the algorithm can be applied on cuda."""
        pass

    @property
    @abstractmethod
    def compatible_algorithms(self) -> Dict[str, list[str]]:
        """Subclasses need to provide a list of compatible algorithms."""
        pass

    @property
    @abstractmethod
    def save_fn(self) -> SAVE_FUNCTIONS | None:
        """Subclasses need to provide a save_fn for the algorithm."""
        pass

    @property
    @abstractmethod
    def tokenizer_required(self) -> bool:
        """Subclasses need to request a tokenizer for the algorithm."""
        pass

    @property
    @abstractmethod
    def processor_required(self) -> bool:
        """Subclasses need to request a processor for the algorithm."""
        pass

    @property
    @abstractmethod
    def dataset_required(self) -> bool:
        """Subclasses need to request a dataset for the algorithm."""
        pass

    @property
    @abstractmethod
    def algorithm_group(self) -> str:
        """Return the config group (i.e. "quantizer", "pruner" or "compiler") of the algorithm."""
        pass

    @abstractmethod
    def model_check_fn(self, model: Any) -> bool:
        """
        Provide a model check function for the algorithm.

        Parameters
        ----------
        model : Any
            The model to check.

        Returns
        -------
        bool
            True if the model is supported, False otherwise.
        """
        pass

    @abstractmethod
    def import_algorithm_packages(self) -> Dict[str, Any]:
        """Provide a algorithm packages for the algorithm."""
        pass

    @abstractmethod
    def get_hyperparameters(self) -> list:
        """Configure all algorithm-specific hyperparameters with ConfigSpace."""
        pass

    @abstractmethod
    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """Apply the algorithm to the model."""
        pass

    def apply(self, model: Any, smash_config: SmashConfig) -> Any:
        """
        Wrap the apply algorithm for e.g. saving callbacks.

        Parameters
        ----------
        model : Any
            The model to apply the algorithm to.
        smash_config : SmashConfig
            The SmashConfig object containing the save and load functions.

        Returns
        -------
        Any
            The model after the algorithm has been applied.
        """
        if self.save_fn == SAVE_FUNCTIONS.save_before_apply and smash_config._prepare_saving:
            save_dir = os.path.join(smash_config.cache_dir, SAVE_BEFORE_SMASH_CACHE_DIR)
            save_pruna_model(model, save_dir, smash_config)

        # save algorithms to reapply after loading
        if self.save_fn == SAVE_FUNCTIONS.save_before_apply or self.save_fn == SAVE_FUNCTIONS.reapply:
            smash_config.reapply_after_load[self.algorithm_group] = self.algorithm_name

        # if the registered save function is None, the original saving function remains
        if self.save_fn is not None and self.save_fn != SAVE_FUNCTIONS.reapply:
            smash_config.save_fns.append(self.save_fn.name)

        prefix = self.algorithm_name + "_"
        wrapped_config = SmashConfigPrefixWrapper(smash_config, prefix)
        return self._apply(model, wrapped_config)

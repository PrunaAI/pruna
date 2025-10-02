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

import copy
import itertools
from typing import Any, Union

from ConfigSpace import (
    CategoricalHyperparameter,
    ConfigurationSpace,
    EqualsCondition,
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
    OrConjunction,
)
from ConfigSpace.hyperparameters.hyperparameter import Hyperparameter

from pruna.config.hyperparameters import Boolean


class IsTrueCondition(EqualsCondition):
    """
    Represents a condition that checks if a hyperparameter is set to True.

    Parameters
    ----------
    child : Hyperparameter
        The child hyperparameter.
    parent : Hyperparameter
        The parent hyperparameter.
    """

    def __init__(self, child: Hyperparameter, parent: Hyperparameter) -> None:
        super().__init__(child, parent, True)

    def __new__(cls, child: Hyperparameter, parent: Hyperparameter) -> EqualsCondition:  # type: ignore
        """Create a new boolean condition."""
        return EqualsCondition(child, parent, True)


class SmashConfigurationSpace(ConfigurationSpace):
    """
    Wraps the ConfigSpace configuration space object to create the space of all smash configurations.

    Parameters
    ----------
    *args : Any
        Additional arguments for the ConfigurationSpace constructor.
    **kwargs : Any
        Additional keyword arguments for the ConfigurationSpace constructor.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.algorithm_buffer: dict[str, list[Union[str, None]]] = dict()
        self.argument_buffer: dict[str, tuple[list[Hyperparameter], str]] = dict()
        self.is_compiled: bool = False
        self.allowed_combinations: dict[str, dict[str, list[str]]] = dict()
        self.model_requirements: dict[str, dict[str, bool]] = dict()
        self.algorithms = []

    def register_algorithm(self, algorithm_name: str, hyperparameters: list) -> None:
        """
        Register algorithm by name.

        Parameters
        ----------
        algorithm_group : str
            The name of the configuration group (e.g., 'quantizer', 'pruner').
        name : str
            The name of the algorithm.
        """
        parent = Boolean(algorithm_name)
        self.algorithms.append(algorithm_name)
        self.add(parent)
        # Wrap hyperparameter names with config group and algorithm name
        for hp in hyperparameters:
            hp.name = f"{algorithm_name}_{hp.name}"
            self.add(hp)

            # Store condition s.t. hyperparameter is active if algorithm is active (is True)
            self.add(EqualsCondition(hp, parent, True))

    def get_all_algorithms(self) -> list[str]:
        """
        Get all algorithms.
        """
        return list(set(self.algorithms))


SMASH_SPACE = SmashConfigurationSpace(name="smash_config", seed=1234)

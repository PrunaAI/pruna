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

QUANTIZER = "quantizer"
PRUNER = "pruner"
COMPILER = "compiler"
CACHER = "cacher"
BATCHER = "batcher"
FACTORIZER = "factorizer"

# this ordering determins the order of smashing, modify carefully
ALGORITHM_GROUPS = [FACTORIZER, PRUNER, QUANTIZER, CACHER, COMPILER, BATCHER]


class Boolean(CategoricalHyperparameter):
    """
    Represents a boolean hyperparameter with choices True and False.

    Parameters
    ----------
    name : str
        The name of the hyperparameter.
    default : bool
        The default value of the hyperparameter.
    meta : Any
        The metadata for the hyperparameter.
    """

    def __init__(self, name: str, default: bool = False, meta: Any = dict()) -> None:
        super().__init__(name, choices=[True, False], default_value=default, meta=meta)

    def __new__(cls, name: str, default: bool = False, meta: Any = None) -> CategoricalHyperparameter:  # type: ignore
        """Create a new boolean hyperparameter."""
        return CategoricalHyperparameter(name, choices=[True, False], default_value=default, meta=meta)


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

    def register_algorithm(self, algorithm_group: str, name: str) -> None:
        """
        Register algorithm by name.

        Parameters
        ----------
        algorithm_group : str
            The name of the configuration group (e.g., 'quantizer', 'pruner').
        name : str
            The name of the algorithm.
        """
        if algorithm_group not in self.algorithm_buffer:
            self.algorithm_buffer[algorithm_group] = [None]
        self.algorithm_buffer[algorithm_group].append(name)

    def register_algorithm_arguments(self, name: str, hyperparameters: list, algorithm_group: str) -> None:
        """
        Register arguments conditional on active algorithm.

        Parameters
        ----------
        name : str
            The name of the algorithm.
        hyperparameters : list
            The hyperparameters to register.
        algorithm_group : str
            The configuration group.
        """
        self.argument_buffer[name] = (hyperparameters, algorithm_group)

    def register_allowed_combinations(self, name: str, combinations: dict) -> None:
        """
        Register allowed combinations for a algorithm.

        Parameters
        ----------
        name : str
            The name of the algorithm.
        combinations : dict
            The allowed combinations for the algorithm.
        """
        self.allowed_combinations[name] = combinations

    def gather_algorithm_buffer(self) -> None:
        """
        Gather the algorithm buffer by setting up hyperparameters, conditions, and constraints.

        This algorithm processes the configuration space in three steps:
        1. Adding group hyperparameters and their conditions
        2. Combining and adding all parameter conditions
        3. Registering forbidden algorithm combinations
        """
        # if the space is already compiled, we can not re-add the same hyperparameters
        if not self.is_compiled:
            conditions_dict = self._setup_hyperparameters()
            self._add_conditions(conditions_dict)
            self._setup_forbidden_combinations()

            self.is_compiled = True

    def _setup_hyperparameters(self) -> dict:
        """
        Set up all hyperparameters and register their hyperparameters as conditionally active hyperparameters.

        Given a algorithm and its hyperparameters, we:
        1. Add the algorithm as a hyperparameter into the configuration space
        2. Add the hyperparameters belonging to this algorithm
        3. Add the conditions that the hyperparameter is conditionallyactive if the algorithm is active

        Returns
        -------
        dict
            Dictionary mapping parameter names to their conditions.
        """
        conditions_dict: dict[str, list[EqualsCondition]] = {}

        # Iterate over compiler, quantizer, pruner, and factorizers
        for group in ALGORITHM_GROUPS:
            # eliminate duplicate entries coming from bidirectional compatibility specification
            self.algorithm_buffer[group] = list(set(self.algorithm_buffer[group]))
            # introduce parent hyperparameter, e.g. compiler=["stable_fast", "deepcache", "torch_compile", ...]
            parent = CategoricalHyperparameter(group, choices=self.algorithm_buffer[group], default_value=None)
            self.add(parent)

            # Process each compression algorithm in the group
            for compression_algorithm in self.algorithm_buffer[group]:
                # The "None" option will not have any hyperparameters
                if compression_algorithm is None:
                    continue

                # Iterate over all algorithms to add hyperparameters conditionally also on combinations
                hyperparameters, _ = self.argument_buffer[compression_algorithm]

                # Wrap hyperparameter names with config group and algorithm name
                for hp in hyperparameters:
                    # Create and add hyperparameter copy
                    hp_copy = copy.deepcopy(hp)
                    hp_copy.name = f"{compression_algorithm}_{hp_copy.name}"

                    # hyperparameter might have already been added in case of combinations
                    if hp_copy not in self.values():
                        self.add(hp_copy)

                    # Store condition s.t. hyperparameter is active if algorithm is active
                    new_condition = EqualsCondition(hp_copy, parent, compression_algorithm)
                    # for now, only collect conditions as we will aggregate them with OrConjunction later
                    conditions_dict.setdefault(hp_copy.name, []).append(new_condition)

        return conditions_dict

    def _add_conditions(self, conditions_dict: dict) -> None:
        """
        Add all collected conditions to the configuration space.

        Parameters
        ----------
        conditions_dict : dict
            Dictionary mapping parameter names to their conditions.
        """
        # Add all collected conditions to the configuration space
        for conditions in conditions_dict.values():
            if len(conditions) == 1:
                self.add(conditions[0])
            else:
                self.add(OrConjunction(*conditions))

    def _setup_forbidden_combinations(self) -> None:
        """Set up forbidden combinations between different configuration groups."""
        # Iterate over all combinations of configuration groups to capture cross-group forbidden combinations
        # This will specify how e.g. compiler and quantizer can or can not be used together
        for algorithm_group_1, algorithm_group_2 in itertools.combinations(ALGORITHM_GROUPS, 2):
            for algorithm_1 in self.algorithm_buffer[algorithm_group_1]:
                for algorithm_2 in self.algorithm_buffer[algorithm_group_2]:
                    if algorithm_1 is None or algorithm_2 is None:
                        continue

                    # Check if neither algorithm specifies the other as compatible
                    is_compatible_1_to_2 = (
                        algorithm_group_1 in self.allowed_combinations.get(algorithm_2, {})
                        and algorithm_1 in self.allowed_combinations[algorithm_2][algorithm_group_1]
                    )

                    is_compatible_2_to_1 = (
                        algorithm_group_2 in self.allowed_combinations.get(algorithm_1, {})
                        and algorithm_2 in self.allowed_combinations[algorithm_1][algorithm_group_2]
                    )

                    if not is_compatible_1_to_2 and not is_compatible_2_to_1:
                        self.add(
                            ForbiddenAndConjunction(
                                ForbiddenEqualsClause(self[algorithm_group_1], algorithm_1),
                                ForbiddenEqualsClause(self[algorithm_group_2], algorithm_2),
                            )
                        )


SMASH_SPACE = SmashConfigurationSpace(name="smash_config", seed=1234)

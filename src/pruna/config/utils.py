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

from typing import Any, cast

from pruna.algorithms import AlgorithmRegistry
from pruna.config.smash_config import SmashConfig, SmashConfigPrefixWrapper
from pruna.config.target_modules import TARGET_MODULES_TYPE, expand_list_of_targeted_paths


def is_empty_config(config: SmashConfig) -> bool:
    """
    Check if the SmashConfig is empty.

    Parameters
    ----------
    config : SmashConfig
        The SmashConfig to check.

    Returns
    -------
    bool
        True if the SmashConfig is empty, False otherwise.
    """
    empty_config = SmashConfig()
    return config == empty_config


def get_target_modules(model: Any, smash_config: SmashConfig, algorithm: str) -> TARGET_MODULES_TYPE:
    """
    Get the target modules for an algorithm.

    Parameters
    ----------
    model : Any
        The model to get the target modules from.
    smash_config : SmashConfig
        The SmashConfig object containing the algorithm configuration.
    algorithm : str
        The name of the algorithm to get the target modules for.

    Returns
    -------
    TARGET_MODULES_TYPE
        The target modules for the algorithm.

    Raises
    ------
    AttributeError
        If the algorithm does not have a target modules hyperparameter.
    """
    target_modules_hyperparameter = f"{algorithm}_target_modules"
    if target_modules_hyperparameter not in smash_config:
        raise AttributeError(f"Algorithm {algorithm} does not have a target modules hyperparameter.")

    target_modules: None | TARGET_MODULES_TYPE = smash_config[target_modules_hyperparameter]
    if target_modules is None:  # target modules exists but is set to default value
        wrapped_config = SmashConfigPrefixWrapper(smash_config, f"{algorithm}_")
        defaults = AlgorithmRegistry[algorithm].get_model_dependent_hyperparameter_defaults(model, wrapped_config)
        target_modules = cast(TARGET_MODULES_TYPE, defaults["target_modules"])
    return target_modules


def are_disjoint_target_modules(model: Any, smash_config: SmashConfig, alg_a: str, alg_b: str) -> tuple[bool, list[str]]:
    """
    Check if two algorithms target disjoint sets of the model's submodules.

    Parameters
    ----------
    model : Any
        The model to check the target modules for.
    smash_config : SmashConfig
        The SmashConfig object containing algorithm configurations.
    alg_a : str
        The first algorithm name.
    alg_b : str
        The second algorithm name.

    Returns
    -------
    are_disjoint : bool
        Whether the target modules of the two algorithms are disjoint.
    overlap : list[str]
        The paths of the submodules targeted by both algorithms. Empty if disjoint.
    """
    target_modules_a = get_target_modules(model, smash_config, alg_a)
    target_modules_b = get_target_modules(model, smash_config, alg_b)

    target_modules_a_expanded = expand_list_of_targeted_paths(target_modules_a, model)
    target_modules_b_expanded = expand_list_of_targeted_paths(target_modules_b, model)
    overlap = list(set(target_modules_a_expanded) & set(target_modules_b_expanded))
    are_disjoint = not overlap
    return are_disjoint, overlap

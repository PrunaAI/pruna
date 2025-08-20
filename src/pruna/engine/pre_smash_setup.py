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

from typing import Any

from pruna import SmashConfig
from pruna.algorithms import PRUNA_ALGORITHMS
from pruna.config.compatibility_checks import check_algorithm_availability


def pre_smash_setup(model: Any, smash_config: SmashConfig, algorithm_dict: dict[str, Any] = PRUNA_ALGORITHMS) -> None:
    """
    Perform any necessary setup steps before the smashing process begins.

    Parameters
    ----------
    model : Any
        The model to apply the setup to.
    smash_config : SmashConfig
        The SmashConfig object containing the algorithm configuration.
    algorithm_dict : dict[str, Any], optional
        Dictionary mapping algorithm groups to algorithm instances. Defaults to PRUNA_ALGORITHMS.
    """
    # algorithm groups are subject to change, make sure we have the latest version
    from pruna.config.smash_space import ALGORITHM_GROUPS

    # iterate through compiler, quantizer, ...
    for current_group in ALGORITHM_GROUPS:
        algorithm = smash_config[current_group]
        if algorithm is not None:
            check_algorithm_availability(algorithm, current_group, algorithm_dict)
            algorithm_dict[current_group][algorithm].pre_smash_setup(model, smash_config)

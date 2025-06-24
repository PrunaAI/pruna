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

from pruna import PrunaModel, SmashConfig
from pruna.algorithms import PRUNA_ALGORITHMS
from pruna.config.compatibility_checks import (
    check_algorithm_availability,
    check_model_compatibility,
    ensure_device_consistency,
)
from pruna.config.smash_space import ALGORITHM_GROUPS
from pruna.logging.logger import PrunaLoggerContext, pruna_logger
from pruna.telemetry import track_usage


@track_usage
def smash(
    model: Any,
    smash_config: SmashConfig,
    verbose: bool = False,
    experimental: bool = False,
) -> PrunaModel:
    """
    Smash an arbitrary model for inference.

    Parameters
    ----------
    model : Any
        Base model to be smashed.
    smash_config : SmashConfig
        Configuration settings for quantization, and compilation.
    verbose : bool
        Whether to print the progress of the smashing process.
    experimental : bool
        Whether to use experimental algorithms, e.g. to avoid checking model compatibility.
        This can lead to undefined behavior or difficult-to-debug errors.

    Returns
    -------
    PrunaModel
        Smashed model wrapped in a `PrunaModel` object.
    """
    with PrunaLoggerContext(verbose=verbose):
        # check the device consistency of the model and the smash config
        ensure_device_consistency(model, smash_config)

        # check if the model type is compatible with the given configuration
        if not experimental:
            check_model_compatibility(model, smash_config)

        # iterate through all algorithms groups in a predefined order
        for algorithm_group in ALGORITHM_GROUPS:
            current_algorithm = smash_config[algorithm_group]

            if current_algorithm is not None:
                check_algorithm_availability(current_algorithm, algorithm_group, PRUNA_ALGORITHMS)
                # apply the active algorithm to the model
                pruna_logger.info(f"Starting {algorithm_group} {current_algorithm}...")
                algorithm_instance = PRUNA_ALGORITHMS[algorithm_group][current_algorithm]
                model = algorithm_instance.apply(model, smash_config=smash_config)
                pruna_logger.info(f"{algorithm_group} {current_algorithm} was applied successfully.")

        # wrap the model in a PrunaModel object before returning
        smashed_model = PrunaModel(model, smash_config=smash_config)

    return smashed_model

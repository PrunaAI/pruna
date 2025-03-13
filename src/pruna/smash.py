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

from typing import Any

from pruna import PrunaModel, SmashConfig
from pruna.algorithms import PRUNA_ALGORITHMS
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


def check_model_compatibility(
    model: Any,
    smash_config: SmashConfig,
    algorithm_dict: dict[str, Any] = PRUNA_ALGORITHMS,
) -> None:
    """
    Check if the model is compatible with the given configuration.

    Parameters
    ----------
    model : Any
        The model to check for compatibility with the SmashConfig.
    smash_config : SmashConfig
        The SmashConfig to check the model against.
    algorithm_dict : dict[str, Any]
        The algorithm dictionary to hold all algorithm instances.
    """
    # algorithm groups are subject to change, make sure we have the latest version
    from pruna.config.smash_space import ALGORITHM_GROUPS

    # iterate through compiler, quantizer, ...
    for current_group in ALGORITHM_GROUPS:
        algorithm = smash_config[current_group]
        if algorithm is not None:
            check_algorithm_availability(algorithm, current_group, algorithm_dict)

            # check for model-algorithm compatibility with the model_check_fn
            if not algorithm_dict[current_group][algorithm].model_check_fn(model):
                raise ValueError(
                    f"Model is not compatible with {algorithm_dict[current_group][algorithm].algorithm_name}"
                )


def check_algorithm_availability(algorithm: str, algorithm_group: str, algorithm_dict: dict[str, Any]) -> None:
    """
    Check if the algorithm is available in the algorithm dictionary.

    Parameters
    ----------
    algorithm : str
        The algorithm to check for availability.
    algorithm_group : str
        The algorithm group to check for availability.
    algorithm_dict : dict[str, Any]
        The algorithm dictionary to check for availability.

    Raises
    ------
    ValueError
        If the algorithm is not available in the algorithm dictionary.
    """
    if algorithm_group not in algorithm_dict:
        raise RuntimeError(f"Algorithm group {algorithm_group} is unavailable with pruna.smash")
    if algorithm not in algorithm_dict[algorithm_group]:
        raise RuntimeError(f"Algorithm {algorithm} is unavailable with pruna.smash")

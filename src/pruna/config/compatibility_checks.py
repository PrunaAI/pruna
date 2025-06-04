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
from pruna.config.smash_space import SMASH_SPACE
from pruna.engine.utils import get_device, get_device_map, move_to_device
from pruna.logging.logger import pruna_logger


def ensure_device_consistency(model, smash_config):
    """
    Ensure consistency between the device state of the model and the smash config.

    Parameters
    ----------
    model : Any
        The model to check for device consistency.
    smash_config : SmashConfig
        The smash config to check for device consistency.
    """
    model_device = get_device(model)

    # model and smash config devices match
    if model_device == smash_config.device:
        pruna_logger.debug("Device consistency check passed.")
        # in case of accelerate, we need to store the device map
        if model_device == "accelerate":
            pruna_logger.debug("Device consistency check passed.")
            hf_device_map = get_device_map(model)
            if not all(isinstance(v, int) for v in hf_device_map.values()):
                raise ValueError("Device map indicates CPU offloading, this is not supported at this time.")
            else:
                smash_config.device_map = hf_device_map

    elif smash_config.device in ["cpu", "cuda", "mps"] and model_device in ["cpu", "cuda", "mps"]:
        pruna_logger.warning(
            (
                f"Model and SmashConfig have different devices. Model: {model_device}, "
                f"SmashConfig: {smash_config.device}. Casting model to {smash_config.device}."
                f"If this is not desired, please use SmashConfig(device='{model_device}')."
            )
        )
        move_to_device(model, smash_config.device)

    elif smash_config.device == "accelerate" or model_device == "accelerate":
        pruna_logger.warning(
            (
                f"Model and SmashConfig have different devices. Model: {model_device}, "
                f"SmashConfig: {smash_config.device}. Updating SmashConfig to device='{model_device}'."
            )
        )
        smash_config.device = model_device
    else:
        raise ValueError(f"Invalid device: {smash_config.device}")


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
            # test if all required packages are installed, if not this will raise an ImportError
            algorithm_dict[current_group][algorithm].import_algorithm_packages()
            check_argument_compatibility(smash_config, algorithm)
            # check for model-algorithm compatibility with the model_check_fn
            if not algorithm_dict[current_group][algorithm].model_check_fn(model):
                raise ValueError(
                    f"Model is not compatible with {algorithm_dict[current_group][algorithm].algorithm_name}"
                )
            if get_device(model) not in algorithm_dict[current_group][algorithm].runs_on:
                raise ValueError(
                    f"{algorithm} is not compatible with device {get_device(model)}, "
                    f"compatible devices are {algorithm_dict[current_group][algorithm].runs_on}"
                )


def check_argument_compatibility(smash_config: SmashConfig, algorithm_name: str) -> None:
    """
    Check if the SmashConfig has the required arguments (tokenizer, processor, dataset) for an algorithm.

    Parameters
    ----------
    smash_config : SmashConfig
        The SmashConfig to check the argument consistency with.
    algorithm_name : str
        The algorithm name that is about to be activated.
    """
    algorithm_requirements = SMASH_SPACE.model_requirements[algorithm_name]
    if algorithm_requirements["tokenizer_required"] and smash_config.tokenizer is None:
        raise ValueError(f"{algorithm_name} requires a tokenizer. Please provide it with smash_config.add_tokenizer().")
    if algorithm_requirements["processor_required"] and smash_config.processor is None:
        raise ValueError(f"{algorithm_name} requires a processor. Please provide it with smash_config.add_processor().")
    if algorithm_requirements["dataset_required"] and smash_config.data is None:
        raise ValueError(f"{algorithm_name} requires a dataset. Please provide it with smash_config.add_data().")
    if smash_config._target_module is not None:
        raise ValueError("Target module is only available in experimental mode. Please set experimental=True.")


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

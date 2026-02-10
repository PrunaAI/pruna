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
from pruna.algorithms import AlgorithmRegistry
from pruna.config.pre_smash_routines import (
    check_algorithm_cross_compatibility,
    check_algorithm_packages_availability,
    check_argument_compatibility,
    check_model_compatibility,
    determine_algorithm_order,
    ensure_device_consistency,
    execute_algorithm_pre_smash_hooks,
)
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
        # check if the algorithm packages are available for the given configuration
        check_algorithm_packages_availability(smash_config)

        if not experimental:
            # check if the algorithms are compatible with each other
            check_algorithm_cross_compatibility(model, smash_config)
            # check if the model type is compatible with the given configuration
            check_model_compatibility(model, smash_config)
            # check if the SmashConfig has the required arguments (tokenizer, processor, dataset) for an algorithm
            check_argument_compatibility(smash_config)

        # determine algorithm order based on requirements of individual algorithms
        algorithm_order = determine_algorithm_order(model, smash_config, experimental=experimental)
        # perform any necessary setup steps before the smashing process begins
        execute_algorithm_pre_smash_hooks(model, smash_config, algorithm_order)

        # ring_attn needs a process group; if we're not already under torchrun/torch.distributed,
        # spawn our local multi-proc server and return early with the wrapped model.
        if "ring_attn" in algorithm_order:
            from pruna.algorithms.ring_attn.utils.server_utils import start_distributed_server_if_needed

            server = start_distributed_server_if_needed(model, smash_config)
            if server is not None:
                return PrunaModel(server, smash_config=smash_config)

        # iterate through all algorithms groups in a predefined order
        for algorithm in algorithm_order:
            pruna_logger.info(f"Starting {algorithm}...")
            model = AlgorithmRegistry[algorithm].apply(model, smash_config=smash_config)
            pruna_logger.info(f"{algorithm} was applied successfully.")

        # wrap the model in a PrunaModel object before returning
        smashed_model = PrunaModel(model, smash_config=smash_config)

    return smashed_model

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

import itertools
from typing import Any

import networkx as nx
from networkx import DiGraph

from pruna import SmashConfig
from pruna.algorithms import AlgorithmRegistry
from pruna.engine.utils import get_device, get_device_map, get_device_type, move_to_device, split_device
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
    _device_options = ["cpu", "cuda", "mps"]
    model_device = get_device(model)

    # to handle the device cases like "cuda:0 and cuda, cuda:1"
    model_device_type, model_device_index = split_device(model_device)
    smash_config_device_type, smash_config_device_index = split_device(smash_config.device)

    # model and smash config devices match
    if (model_device_type == smash_config_device_type) and (model_device_index == smash_config_device_index):
        pruna_logger.debug("Device consistency check passed.")
        # in case of accelerate, we need to store the device map
        if model_device_type == "accelerate":
            pruna_logger.debug("Device consistency check passed.")
            hf_device_map = get_device_map(model)
            if not all(isinstance(v, int) for v in hf_device_map.values()):
                raise ValueError("Device map indicates CPU offloading, this is not supported at this time.")
            else:
                smash_config.device_map = hf_device_map
    # Check if the device or device index (e.g., 'cuda:0', 'cpu:1', 'mps:0') matches any of the valid device options
    elif smash_config_device_type in _device_options and model_device_type in _device_options:
        pruna_logger.warning(
            (
                f"Model and SmashConfig have different devices. Model: {model_device}, "
                f"SmashConfig: {smash_config.device}. Casting model to {smash_config.device}."
                f"If this is not desired, please use SmashConfig(device='{model_device}')."
            )
        )
        move_to_device(model, smash_config.device)

    elif (smash_config_device_type == "accelerate") or (model_device_type == "accelerate"):
        pruna_logger.warning(
            (
                f"Model and SmashConfig have different devices. Model: {model_device}, "
                f"SmashConfig: {smash_config.device}. Updating SmashConfig to device='{model_device}'."
            )
        )
        smash_config.device = model_device
    else:
        raise ValueError(f"Invalid device: {smash_config.device}")


def check_model_compatibility(model: Any, smash_config: SmashConfig) -> None:
    """
    Check if the model and its device state is compatible with the given configuration.

    Parameters
    ----------
    model : Any
        The model to check for compatibility with the SmashConfig.
    smash_config : SmashConfig
        The SmashConfig to check the model against.
    """
    for algorithm in smash_config.get_active_algorithms():
        algorithm_class = AlgorithmRegistry[algorithm]
        if not algorithm_class.model_check_fn(model):
            raise ValueError(f"Model is not compatible with {algorithm_class.algorithm_name}")
        if get_device_type(model) not in algorithm_dict[current_group][algorithm].runs_on:
            raise ValueError(
                f"{algorithm} is not compatible with model device {get_device(model)}, "
                f"compatible devices are {algorithm_class.runs_on}"
            )


def check_algorithm_packages_availability(smash_config: SmashConfig) -> None:
    """
    Check if the algorithm packages are available for the given configuration.

    Parameters
    ----------
    smash_config : SmashConfig
        The SmashConfig object containing the algorithm configuration.
    """
    for algorithm in smash_config.get_active_algorithms():
        algorithm_class = AlgorithmRegistry[algorithm]
        algorithm_class.import_algorithm_packages()


def check_argument_compatibility(smash_config: SmashConfig) -> None:
    """
    Check if the SmashConfig has the required arguments (tokenizer, processor, dataset) for all active algorithms.

    Parameters
    ----------
    smash_config : SmashConfig
        The SmashConfig to check the argument consistency with.
    """
    for algorithm in smash_config.get_active_algorithms():
        algorithm_class = AlgorithmRegistry[algorithm]

        if algorithm_class.tokenizer_required and smash_config.tokenizer is None:
            raise ValueError(f"{algorithm} requires a tokenizer. Please provide it with smash_config.add_tokenizer().")
        if algorithm_class.processor_required and smash_config.processor is None:
            raise ValueError(f"{algorithm} requires a processor. Please provide it with smash_config.add_processor().")
        if algorithm_class.dataset_required and smash_config.data is None:
            raise ValueError(f"{algorithm} requires a dataset. Please provide it with smash_config.add_data().")
        if smash_config._target_module is not None:
            raise ValueError("Target module is only available in experimental mode. Please set experimental=True.")


def check_algorithm_availability(smash_config: SmashConfig) -> None:
    """
    Check if the algorithm is available in the algorithm dictionary.

    Parameters
    ----------
    smash_config : SmashConfig
        The SmashConfig object containing the algorithm configuration.
    """
    for algorithm in smash_config.get_active_algorithms():
        algorithm_class = AlgorithmRegistry[algorithm]
        if "pruna_pro" in algorithm_class.__module__:
            raise RuntimeError(f"Algorithm {algorithm} is unavailable with pruna.smash")


def execute_algorithm_pre_smash_hooks(model: Any, smash_config: SmashConfig, algorithm_order: list[str]) -> None:
    """
    Loop through all active algorithms and calls the pre_smash_hook method for each algorithm.

    Parameters
    ----------
    model : Any
        The model to apply the setup to.
    smash_config : SmashConfig
        The SmashConfig object containing the algorithm configuration.
    algorithm_order : list[str]
        The order of the active algorithms to be applied.
    """
    active_algorithms = algorithm_order
    for algorithm in active_algorithms:
        algorithm_class = AlgorithmRegistry[algorithm]
        algorithm_class.pre_smash_hook(model, smash_config)


def check_algorithm_cross_compatibility(smash_config: SmashConfig) -> None:
    """
    Check if the active algorithms are compatible with each other.

    Parameters
    ----------
    smash_config : SmashConfig
        The SmashConfig object containing the algorithm configuration.
    """
    active_algorithms = smash_config.get_active_algorithms()
    algorithm_pairs = list(itertools.permutations(active_algorithms, 2))

    for alg_a, alg_b in algorithm_pairs:
        if alg_a in AlgorithmRegistry[alg_b].get_incompatible_algorithms():
            raise ValueError(f"Algorithm {alg_a} is incompatible with {alg_b}")


def determine_algorithm_order(smash_config: SmashConfig) -> list[str]:
    """
    Determine the order of the active algorithms based on their ordering requirements.

    Parameters
    ----------
    smash_config : SmashConfig
        The SmashConfig object containing the algorithm configuration.

    Returns
    -------
    list[str]
        The order of the active algorithms to be applied.
    """
    if smash_config._algorithm_order is not None:
        return smash_config._algorithm_order
    graph = construct_algorithm_directed_graph(smash_config)
    remove_reciprocals(graph)
    try:
        order = list(nx.topological_sort(graph))
        pruna_logger.info(f"Determined algorithm order: {', '.join(order)}")
        return order
    except nx.NetworkXUnfeasible:
        raise ValueError("Cycle detected in the algorithm order, the current algorithm configuration is not possible.")


def construct_algorithm_directed_graph(smash_config: SmashConfig) -> nx.DiGraph:
    """
    Construct a directed graph of the algorithms to be applied based on their ordering requirements.

    Parameters
    ----------
    smash_config : SmashConfig
        The SmashConfig object containing the algorithm configuration.

    Returns
    -------
    nx.DiGraph
        The directed graph of the algorithms to be applied.
    """
    graph = DiGraph()
    active_algorithms = smash_config.get_active_algorithms()
    algorithm_pairs = list(itertools.permutations(active_algorithms, 2))

    for algorithm in active_algorithms:
        graph.add_node(algorithm)

    for alg_a, alg_b in algorithm_pairs:
        if alg_a in AlgorithmRegistry[alg_b].get_algorithms_to_run_before():
            graph.add_edge(alg_a, alg_b)
        if alg_a in AlgorithmRegistry[alg_b].get_algorithms_to_run_after():
            graph.add_edge(alg_b, alg_a)

    return graph


def remove_reciprocals(graph: nx.DiGraph):
    """
    Remove reciprocal (bidirectional) edges from a directed graph.

    Parameters
    ----------
    graph : nx.DiGraph
        The directed graph to remove reciprocal edges from.
    """
    # Undirected view collapses mutual edges into one
    reciprocal_pairs = {
        (u, v) for u, v in graph.to_undirected().edges() if graph.has_edge(u, v) and graph.has_edge(v, u)
    }

    graph.remove_edges_from((u, v) for u, v in reciprocal_pairs)
    graph.remove_edges_from((v, u) for u, v in reciprocal_pairs)

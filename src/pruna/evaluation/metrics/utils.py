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

from collections import defaultdict
from inspect import Signature, getmro, signature
from typing import Any, Callable, Dict, List, Tuple, Type, cast
from warnings import warn

import torch

from pruna.evaluation.metrics.metric_base import BaseMetric
from pruna.logging.logger import pruna_logger

SINGLE = "single"
PAIRWISE = "pairwise"
CALL_TYPES = (SINGLE, PAIRWISE)


def metric_data_processor(
    x: List[Any] | torch.Tensor, gt: List[Any] | torch.Tensor, outputs: Any, call_type: str
) -> List[Any]:
    """
    Arrange metric inputs based on the specified configuration call type.

    This function determines the order and selection of inputs to be passed to various metrics.

    The function supports different input arrangements through the 'call_type' configuration:
    - 'x_y': Uses input data (x) and model outputs
    - 'gt_y': Uses ground truth (gt) and model outputs
    - 'y_x': Uses model outputs and input data (x)
    - 'y_gt': Uses model outputs and ground truth (gt)
    - 'pairwise_gt_y': Uses cached base model outputs (gt) and smashed model outputs (y).
    - 'pairwise_y_gt': Uses smashed model outputs (y) and cached base model outputs (gt).
    The evaluation agent is expected to pass the cached base model outputs as gt.

    Parameters
    ----------
    x : Any
        The input data (e.g., input images, text prompts).
    gt : Any
        The ground truth data (e.g., correct labels, target images, cached model outputs).
    outputs : Any
        The model outputs or predictions.
    call_type : str
        The type of call to be made to the metric.

    Returns
    -------
    List[Any]
        A list containing the arranged inputs in the order specified by call_type.

    Raises
    ------
    ValueError
        If the specified call_type is not one of: 'x_y', 'gt_y', 'y_x', 'y_gt', 'pairwise'.

    Examples
    --------
    >>> call_type = "gt_y"
    >>> inputs = metric_data_processor(x_data, ground_truth, model_outputs, call_type)
    >>> # Returns [ground_truth, model_outputs]
    """
    if call_type == "x_y":
        return [x, outputs]
    elif call_type == "gt_y":
        return [gt, outputs]
    elif call_type == "y_x":
        return [outputs, x]
    elif call_type == "y_gt":
        return [outputs, gt]
    elif call_type == "pairwise_gt_y":
        return [gt, outputs]
    elif call_type == "pairwise_y_gt":
        return [outputs, gt]
    else:
        raise ValueError(f"Invalid call type: {call_type}")


def get_param_names_from_signature(sig: Signature) -> list[str]:
    """
    Extract the parameter names (excluding 'self') from a constructor signature.

    Parameters
    ----------
    sig : Signature
        The signature to extract the parameter names from.

    Returns
    -------
    List[str]
        A list of the parameter names.
    """
    return [
        p.name
        for p in sig.parameters.values()
        if p.name != "self" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]


def get_hyperparameters(instance: Any, reference_function: Callable[..., Any]) -> Dict[str, Any]:
    """
    Get hyperparameters from an instance.

    This is the most basic and self-contained case.

    Parameters
    ----------
    instance : Any
        The instance to get the hyperparameters from.
    reference_function : Callable[..., Any]
        The reference function to get the hyperparameters from.

    Returns
    -------
    Dict[str, Any]
        A dictionary of the hyperparameters.
    """
    sig = signature(reference_function)
    param_names = get_param_names_from_signature(sig)
    return {name: getattr(instance, name, None) for name in param_names}


def group_metrics_by_inheritance(list_of_instances: List[Any]) -> Tuple[Dict[Any, List[Any]], List[Any]]:
    """
    Split a list of metric instances based on their direct parent class and configuration.

    Specifically, the function:
    - Groups instances that share the same direct parent class and initialization hyperparameters.
    - Separately collects instances that directly inherit from BaseMetric.

    Parameters
    ----------
    list_of_instances : List[Any]
        A list of instances.

    Returns
    -------
    Tuple[Dict[Any, List[Any]], List[Any]]
        A tuple of a dictionary where the keys are the direct parents and the values are the direct children,
        and a list of instances that directly inherit from BaseMetric.
    """
    # Metrics with shared parents and configs are grouped together
    parents_to_children = defaultdict(list)
    # Metrics who directly inherit from BaseMetric should not be included
    children_of_base = []

    for instance in list_of_instances:
        mro = getmro(instance.__class__)
        parent = cast(Type, mro[1])
        if parent == BaseMetric:
            children_of_base.append(instance)
            continue
        # Only group metrics with shared inference hyper-parameters.
        config = frozenset(get_hyperparameters(instance, parent.__init__).items())
        key = (parent, config)
        parents_to_children[key].append(instance)
    return parents_to_children, children_of_base


def get_pairwise_pairing(call_type: str) -> str:
    """
    Get the pairwise pairing for a call type.

    Parameters
    ----------
    call_type : str
        The call type to get the pairing for.

    Returns
    -------
    str
        The pairwise pairing for the call type.
    """
    if call_type.startswith("y_"):
        return "pairwise_y_gt"
    else:
        return "pairwise_gt_y"


def get_single_pairing(call_type: str) -> str:
    """
    Get the single pairing for a call type.

    Parameters
    ----------
    call_type : str
        The call type to get the pairing for.

    Returns
    -------
    str
        The single pairing for the call type.
    """
    return call_type.removeprefix(PAIRWISE + "_")


def get_any_call_type_pairing(call_type: str) -> str:
    """
    Get the pairing for a call type.

    Parameters
    ----------
    call_type : str
        The call type to get the pairing for.

    Returns
    -------
    str
        The pairing for the call type.
    """
    if call_type.startswith(PAIRWISE):
        return get_single_pairing(call_type)
    else:
        return get_pairwise_pairing(call_type)


def get_call_type_for_pairwise_metric(call_type_requested: str, default_call_type: str) -> str:
    """
    Get the call type for a pairwise metric.

    Parameters
    ----------
    call_type_requested : str
        The call type to get the pairing for.
    default_call_type : str
        The default call type for the metric.

    Returns
    -------
    str
        The call type pairing for the metric.
    """
    if call_type_requested == PAIRWISE:
        return default_call_type
    elif call_type_requested == SINGLE:
        return get_single_pairing(default_call_type)
    else:
        if call_type_requested == default_call_type or call_type_requested == get_single_pairing(default_call_type):
            warn(
                f"Calling metric with its call type is deprecated and will be removed in 'v0.2.8' release. \n"
                f"Use {SINGLE} or {PAIRWISE} instead. \n"
                f"Using default call type {default_call_type}.",
                DeprecationWarning,
                stacklevel=2,
            )
            return default_call_type
        else:
            pruna_logger.error(f"Invalid call type: {call_type_requested}. Must be one of {CALL_TYPES}.")
            raise ValueError(f"Invalid call type: {call_type_requested}. Must be one of {CALL_TYPES}.")


def get_call_type_for_single_metric(call_type_requested: str, default_call_type: str) -> str:
    """
    Get the call type for a single metric.

    Parameters
    ----------
    call_type_requested : str
        The call type to get the pairing for.
    default_call_type : str
        The default call type for the metric.

    Returns
    -------
    str
        The call type for the metric.
    """
    if call_type_requested == PAIRWISE:
        return get_pairwise_pairing(default_call_type)
    elif call_type_requested == SINGLE:
        return default_call_type
    else:
        if call_type_requested == default_call_type or call_type_requested == get_pairwise_pairing(default_call_type):
            warn(
                f"Calling metric with its call type is deprecated and will be removed in 'v0.2.8' release. \n"
                f"Use {SINGLE} or {PAIRWISE} instead. \n"
                f"Using default call type {default_call_type}.",
                DeprecationWarning,
                stacklevel=2,
            )
            return default_call_type
        else:
            pruna_logger.error(f"Invalid call type: {call_type_requested}. Must be one of {CALL_TYPES}.")
            raise ValueError(f"Invalid call type: {call_type_requested}. Must be one of {CALL_TYPES}.")

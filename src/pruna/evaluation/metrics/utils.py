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

import torch

from pruna.evaluation.metrics.metric_base import BaseMetric


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


def get_direct_parents(list_of_instances: List[Any]) -> Tuple[Dict[Any, List[Any]], List[Any]]:
    """
    Get the direct parents of a list of instances.

    Returns a dictionary where the keys are the direct parents and the values are the direct children.
    Also returns a list of instances that directly inherit from BaseMetric.

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

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

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pruna.engine.pruna_model import PrunaModel
    from pruna.evaluation.task import Task
from collections import defaultdict
from inspect import Signature, getmro, signature
from typing import Callable, Dict, List, Tuple, Type

import torch

from pruna.data.utils import move_batch_to_device
from pruna.engine.utils import (
    device_to_string,
    find_bytes_free_per_gpu,
    get_device,
    get_device_map,
    set_to_best_available_device,
    split_device,
)
from pruna.evaluation.metrics.metric_base import BaseMetric
from pruna.logging.logger import pruna_logger

SINGLE = "single"
PAIRWISE = "pairwise"
CALL_TYPES = (SINGLE, PAIRWISE)


def metric_data_processor(
    x: List[Any] | torch.Tensor,
    gt: List[Any] | torch.Tensor,
    outputs: Any,
    call_type: str,
    device: str | torch.device | None = None,
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
    device : str | torch.device | None
        The device to move the data to.

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
    if device is not None:
        x = move_batch_to_device(x, device)
        gt = move_batch_to_device(gt, device)
        outputs = move_batch_to_device(outputs, device)

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
    elif call_type == "y":  # IQA metrics that have an internal dataset
        return [outputs]
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
    if call_type == "y":
        pruna_logger.error("IQA metrics cannot be used with pairwise call type")
        raise Exception
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
        pruna_logger.error(f"Invalid call type: {call_type_requested}. Must be one of {CALL_TYPES}.")
        raise ValueError(f"Invalid call type: {call_type_requested}. Must be one of {CALL_TYPES}.")


def ensure_device_consistency(model: PrunaModel, task: Task) -> None:
    """
    Ensure the model and the task agree on the device they will run on.

    Parameters
    ----------
    model : Any
        The model to check.
    task : Task
        The task to check.
    """
    # Preprocessing the devices
    model_device_raw = cast(str, get_device(model))
    model_device, idx_m = split_device(model_device_raw, strict=True)
    task_device, idx_t = split_device(device_to_string(task.device), strict=True)

    # Everything is fine scenario.
    if (model_device, idx_m) == (task_device, idx_t):
        pruna_logger.debug("Device consistency check passed.")
        if model_device == "accelerate":
            # in case of accelerate, we need to check loading the metrics didn't offload the model.
            _check_offload(model)
        return
    if model_device == task_device == "cuda":  # Cases like cuda:0 and cuda:1
        pruna_logger.info(
            f"Model on cuda:{idx_m}, task on cuda:{idx_t}. If undesired, call task(device='{model_device_raw}')."
        )
        return

    # If the user explicitly provided a device and it doesn't match the model's device,
    # raise an error: we assume they know what they're doing and want control.
    if not task.auto_device:
        raise ValueError(
            f"Model and task have different devices. Model: {model_device}, task: {task.device}. \n"
            f"If you want auto device casting, create the task without providing a device."
        )

    # Only auto-resolve device mismatches when no device was provided.
    # We take model's device as the default device for the task in this case.
    else:
        pruna_logger.warning(
            (
                f"Model and task have different devices. Model: {model_device}, "
                f"task: {task.device}. Updating task to device='{model_device}'."
            )
        )
        task.device = model_device_raw
        if model_device in ["cuda", "mps", "accelerate"]:
            if not task.low_memory:
                free_bytes = find_bytes_free_per_gpu() if model_device == "accelerate" else None
                # Return the best available device with them most free memory.
                task.stateful_metric_device = set_to_best_available_device("cuda", free_bytes)
            else:
                task.stateful_metric_device = "cpu"
            # We update the inference device for the metrics in the task.
            _update_metric_devices(task, inference_device=model_device_raw, stateful_device=task.stateful_metric_device)
            if model_device == "accelerate":
                _check_offload(model)  # We want to make sure to catch if the model is offloaded to CPU.
        elif model_device == "cpu":
            task.stateful_metric_device = "cpu"
            _update_metric_devices(task, inference_device=model_device_raw, stateful_device=task.stateful_metric_device)
        else:
            raise ValueError(
                f"Invalid model device: {model_device}. Must be one of {['cuda', 'mps', 'accelerate', 'cpu']}."
            )


def _check_offload(model: Any) -> None:
    """
    Check if the model is offloaded to CPU.

    Parameters
    ----------
    model : Any
        The model to check.
    """
    hf_device_map = get_device_map(model)
    if not all(isinstance(v, int) for v in hf_device_map.values()):
        raise ValueError(
            "Device map indicates CPU offloading; not supported at this time. \n"
            "Please initialize Task with `low_memory=True` to run stateful metrics on cpu."
        )


def _update_metric_devices(task: Task, inference_device: str, stateful_device: str) -> None:
    """
    Update the inference device for the metrics in the task.

    Parameters
    ----------
    task : Task
        The task to update.
    device : str
        The device to update the metrics to.
    """
    for metric in task.metrics:
        if isinstance(metric, BaseMetric):
            metric.device = inference_device
        else:
            metric.move_to_device(stateful_device)

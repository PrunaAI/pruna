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

import contextlib
import gc
import inspect
import json
import os
import warnings
from typing import Any

import torch
import torch.nn as nn

from pruna.logging.logger import pruna_logger


def safe_memory_cleanup() -> None:
    """Perform safe memory cleanup by collecting garbage and clearing CUDA cache."""
    gc.collect()
    torch.cuda.empty_cache()


def load_json_config(path: str, json_name: str) -> dict:
    """
    Load and parse a JSON configuration file.

    Parameters
    ----------
    path : str
        Directory path containing the JSON file.
    json_name : str
        Name of the JSON file to load.

    Returns
    -------
    dict
        Parsed JSON configuration as a dictionary.
    """
    with open(os.path.join(path, json_name), "r") as f:
        model_index = json.load(f)
    return model_index


def get_nn_modules(model: Any) -> dict[str | None, torch.nn.Module]:
    """
    Return a dictionary containing the model itself or its torch.nn.Module components.

    Modules are referenced by their attribute name in model. In the case where the model
    is a torch.nn.Module, it is returned with the key None.

    Parameters
    ----------
    model : Any
        The model whose nn.Module we want to get.

    Returns
    -------
    dict[str | None, torch.nn.Module]
        The dictionary containing the model (key None) itself or its torch.nn.Module
        referenced by their corresponding attribute name in model.
    """
    if isinstance(model, torch.nn.Module):
        return {None: model}
    else:
        return {
            module_name: module
            for module_name, module in inspect.getmembers(model)
            if isinstance(module, torch.nn.Module)
        }


def move_to_device(model: Any, device: str | torch.device, raise_error: bool = False) -> None:
    """
    Move the model to a specific device.

    Parameters
    ----------
    model : Any
        The model to move.
    device : str
        The device to move the model to.
    raise_error : bool
        Whether to raise an error when the device movement fails.
    """
    if hasattr(model, "device") and check_model_already_on_device(model, device):
        return
    if hasattr(model, "to"):
        try:
            model.to(device)
        except torch.cuda.OutOfMemoryError as e:
            # there is anyway no way to recover from this error
            # raise it here for better traceability
            raise e
        except (ValueError, RecursionError, RuntimeError, AttributeError) as e:
            if raise_error:
                raise ValueError(f"Could not move model to device: {str(e)}")
            else:
                pruna_logger.warning(f"Could not move model to device: {str(e)}")
    elif hasattr(model, "task") and getattr(model, "task") == "automatic-speech-recognition":
        model.model.to(device)
    else:
        if raise_error:
            raise ValueError("Model does not support device movement.")
        else:
            pruna_logger.warning("Model does not support device movement.")


def check_model_already_on_device(model: Any, device: str | torch.device) -> bool:
    """
    Check if the model is already on the device.

    Parameters
    ----------
    model : Any
        The model to check.
    device : str | torch.device
        The device to check.

    Returns
    -------
    bool
        True if the model is already on the device, False otherwise.
    """
    return model.device == device or model.device == torch.device(device)


def set_to_eval(model: Any) -> None:
    """
    Set the model to evaluation mode.

    Parameters
    ----------
    model : Any
        The model to set to evaluation mode.
    """
    if hasattr(model, "eval"):
        try:
            model.eval()
        except RecursionError:
            recursive_set_to_eval(model)
    else:
        nn_modules = get_nn_modules(model)
        for _, module in nn_modules.items():
            if hasattr(module, "eval"):
                module.eval()


def recursive_set_to_eval(model: Any, visited: set | None = None) -> None:
    """
    For the case where the model is referencing itself.

    This is a recursive function that will set the model to evaluation mode.
    It is used to avoid the RecursionError that occurs when the model is referencing itself.

    Parameters
    ----------
    model : Any
        The model to set to evaluation mode.
    visited : set
        A set of visited models to avoid infinite recursion.
    """
    if visited is None:
        visited = set()

    model_id = id(model)
    if model_id in visited:
        return
    visited.add(model_id)

    with contextlib.suppress(Exception):
        model.eval()

    if hasattr(model, "_modules") and isinstance(model._modules, dict):
        for child in model._modules.values():
            if isinstance(child, torch.nn.Module):
                recursive_set_to_eval(child, visited)


def set_to_train(model: Any) -> None:
    """
    Set the model to training mode.

    Parameters
    ----------
    model : Any
        The model to set to training mode.
    """
    if hasattr(model, "train"):
        model.train()
    else:
        # Here, similar to the eval case we can iterate over the nn_modules.
        # Since after compression most of the models are inference only, the iteration could lead to unexpected behavior. # noqa: E501
        # This should be investigated in the future.
        pruna_logger.warning("Model does not support training mode.")


def determine_dtype(pipeline: Any) -> torch.dtype:
    """
    Determine the dtype of a given diffusers pipeline or model.

    Parameters
    ----------
    pipeline : Any
        The pipeline or model to determine the dtype of.

    Returns
    -------
    torch.dtype
        The dtype of the model.
    """
    if hasattr(pipeline, "torch_dtype"):
        return pipeline.torch_dtype

    if hasattr(pipeline, "dtype"):
        return pipeline.dtype

    found_dtypes = set()
    for m in pipeline.components.values():
        if isinstance(m, nn.Module):
            try:
                p = next(m.parameters())
                found_dtypes.add(p.dtype)
            except StopIteration:
                pass

    if len(found_dtypes) == 1:
        return list(found_dtypes)[0]

    pruna_logger.warning("Could not determine dtype of model, defaulting to torch.float32.")
    return torch.float32


def check_device_compatibility(device: str | torch.device | None) -> str:
    """Validate device availability and return best available device.

    Parameters
    ----------
    device : str | torch.device | None
        Device to validate ('cuda', 'mps', 'cpu' etc)

    Returns
    -------
    str
        Best available device name
    """
    if isinstance(device, torch.device):
        device = str(device)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        pruna_logger.info(f"No device specified. Using best available device: '{device}'")
        return device

    if device == "cpu":
        return "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        warnings.warn("'cuda' requested but not available. Falling back to 'cpu'")
        return "cpu"

    if device == "mps" and not torch.backends.mps.is_available():
        warnings.warn("'mps' requested but not available. Falling back to 'cpu'")
        return "cpu"

    return device

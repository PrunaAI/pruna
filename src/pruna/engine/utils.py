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
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from accelerate import dispatch_model
from accelerate.hooks import remove_hook_from_module
from diffusers.models.modeling_utils import ModelMixin
from transformers import Pipeline

from pruna.logging.logger import pruna_logger


def safe_memory_cleanup() -> None:
    """Perform safe memory cleanup by collecting garbage and clearing CUDA cache."""
    gc.collect()
    torch.cuda.empty_cache()


def load_json_config(path: str | Path, json_name: str) -> dict:
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


def safe_is_instance(model: Any, instance_type: type) -> bool:
    """
    Safely check if the model is an instance of the given type.

    Parameters
    ----------
    model : Any
        The model to check.
    instance_type : type
        The type to check against.

    Returns
    -------
    bool
        True if the model is an instance of the given type, False otherwise.
    """
    if hasattr(model, "is_instance"):
        return model.is_instance(instance_type)
    return isinstance(model, instance_type)


def move_to_device(model: Any, device: str, raise_error: bool = False, device_map: dict[str, str] | None = None) -> None:
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
    device_map : dict[str, str] | None
        The device map to use if the target device is "accelerate".
    """
    if isinstance(model, Pipeline):
        move_to_device(model.model, device, raise_error, device_map)
        # this is a workaround for a flaw in the transformers pipeline handling
        # specifically for a pipeline, the model is not expected to have a hf_device_map attribute
        if device != "accelerate" and hasattr(model.model, "hf_device_map"):
            delattr(model.model, "hf_device_map")
        return

    # sanity check for expected device types
    if device not in ["cpu", "cuda", "mps", "accelerate"]:
        raise ValueError("Device must be a string in [cpu, cuda, mps, accelerate].")

    # do not cast if the model is already on the correct device
    if get_device(model) == device:
        return

    if device == "accelerate":
        if device_map is None:
            raise ValueError("Device map is required when moving to accelerate.")
        cast_model_to_accelerate_device_map(model, device_map)

    else:
        if get_device(model) == "accelerate":
            remove_all_accelerate_hooks(model)
            # transformers model maintain single-device models with a None map, diffusers does not
            model.hf_device_map = {"": "cpu" if device == "cpu" else 0}

        try:
            model.to(device)
        except torch.cuda.OutOfMemoryError as e:
            # there is anyway no way to recover from this error
            # raise it here for better traceability
            raise e
        except (ValueError, RecursionError, RuntimeError, AttributeError, TypeError) as e:
            if raise_error:
                raise ValueError(f"Could not move model to device: {str(e)}")
            else:
                pruna_logger.warning(f"Could not move model to device: {str(e)}")
    safe_memory_cleanup()


def remove_all_accelerate_hooks(model: Any) -> None:
    """
    Remove all hooks from the model.

    This is a helper function to remove all hooks from the model.
    It is used to avoid the RecursionError that occurs when the model is referencing itself.

    Parameters
    ----------
    model : Any
        The model to remove the hooks from.
    """
    if hasattr(model, "reset_device_map"):
        # remove distributed device state to be able to use ".to" for diffusers models
        try:
            model.reset_device_map()
        # inside reset device map, diffusers will attempt device casting and bnb is being difficult
        except ValueError as e:
            if "bitsandbytes" in str(e):
                pass
            else:
                raise e

    if safe_is_instance(model, torch.nn.Module):
        # transformers models are all torch.nn.Module, which is what the hook removal expects
        remove_hook_from_module(model, recurse=True)
    elif hasattr(model, "components"):
        # diffusers pipelines e.g. are not torch modules, so we need to find all attributes that are modules
        # we only do this at the first level, recurse will take care of the rest
        for attr in model.components:
            if isinstance(getattr(model, attr), torch.nn.Module):
                remove_hook_from_module(getattr(model, attr), recurse=True)
    else:
        pruna_logger.warning(
            f"Could not remove hooks from {type(model)}, is not a torch.nn.Module and does not have 'components' "
        )


def cast_model_to_accelerate_device_map(model, device_map):
    """
    Cast a Transformers or Diffusers model to devices according to a given device_map.

    Assumes:
    - device_map only contains CUDA device indices as integers (e.g., 0, 1, 2, ...)
    - device_map is the one created by accelerate/diffusers/transformers during from_pretrained
    - No disk or CPU devices in device_map (raises ValueError if encountered)

    Parameters
    ----------
    model : torch.nn.Module
        The model to cast.
    device_map : dict
        A dictionary mapping module names (str) to CUDA device indices (int).
    """
    if any(not isinstance(dev, int) for dev in device_map.values()):
        raise ValueError("All devices in device_map must be CUDA device indices (integers).")

    if not isinstance(model, torch.nn.Module):
        for target, device in device_map.items():
            dispatch_model(getattr(model, target), device_map={"": device}, force_hooks=True)
    else:
        dispatch_model(model, device_map=device_map, force_hooks=True)

    model.hf_device_map = device_map.copy()


def get_device(model: Any) -> str:
    """
    Get the device of the model.

    Parameters
    ----------
    model : Any
        The model to get the device from.

    Returns
    -------
    str
        The device or device map of the model.
    """
    if isinstance(model, Pipeline):
        return get_device(model.model)

    # a device map that points the whole model to the same device (only key is "") is not considered distributed
    # when casting a model like this with "to" the device map is not maintained, so we rely on the model.device attribute
    if hasattr(model, "hf_device_map") and model.hf_device_map is not None and list(model.hf_device_map.keys()) != [""]:
        model_device = "accelerate"

    elif hasattr(model, "device"):
        model_device = model.device

    else:
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            raise ValueError("Could not determine device of model, model has no device attribute.")

    if isinstance(model_device, torch.device):
        model_device = model_device.type

    return model_device


def get_device_map(model: Any, subset_key: str | None = None) -> dict[str, str]:
    """
    Get the device map of the model.

    Parameters
    ----------
    model : Any
        The model to get the device map from.
    subset_key : str | None
        The key of a submodule for which to get the device map. This only applies in the case of accelerate-distributed
        models, in all other cases the mapping will just be {"": device} which is applicable also for submodules.

    Returns
    -------
    dict[str, str]
        The device map of the model.
    """
    model_device = get_device(model)
    if model_device == "accelerate":
        if subset_key is None:
            return model.hf_device_map
        else:
            return model.hf_device_map[subset_key]
    else:
        device = "cuda:0" if model_device == "cuda" else model_device
        return {"": device}


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


def set_to_best_available_device(device: str | torch.device | None) -> str:
    """
    Set the device to the best available device.

    Supports 'cuda', 'mps', 'cpu' and other PyTorch devices.
    If device is None, the best available device will be returned.

    Parameters
    ----------
    device : str | torch.device | None
        Device to validate (e.g. 'cuda', 'mps', 'cpu').

    Returns
    -------
    str
        Best available device name.
    """
    if isinstance(device, torch.device):
        device = device.type

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        pruna_logger.info(f"Using best available device: '{device}'")
        return device

    if device == "cpu":
        return "cpu"

    if device == "accelerate":
        if not torch.cuda.is_available() and not torch.backends.mps.is_available():
            raise ValueError("'accelerate' requested but neither CUDA nor MPS is available.")
        return "accelerate"

    if device == "cuda":
        if not torch.cuda.is_available():
            pruna_logger.warning("'cuda' requested but not available.")
            return set_to_best_available_device(device=None)
        return "cuda"

    if device == "mps":
        if not torch.backends.mps.is_available():
            pruna_logger.warning("'mps' requested but not available.")
            return set_to_best_available_device(device=None)
        return "mps"

    raise ValueError(f"Device not supported: '{device}'")


class ModelContext:
    """
    Context manager for handling the model.

    Parameters
    ----------
    model : ModelMixin
        The model to handle. Can be a transformer model, UNet, or other ModelMixin.
    """

    def __init__(self, model: "ModelMixin") -> None:
        """
        Context manager for handling the model.

        Parameters
        ----------
        model : ModelMixin
            The model to handle. Can be a transformer model, UNet, or other pipeline.
        """
        self.pipeline = model

    def __enter__(self) -> tuple[ModelMixin, Any, str | None]:
        """
        Enter the context manager.

        Returns
        -------
        ModelMixin
            The working model.
        Any
            The denoiser type.
        str | None
            The denoiser type.
        """
        if hasattr(self.pipeline, "transformer"):
            self.working_model = self.pipeline.transformer
            self.denoiser_type = "transformer"
        elif hasattr(self.pipeline, "unet"):
            self.working_model = self.pipeline.unet
            self.denoiser_type = "unet"
        elif hasattr(self.pipeline, "model") and hasattr(self.pipeline.model, "language_model"):
            self.working_model = self.pipeline.model.language_model
            self.denoiser_type = "language_model"
        else:
            self.working_model = self.pipeline
            self.denoiser_type = None  # type: ignore [assignment]
        return self.pipeline, self.working_model, self.denoiser_type

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """
        Exit the context manager.

        Parameters
        ----------
        exc_type : Exception
            The exception type.
        exc_value : Exception
            The exception value.
        traceback : Exception
            The traceback.
        """
        if hasattr(self.pipeline, "transformer"):
            self.pipeline.transformer = self.pipeline.working_model
        elif hasattr(self.pipeline, "unet"):
            self.pipeline.unet = self.pipeline.working_model
        elif hasattr(self.pipeline, "model") and hasattr(self.pipeline.model, "language_model"):
            self.pipeline.model.language_model = self.pipeline.working_model
        else:
            self.pipeline = self.pipeline.working_model
        del self.pipeline.working_model

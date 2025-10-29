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
from contextlib import AbstractContextManager
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
    file_path = Path(path) / json_name
    with file_path.open("r") as fp:
        model_index = json.load(fp)
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


def safe_is_instance(model: Any, instance_type: Any) -> bool:
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


def move_to_device(
    model: Any,
    device: str | torch.device,
    raise_error: bool = False,
    device_map: dict[str, str] | None = None,
) -> None:
    """
    Move the model to a specific device.

    Parameters
    ----------
    model : Any
        The model to move.
    device : str | torch.device
        The device to move the model to. Can be a string like "cpu", "cuda:0", "mps", "accelerate"
        or a torch.device object.
    raise_error : bool
        Whether to raise an error when the device movement fails.
    device_map : dict[str, str] | None
        The device map to use if the target device is "accelerate".
    """
    if safe_is_instance(model, Pipeline):
        move_to_device(model.model, device, raise_error, device_map)
        # this is a workaround for a flaw in the transformers pipeline handling
        # specifically for a pipeline, the model is not expected to have a hf_device_map attribute
        if device != "accelerate" and hasattr(model.model, "hf_device_map"):
            delattr(model.model, "hf_device_map")
        return

    device = device_to_string(device)
    # To handle the device cases like "cuda:0, cuda:1"
    device_type, device_index = split_device(device)

    # sanity check for expected device types
    if device_type in ["cpu", "cuda", "mps"]:
        device = f"{device_type}:{device_index}" if device_index is not None else device_type
    elif device_type == "accelerate":
        pass  # Handle accelerate separately
    else:
        raise ValueError("Device must be a string starting with [cpu, cuda, mps, accelerate].")

    # do not cast if the model is already on the correct device
    if get_device(model) == device:
        return

    if device == "accelerate":
        if hasattr(model, "smash_config") and device_map is None:
            device_map = model.smash_config.device_map
        if device_map is None:
            raise ValueError("Device map is required when moving to accelerate.")
        cast_model_to_accelerate_device_map(model, device_map)
    else:
        if get_device(model) == "accelerate":
            remove_all_accelerate_hooks(model)
            # transformers model maintain single-device models with a None map, diffusers does not
            # Parse device index from device string for proper device mapping
            if device.startswith("cuda:"):
                model.hf_device_map = {"": device_index}
            else:
                model.hf_device_map = {"": "cpu" if device == "cpu" else 0}
        try:
            model.to(device)
            # Avoid circular imports
            from pruna.engine.model_checks import is_gptq_model

            # Special handling for GPTQ models to ensure all quantization tensors are on the correct device
            if is_gptq_model(model):
                _ensure_gptq_device_consistency(model, device)

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


def _ensure_gptq_device_consistency(model: Any, target_device: str) -> None:
    """
    Ensure all GPTQ quantization tensors are on the correct device.

    This fixes device mismatch issues where GPTQ quantization parameters
    might be on different devices than the model weights.

    Parameters
    ----------
    model : Any
        The GPTQ model to fix.
    target_device : str
        The target device string (e.g., "cuda", "cuda:0", "cpu").
    """
    try:
        for name, module in model.named_modules():
            # Handle GPTQ-specific quantized linear layers
            if hasattr(module, "qweight") or "qlinear" in str(type(module)).lower():
                # Move all quantization-related tensors to the target device
                # Include all Marlin-specific tensors for GPTQ compatibility
                known_attrs = ["qweight", "qzeros", "scales", "g_idx", "g_idx_sort_indices", "workspace", "zp", "bias"]

                for attr_name in known_attrs:
                    if hasattr(module, attr_name):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, torch.Tensor) and attr.device != torch.device(target_device):
                            try:
                                setattr(module, attr_name, attr.to(target_device))
                            except Exception as e:
                                # Some tensors might be read-only or have special handling
                                pruna_logger.debug(f"Could not move {attr_name} in {name} to {target_device}: {e}")

                # Comprehensive scan: move ALL tensor attributes that might be missed
                # This catches TorchQuantLinear and other GPTQ variant tensors
                for attr_name in dir(module):
                    if not attr_name.startswith("_") and attr_name not in known_attrs:
                        try:
                            attr = getattr(module, attr_name)
                            if isinstance(attr, torch.Tensor) and attr.device != torch.device(target_device):
                                try:
                                    setattr(module, attr_name, attr.to(target_device))
                                except Exception as e:
                                    # Some tensors might be read-only or have special handling
                                    pruna_logger.debug(f"Could not move {attr_name} in {name} to {target_device}: {e}")
                        except Exception:
                            # Skip attributes that can't be accessed
                            pass

                # Also check for any buffer tensors
                for buffer_name, buffer in module.named_buffers():
                    if buffer.device != torch.device(target_device):
                        try:
                            buffer.data = buffer.data.to(target_device)
                        except Exception as e:
                            pruna_logger.warning(
                                f"Could not move buffer {buffer_name} in {name} to {target_device}: {e}"
                            )

                # Check parameters as well
                for param_name, param in module.named_parameters(recurse=False):
                    if param.device != torch.device(target_device):
                        try:
                            param.data = param.data.to(target_device)
                        except Exception as e:
                            pruna_logger.warning(
                                f"Could not move parameter {param_name} in {name} to {target_device}: {e}"
                            )

    except Exception as e:
        pruna_logger.warning(f"Error during GPTQ device consistency check: {e}")


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


def get_device_type(model: Any) -> str:
    """
    Get the device type of the model.

    Parameters
    ----------
    model : Any
        The model to get the device type from.

    Returns
    -------
    str
        The device type of the model.
    """
    return split_device(get_device(model))[0]


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
    if safe_is_instance(model, Pipeline):
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

    # model_device.type ignores the device index. Added a new function to convert to string.
    model_device = device_to_string(model_device)

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
        if model_device.startswith("cuda"):
            model_device = _resolve_cuda_device(model_device)
        return {"": model_device}


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
        except Exception as e:
            pruna_logger.warning(f"Could not set model to evaluation mode: {str(e)}")
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


def _resolve_cuda_device(device: str, bytes_free_per_gpu: dict[int, int] | None = None) -> str:
    """
    Resolve CUDA device string to a valid CUDA device.

    Parameters
    ----------
    device : str
        CUDA device string (e.g. "cuda", "cuda:0", "cuda:1")

    Returns
    -------
    str
        Valid CUDA device string
    """
    device_type, device_index = split_device(device)
    if not torch.cuda.is_available():
        pruna_logger.warning("'cuda' requested but not available.")
        return set_to_best_available_device(device=None)

    # When we have a dict of available GPUs and space on them,
    # we set the device to the one with the most free memory.
    if bytes_free_per_gpu is not None:
        if device_index != 0:  # Not the default device
            pruna_logger.warning(
                "You're requesting a specific CUDA device, "
                "but the function will return the device with the most free memory."
            )
        biggest_free_gpu = max(bytes_free_per_gpu, key=lambda x: bytes_free_per_gpu[x])
        return f"cuda:{biggest_free_gpu}"

    if device_index is None or device_index >= torch.cuda.device_count():
        pruna_logger.warning(f"CUDA device {device_index} not available, using device 0")
        device_index = 0
    return f"cuda:{device_index}"


def find_bytes_free_per_gpu() -> dict[int, int]:
    """
    Compute the number of bytes free per GPU.

    Returns
    -------
    dict[int, int]
        The number of bytes free per GPU.
    """
    if torch.cuda.is_available():
        return {i: torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())}
    else:
        return {}


def set_to_best_available_device(
    device: str | torch.device | None, bytes_free_per_gpu: dict[int, int] | None = None
) -> str:
    """
    Set the device to the best available device.

    Supports 'cuda', 'mps', 'cpu' and other PyTorch devices.
    If device is None, the best available device will be returned.

    Parameters
    ----------
    device : str | torch.device | None
        Device to validate (e.g. 'cuda', 'mps', 'cpu').
    bytes_free_per_gpu : dict[int, int] | None
        The number of bytes free per GPU.

    Returns
    -------
    str
        Best available device name.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        pruna_logger.info(f"Using best available device: '{device}'")
        return device

    device = device_to_string(device)
    device_type, device_index = split_device(device)

    if device_type == "cpu":
        return "cpu"
    elif device_type == "accelerate":
        if not torch.cuda.is_available() and not torch.backends.mps.is_available():
            raise ValueError("'accelerate' requested but neither CUDA nor MPS is available.")
        return "accelerate"
    elif device_type == "cuda":
        return _resolve_cuda_device(f"cuda:{device_index}" if device_index is not None else "cuda", bytes_free_per_gpu)
    elif device_type == "mps":
        if not torch.backends.mps.is_available():
            pruna_logger.warning("'mps' requested but not available.")
            return set_to_best_available_device(device=None)
        return "mps"

    raise ValueError(f"Device not supported: '{device}'")


def device_to_string(device: str | torch.device) -> str:
    """
    Convert a device to a string.

    Parameters
    ----------
    device : str | torch.device
        The device to convert.

    Returns
    -------
    str
        The device as a string.
    """
    if isinstance(device, torch.device):
        return str(device)
    elif isinstance(device, str):
        return device
    else:
        raise ValueError(f"Unsupported device type: {type(device)}")


def split_device(device: str, strict: bool = True) -> tuple[str, int | None]:
    """
    Split a device string into a type and index.

    Parameters
    ----------
    device : str
        The device to split.
    strict : bool
        Whether to raise an error if the device is not in allowed devices

    Returns
    -------
    tuple[str, int | None]
        The type and index of the device.
    """
    device = device.lower()
    if ":" in device:
        device_type, device_index_str = device.split(":", 1)
        if device_type not in ("cuda", "mps") and strict:
            raise ValueError(f"Unsupported device type '{device_type}'.")
        try:
            device_index = int(device_index_str)
        except ValueError:
            raise ValueError("Device index must be an integer.")
        return device_type, device_index
    if device in ("cuda", "mps"):
        return device, 0  # treat bare cuda and mps as first device
    if device in ("cpu", "accelerate"):
        return device, None
    if strict:
        raise ValueError(f"Unsupported device: '{device}'.")
    return device, None


class ModelContext(AbstractContextManager):
    """
    Context manager for handling the model.

    Parameters
    ----------
    model : ModelMixin
        The model to handle. Can be a transformer model, UNet, or other ModelMixin.
    """

    def __init__(self, model: "ModelMixin", read_only: bool = False) -> None:
        """
        Context manager for handling the model.

        Parameters
        ----------
        model : ModelMixin
            The model to handle. Can be a transformer model, UNet, or other pipeline.
        """
        self.model = model
        self.read_only = read_only
        self.smashed_working_model = None
        self.path_to_working_model: str | None = None

    def __enter__(self) -> tuple[ModelContext, Any]:
        """
        Enter the context manager.

        Returns
        -------
        ModelContext
            The context manager.
        Any
            The working model.
        """
        if hasattr(self.model, "transformer"):
            working_model = self.model.transformer
            self.path_to_working_model = "transformer"
        elif hasattr(self.model, "unet"):
            working_model = self.model.unet
            self.path_to_working_model = "unet"
        elif hasattr(self.model, "model") and hasattr(self.model.model, "language_model"):
            working_model = self.model.model.language_model
            self.path_to_working_model = "model.language_model"
        else:
            working_model = self.model

        return self, working_model

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
        if self.smashed_working_model is None:
            if self.read_only:
                return
            else:
                raise RuntimeError(
                    "ModelContext is not in read-only mode, but the working model has not been updated. "
                    "Make sure to call `update_working_model` with the adapted model "
                    "before exiting the context manager. After exiting the context manager retrieve "
                    "the updated abstracted model via `get_updated_model` to e.g. return "
                    "it after finishing the current compression algorithm."
                )
        else:
            if self.read_only:
                raise RuntimeWarning(
                    "ModelContext is in read-only mode, but the working model has been updated. "
                    "This can lead to unexpected behavior. If you want to update the model "
                    "make sure to enter the context manager with `read-only=False` which is the default."
                )

        if self.path_to_working_model is not None:
            # Handle nested paths like "model.language_model"
            path_parts = self.path_to_working_model.split(".")
            current_obj = self.model
            # Navigate to the parent object that contains the final attribute
            for part in path_parts[:-1]:
                current_obj = getattr(current_obj, part)
            # Set the final attribute
            setattr(current_obj, path_parts[-1], self.smashed_working_model)
        else:
            self.model = self.smashed_working_model

        del self.smashed_working_model
        safe_memory_cleanup()

    def update_working_model(self, working_model: Any) -> None:
        """
        Set the smashed working model.

        Parameters
        ----------
        working_model : Any
            The smashed working model.
        """
        self.smashed_working_model = working_model

    def get_updated_model(self) -> "ModelMixin":
        """
        Get the smashed model.

        Returns
        -------
        ModelMixin
            The smashed model.
        """
        return self.model

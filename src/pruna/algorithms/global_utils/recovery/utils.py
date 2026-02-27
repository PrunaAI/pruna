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

import inspect
from typing import Any, Callable, List, Tuple

import torch


def split_defaults(defaults: dict[str, Any]) -> tuple[dict[str, str], dict[str, int | float]]:
    """
    Split the defaults into string and numeric defaults.

    Parameters
    ----------
    defaults : dict[str, Any]
        The defaults to split.

    Returns
    -------
    tuple[dict[str, str], dict[str, int | float]]
        The string and numeric defaults.
    """
    string_defaults: dict[str, str] = {k: str(v) for k, v in defaults.items() if isinstance(v, str)}
    numeric_defaults: dict[str, int | float] = {k: v for k, v in defaults.items() if k not in string_defaults}
    return string_defaults, numeric_defaults


def cast_parameters(parameters: List[torch.nn.Parameter] | torch.nn.ParameterList, dtype: torch.dtype | str) -> None:
    """
    Cast the parameters of the model to the given dtype.

    Parameters
    ----------
    parameters : List[torch.nn.Parameter] | torch.nn.ParameterList
        The parameters to cast.
    dtype : torch.dtype
        The dtype to cast the parameters to.
    """
    for param in parameters:
        param.data = param.data.to(dtype)
        if hasattr(param, "grad") and param.grad is not None:
            param.grad = param.grad.to(dtype)


def get_dtype(model: Any) -> torch.dtype:
    """
    Get the dtype of the model.

    Parameters
    ----------
    model : Any
        The model to get the dtype from.

    Returns
    -------
    torch.dtype
        The dtype of the model.
    """
    if hasattr(model, "dtype"):
        dtype = model.dtype
        return dtype
    else:  # fallback by looking for a float type parameter
        for param in model.parameters():
            if "float" in str(param.dtype):
                dtype = param.dtype
                return dtype
    # last resort: use the first parameter's type
    return next(iter(model.parameters())).dtype


def get_trainable_parameters(model: Any) -> List[torch.nn.Parameter]:
    """
    Get the trainable parameters of the model or pipeline.

    Parameters
    ----------
    model : Any
        The model or pipeline to get the trainable parameters from.

    Returns
    -------
    List[torch.nn.Parameter]
        The trainable parameters of the model or pipeline.
    """
    if isinstance(model, torch.nn.Module):
        return [param for param in model.parameters() if param.requires_grad]

    modules = [component for _, component in inspect.getmembers(model) if isinstance(component, torch.nn.Module)]
    return [param for module in modules for param in module.parameters() if param.requires_grad]


def str_to_int(s: str) -> int:
    """
    Deterministically convert a string to an integer.

    Parameters
    ----------
    s : str
        The string to convert to an integer.

    Returns
    -------
    int
        An integer obtained from the string.
    """
    return int(s.encode("utf-8").hex(), 16)


def filter_kwargs(function: Callable, kwargs: dict[str, Any]) -> Tuple[dict[str, Any], dict[str, Any]]:
    """
    Filter the kwargs of a function to separate the arguments that are valid for the function and those that are not.

    Parameters
    ----------
    function : Callable
        The function to filter the kwargs of.
    kwargs : dict[str, Any]
        The kwargs to filter.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        The valid and invalid kwargs.
    """
    valid_kwargs = {}
    invalid_kwargs = {}
    signature = inspect.signature(function)
    for key, value in kwargs.items():
        if key in signature.parameters:
            valid_kwargs[key] = value
        else:
            invalid_kwargs[key] = value
    return valid_kwargs, invalid_kwargs

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
from functools import partial
from typing import Any, Callable, Dict, Tuple

import torch
from peft import PeftModel

from pruna.algorithms.global_utils.recovery.finetuners.diffusers.utils import get_denoiser_attr
from pruna.logging.logger import pruna_logger


def get_latent_replacement_fn(pipeline: Any) -> Callable:
    """
    Get a function replacing the denoiser's latent argument with the recorded latent.

    Parameters
    ----------
    pipeline : Any
        The pipeline calling the denoiser.

    Returns
    -------
    Callable
        A function replacing the denoiser's latent argument with the recorded latent.
    """
    expected_arg_name, expected_arg_idx = _get_expected_arg_name_and_idx(pipeline)
    return partial(_replace_latent, expected_arg_idx, arg_name=expected_arg_name)


def get_latent_extractor_fn(pipeline: Any) -> Callable:
    """
    Get a function extracting the latent from the pipeline's input arguments.

    Parameters
    ----------
    pipeline : Any
        The pipeline calling the denoiser.

    Returns
    -------
    Callable
        A function extracting the latent from the pipeline's input arguments.
    """
    expected_arg_name, expected_arg_idx = _get_expected_arg_name_and_idx(pipeline)
    return partial(_extract_latent, expected_arg_idx, arg_name=expected_arg_name)


def _get_expected_arg_name_and_idx(pipeline: Any) -> Tuple[str, int]:
    """
    Get the expected argument name and index in the denoiser's forward method.

    This is used to generalize latent manipulation across different pipelines.
    Some pipelines call their denoiser with latent as the first argument, others
    with a name argument. This function returns both a name derived from the pipeline's
    architecture, and the index, so both can be used depending on the situation.

    Parameters
    ----------
    pipeline : Any
        The pipeline calling the denoiser.

    Returns
    -------
    Tuple[str, int]
        A tuple of the expected argument name and index.
    """
    denoiser, denoiser_attr = get_denoiser_attr(pipeline)
    if denoiser is None:
        raise ValueError("No denoiser attribute found in pipeline")

    if denoiser_attr == "unet":
        expected_arg_name = "sample"
    elif denoiser_attr == "transformer":
        expected_arg_name = "hidden_states"
    else:
        raise ValueError(f"Unknown denoiser attribute: {denoiser_attr}")

    if isinstance(denoiser, PeftModel):
        # PEFTModel does not wrap the forward method with the base model's signature
        sig = inspect.signature(denoiser.model.forward)
    else:
        sig = inspect.signature(denoiser.forward)

    expected_arg_idx = next((i for i, p in enumerate(sig.parameters.keys()) if p == expected_arg_name), None)
    if expected_arg_idx is None:
        pruna_logger.error(f"Argument '{expected_arg_name}' not found in the denoiser's signature.")
        raise ValueError(f"Argument '{expected_arg_name}' not found in the denoiser's signature")
    else:
        expected_arg_idx -= int("self" in sig.parameters)

    return expected_arg_name, expected_arg_idx


def _replace_latent(
    arg_idx: int, latent: torch.Tensor, args: Tuple, kwargs: Dict[str, Any], arg_name: str | None = None
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """
    Replace the argument at the given index with the given name with the given value.

    Parameters
    ----------
    arg_idx : int
        The index of the argument to replace.
    latent : torch.Tensor
        The latent to replace the argument with.
    args : Tuple
        The arguments to the function.
    arg_name : str | None
        The name of the argument to replace.
    kwargs : Dict[str, Any]
        The keyword arguments to the function.

    Returns
    -------
    Tuple[Tuple, Dict[str, Any]]
        The arguments and keyword arguments to the function with the argument replaced.
    """
    if arg_name is not None and arg_name in kwargs:
        kwargs[arg_name] = latent
    elif len(args) > arg_idx:
        args = tuple(latent if i == arg_idx else x for i, x in enumerate(args))
    else:
        raise ValueError(
            "Argument mismatch when replacing denoiser arguments: "
            f"attempted to replace {arg_name} at position {arg_idx}, "
            f"but found {len(args)} positional arguments, and keyword arguments {list(kwargs.keys())}"
        )
    return args, kwargs


def _extract_latent(arg_idx: int, args: Tuple, kwargs: Dict[str, Any], arg_name: str | None = None) -> torch.Tensor:
    """
    Extract the latent from the pipeline's input arguments.

    Parameters
    ----------
    arg_idx : int
        The index of the argument to extract.
    args : Tuple
        The arguments to the function.
    kwargs : Dict[str, Any]
        The keyword arguments to the function.
    arg_name : str | None
        The name of the argument to extract.

    Returns
    -------
    torch.Tensor
        The latent extracted from the pipeline's input arguments.
    """
    if arg_name is not None and arg_name in kwargs:
        if not isinstance(kwargs[arg_name], torch.Tensor):
            raise ValueError(f"Expected a tensor, got {type(kwargs[arg_name])}")
        return kwargs[arg_name]
    elif len(args) > arg_idx:
        if not isinstance(args[arg_idx], torch.Tensor):
            raise ValueError(f"Expected a tensor, got {type(args[arg_idx])}")
        return args[arg_idx]
    else:
        raise ValueError(
            f"Argument mismatch when extracting denoiser arguments: "
            f"attempted to extract {arg_name} at position {arg_idx}, "
            f"but found {len(args)} positional arguments, and keyword arguments {list(kwargs.keys())}"
        )

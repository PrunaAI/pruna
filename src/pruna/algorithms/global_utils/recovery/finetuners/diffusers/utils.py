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
from typing import Any, Dict, Tuple, Union, get_args, get_origin

import torch
from diffusers.utils import BaseOutput

from pruna.engine.model_checks import (
    is_flux_pipeline,
    is_sana_pipeline,
    is_sd_pipeline,
    is_sdxl_pipeline,
)
from pruna.logging.logger import pruna_logger


def get_denoiser_attr(pipeline: Any) -> Tuple[Any | None, str]:
    """
    Get the denoiser attribute in a pipeline and its name.

    Parameters
    ----------
    pipeline : Any
        The pipeline to get the denoiser attribute from.

    Returns
    -------
    Tuple[Any | None, str]
        The denoiser attribute and its name. If no attribute is found, return None and an empty string.
    """
    possible_names = ["unet", "transformer"]
    for name in possible_names:
        if hasattr(pipeline, name):
            return getattr(pipeline, name), name
    return None, ""


def set_denoiser_attr(pipeline: Any, denoiser: torch.nn.Module) -> None:
    """
    Set the denoiser attribute in a pipeline.

    Parameters
    ----------
    pipeline : Any
        The pipeline to set the denoiser attribute in.
    denoiser : torch.nn.Module
        The denoiser to set in the pipeline.
    """
    possible_names = ["unet", "transformer"]
    for name in possible_names:
        if hasattr(pipeline, name):
            setattr(pipeline, name, denoiser)
            return
    raise ValueError(f"Unknown pipeline: {pipeline.__class__.__name__}")


def uses_prompt_2(pipeline: Any) -> bool:
    """
    Check if the pipeline uses a second prompt.

    Parameters
    ----------
    pipeline : Any
        The pipeline to check.

    Returns
    -------
    bool
        True if the pipeline uses a second prompt, False otherwise.
    """
    return is_flux_pipeline(pipeline)


def get_encode_arguments(pipeline: Any) -> Dict[str, Any]:
    """
    Get arguments specific to the encode_prompt function of each pipeline type.

    Parameters
    ----------
    pipeline : Any
        The pipeline to get the encode_prompt method from.

    Returns
    -------
    Dict[str, Any]
        Keyword arguments to pass to the encode_prompt function.
    """
    if is_sd_pipeline(pipeline) or is_sdxl_pipeline(pipeline):
        return dict(do_classifier_free_guidance=True)
    elif is_sana_pipeline(pipeline) or is_flux_pipeline(pipeline):
        return dict()
    else:
        raise ValueError(f"Unknown pipeline: {pipeline.__class__.__name__}")


def move_secondary_components(pipeline: Any, device: str | torch.device) -> None:
    """
    Move the secondary components of the pipeline to the device.

    Parameters
    ----------
    pipeline : Any
        The pipeline to move the secondary components to.
    device : str | torch.device
        The device to move the secondary components to.
    """
    if hasattr(pipeline, "text_encoder"):
        pipeline.text_encoder.to(device=device)
    if hasattr(pipeline, "text_encoder_2"):
        pipeline.text_encoder_2.to(device=device)


def check_resolution_mismatch(pipeline: Any, dataloader: torch.utils.data.DataLoader) -> None:
    """
    Log a warning if there's a mismatch between the dataloader image resolution and the pipeline's configured resolution.

    Parameters
    ----------
    pipeline : Any
        The pipeline.
    dataloader : torch.utils.data.DataLoader
        The dataloader containing the training images.
    """
    # Get first batch to check image size
    first_batch = next(iter(dataloader))
    images = first_batch[1]  # (batch_size, channels, height, width)
    image_height, image_width = images.shape[-2:]

    if is_sd_pipeline(pipeline) or is_sdxl_pipeline(pipeline):
        config_size = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    elif is_sana_pipeline(pipeline):
        config_size = pipeline.transformer.config.sample_size * pipeline.vae_scale_factor
    elif is_flux_pipeline(pipeline):
        config_size = pipeline.vae.config.sample_size
    else:
        pruna_logger.warning(
            f"Unknown pipeline: {pipeline.__class__.__name__}, please make sure the image resolution matches "
            "the pipeline's configured resolution for finetuning as it might affect recovery performance."
        )
        return

    if image_height != config_size or image_width != config_size:
        pruna_logger.warning(
            f"The resolution of the provided dataset ({image_height}x{image_width}) differs from "
            f"the pipeline's configured resolution ({config_size}x{config_size}). "
            "This might affect recovery performance."
        )


def get_denoiser_output_class(denoiser: Any) -> type[BaseOutput]:
    """
    Get the denoiser output class for a denoiser by inspecting its forward method signature.

    Parameters
    ----------
    denoiser : Any
        The denoiser whose output class to get.

    Returns
    -------
    type[BaseOutput] | None
        The denoiser output class if found, otherwise None.
    """
    # Get the forward method signature
    signature = inspect.signature(denoiser.forward)
    output_type = signature.return_annotation

    # Extract different types from Union
    output_types = get_args(output_type) if get_origin(output_type) is Union else [output_type]
    base_output_types = [t for t in output_types if inspect.isclass(t) and issubclass(t, BaseOutput)]

    if len(base_output_types) == 1:
        return base_output_types[0]
    else:
        raise ValueError(
            f"Could not infer the denoiser's return type, expected exactly one BaseOutput type, got {output_type}"
        )

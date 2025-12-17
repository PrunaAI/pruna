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

import sys
import tempfile
import types
from collections.abc import Iterable
from typing import Any

import numpy as np
from ConfigSpace import Constant
from PIL import Image

from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.engine.model_checks import is_diffusers_pipeline
from pruna.engine.save import SAVE_FUNCTIONS


class RealESRGAN(PrunaAlgorithmBase):
    """
    Implement Real-ESRGAN upscaling for images.

    This enhancer applies the Real-ESRGAN model to upscale images produced by
    diffusion models or other image generation pipelines.

    Attributes
    ----------
    algorithm_name : str
        The name identifier for this algorithm.
    references : dict[str, str]
        Dictionary containing references to the original paper and implementation.
    tokenizer_required : bool
        Whether a tokenizer is required for this enhancer.
    processor_required : bool
        Whether a processor is required for this enhancer.
    run_on_cpu : bool
        Whether this enhancer can run on CPU.
    run_on_cuda : bool
        Whether this enhancer can run on CUDA devices.
    dataset_required : bool
        Whether a dataset is required for this enhancer.
    compatible_algorithms : dict
        Dictionary of algorithms that are compatible with this enhancer.
    """

    algorithm_name: str = "realesrgan_upscale"
    group_tags: list[AlgorithmTag] = []
    save_fn = SAVE_FUNCTIONS.reapply
    references: dict[str, str] = {
        "Paper": "https://arxiv.org/abs/2107.10833",
        "GitHub": "https://github.com/xinntao/Real-ESRGAN",
    }
    tokenizer_required: bool = False
    processor_required: bool = False
    runs_on: list[str] = ["cpu", "cuda", "accelerate"]
    dataset_required: bool = False
    compatible_before: Iterable[str | AlgorithmTag] = [
        AlgorithmTag.CACHER,
        "torch_compile",
        "stable_fast",
        "hqq_diffusers",
        "diffusers_int8",
        "torchao",
        "qkv_diffusers",
        "ring_attn"
    ]

    def get_hyperparameters(self) -> list:
        """
        Get the hyperparameters for the RealESRGAN enhancer.

        This method defines the configurable parameters for the RealESRGAN model,
        including scaling factors, tile sizes, padding values, and precision options.

        Returns
        -------
        list
            A list of hyperparameters for the RealESRGAN model, including:
            - outscale: Output scaling factor
            - tile: Tile size for processing large images (0 means no tiling)
            - tile_pad: Padding size for tiles to avoid boundary artifacts
            - pre_pad: Padding before processing
            - face_enhance: Whether to enhance faces specifically
            - fp32: Whether to use FP32 precision instead of FP16
            - netscale: Network scaling factor
        """
        return [
            Constant("outscale", value=4),
            Constant("tile", value=0),
            Constant("tile_pad", value=10),
            Constant("pre_pad", value=0),
            Constant("face_enhance", value=False),
            Constant("fp32", value=False),
            Constant("netscale", value=4),
        ]

    def model_check_fn(self, model: Any) -> bool:
        """
        Check if the model has a unet or transformer from diffusers as an attribute.

        Parameters
        ----------
        model : Any
            The model instance to check.

        Returns
        -------
        bool
            True if the model has a unet or transformer from diffusers as an attribute, False otherwise.
        """
        return is_diffusers_pipeline(model, include_video=False)

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Apply RealESRGAN upscaling to the model.

        This method sets up the RealESRGAN model and integrates it with the
        provided model by wrapping the model's output function to apply
        upscaling automatically.

        Parameters
        ----------
        model : Any
            The model to enhance with RealESRGAN upscaling. This is typically
            an image generation model like a diffusion model.
        smash_config : SmashConfigPrefixWrapper
            The configuration for enhancement, containing parameters like
            tile size, padding values, and precision options.

        Returns
        -------
        Any
            The model with RealESRGAN upscaling applied, which will automatically
            upscale any images produced by the model.
        """
        imported_modules = self.import_algorithm_packages()

        rrdb_net = imported_modules["RRDBNet"](
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
        )
        netscale = smash_config["netscale"]
        model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

        with tempfile.TemporaryDirectory(prefix=str(smash_config["cache_dir"])) as temp_dir:
            model_path = imported_modules["load_file_from_url"](model_url, model_dir=temp_dir, progress=True)

            upsampler = imported_modules["RealESRGANer"](
                scale=netscale,
                model_path=model_path,
                model=rrdb_net,
                tile=smash_config["tile"],
                tile_pad=smash_config["tile_pad"],
                pre_pad=smash_config["pre_pad"],
                half=not smash_config["fp32"],
                gpu_id=0,
            )

            model.upscale_helper = UpscaleHelper(model, upsampler)
            model.upscale_helper.enable()

        return model

    def import_algorithm_packages(self) -> dict[str, Any]:
        """
        Import the necessary packages for the RealESRGAN algorithm.

        This method imports all required dependencies for the RealESRGAN upscaling
        algorithm, including the RRDBNet architecture, download utilities, and
        the RealESRGANer implementation. It also handles a compatibility fix for
        torchvision.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the imported modules, with keys:
            - 'RealESRGANer': The main RealESRGAN implementation
            - 'RRDBNet': The neural network architecture used by RealESRGAN
            - 'load_file_from_url': Utility function to download model weights
        """
        from torchvision.transforms.functional import rgb_to_grayscale

        # Create a module for `torchvision.transforms.functional_tensor`
        functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
        functional_tensor.rgb_to_grayscale = rgb_to_grayscale  # type: ignore

        # Add this module to sys.modules so other imports can access it
        sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor

        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.utils.download_util import load_file_from_url
        from realesrgan import RealESRGANer

        return {"RealESRGANer": RealESRGANer, "RRDBNet": RRDBNet, "load_file_from_url": load_file_from_url}


class UpscaleHelper:
    """
    Helper class for RealESRGAN upscaling.

    This class provides functionality to integrate RealESRGAN upscaling with
    any model by wrapping the model's call method to automatically apply
    upscaling to the output images.

    Parameters
    ----------
    model : Any
        The model to enhance with upscaling.
    upsampler : Any
        The RealESRGAN upsampler instance that performs the actual upscaling.
    """

    def __init__(self, model: Any, upsampler: Any) -> None:
        self.model = model
        self.upsampler = upsampler
        self.original_pipe_call = self.model.__call__

    def _wrapped_pipe_call(self, *args, **kwargs) -> Any:
        """
        Wrap the pipeline call to apply upscaling to the output.

        This method intercepts calls to the model, runs the original call,
        and then applies RealESRGAN upscaling to the output images before
        returning them.

        Parameters
        ----------
        *args : tuple
            Positional arguments to pass to the original pipeline call.
        **kwargs : dict
            Keyword arguments to pass to the original pipeline call.

        Returns
        -------
        Any
            The upscaled output from the pipeline, with RealESRGAN enhancement
            applied to improve image quality and resolution.
        """
        output = self.original_pipe_call(*args, **kwargs)
        enhanced_images = []
        for image in output.images:
            # Get the original image size before enhancement
            original_width, original_height = image.size
            enhanced_image = self.upsampler.enhance(np.array(image))[0]
            enhanced_image = Image.fromarray(enhanced_image)
            enhanced_images.append(enhanced_image)
        output.images = enhanced_images
        return output

    def enable(self) -> None:
        """
        Enable the RealESRGAN upscaling by replacing the pipeline call.

        This method replaces the model's __call__ method with the wrapped
        version that applies upscaling, effectively enabling automatic
        upscaling for all outputs from the model.
        """
        self.model.__call__ = self._wrapped_pipe_call

    def disable(self) -> None:
        """
        Disable the RealESRGAN upscaling by restoring the original pipeline call.

        This method restores the model's original __call__ method, effectively
        disabling the automatic upscaling of outputs from the model.
        """
        self.model.__call__ = self.original_pipe_call

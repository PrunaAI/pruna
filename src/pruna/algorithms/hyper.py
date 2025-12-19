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

import functools
from collections.abc import Iterable
from typing import Any

from diffusers import DDIMScheduler, TCDScheduler
from huggingface_hub import hf_hub_download
from huggingface_hub.utils.tqdm import disable_progress_bars, enable_progress_bars

from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag
from pruna.config.hyperparameters import Boolean
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.engine.model_checks import (
    is_flux_pipeline,
    is_sd_3_pipeline,
    is_sd_pipeline,
    is_sdxl_pipeline,
)
from pruna.engine.save import SAVE_FUNCTIONS
from pruna.logging.logger import pruna_logger


class Hyper(PrunaAlgorithmBase):
    """
    Implement distillation through Hyper LoRA adapters.

    Hyper-SD is a distillation framework that segments the diffusion process into time-step groups to preserve and
    reformulate the ODE trajectory. By integrating human feedback and score distillation, it enables near-lossless
    performance with drastically fewer inference steps.
    """

    algorithm_name: str = "hyper"
    group_tags: list[AlgorithmTag] = [AlgorithmTag.DISTILLER]  # type: ignore[attr-defined]
    references: dict[str, str] = {"Paper": "https://arxiv.org/abs/2404.13686"}
    save_fn = SAVE_FUNCTIONS.save_before_apply
    tokenizer_required: bool = False
    processor_required: bool = False
    runs_on: list[str] = ["cpu", "cuda", "accelerate"]
    dataset_required: bool = False
    compatible_before: Iterable[str | AlgorithmTag] = ["half", "diffusers_int8", "padding_pruning"]
    compatible_after: Iterable[str | AlgorithmTag] = [
        AlgorithmTag.CACHER,
        "torch_compile",
        "stable_fast",
        AlgorithmTag.ENHANCER,  # type: ignore[attr-defined]
        AlgorithmTag.RESAMPLER,  # type: ignore[attr-defined]
    ]

    def get_hyperparameters(self) -> list:
        """
        Get the hyperparameters for the Flux Caching Compiler.

        Returns
        -------
        list
            A list of hyperparameters including cache interval, start step,
            compile_cuda flag, and save_model flag.
        """
        # Default values are chosen based on Table 1 of the FORA paper
        return [
            Boolean(
                "agressive",
                default=False,
                meta=dict(desc="When this is set to True, the model is distilled to even less steps"),
            ),
        ]

    def get_model_type(self, model: Any) -> str:
        """
        Get the model type.

        Parameters
        ----------
        model : Any
            The model to check the type of.

        Returns
        -------
        str
            The type of model - one of 'sdxl', 'sd', 'sd3', or 'flux'.
        """
        if is_sdxl_pipeline(model):
            return "sdxl"
        elif is_sd_pipeline(model):
            return "sd"
        elif is_sd_3_pipeline(model):
            return "sd3"
        else:
            # is_flux_pipeline(model) is guaranteed to be true
            return "flux"

    def model_check_fn(self, model: Any) -> bool:
        """
        Check if the provided model is a valid Flux model.

        Parameters
        ----------
        model : Any
            The model instance to check.

        Returns
        -------
        bool
            True if the model is a valid Flux model, False otherwise.
        """
        return is_flux_pipeline(model) or is_sdxl_pipeline(model) or is_sd_pipeline(model) or is_sd_3_pipeline(model)

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Distill the model.

        Parameters
        ----------
        model : Any
            The model to distill.
        smash_config : SmashConfigPrefixWrapper
            The configuration for the distilling.

        Returns
        -------
        Any
        """
        existing_adapters = model.get_active_adapters() if hasattr(model, "get_active_adapters") else []

        # Define model-specific configurations
        model_configs: dict[str, dict[str, Any]] = {
            "sdxl": {
                "lora_path": (
                    "Hyper-SDXL-8steps-lora.safetensors"
                    if not smash_config["agressive"]
                    else "Hyper-SDXL-4steps-lora.safetensors"
                ),
                "scheduler": lambda m: TCDScheduler.from_config(m.scheduler.config),
                "lora_scale": 1.0,
                "num_inference_steps": 8 if not smash_config["agressive"] else 4,
                "guidance_scale": 0,
            },
            "sd": {
                "lora_path": (
                    "Hyper-SD15-8steps-lora.safetensors"
                    if not smash_config["agressive"]
                    else "Hyper-SD15-4steps-lora.safetensors"
                ),
                "scheduler": lambda m: DDIMScheduler.from_config(m.scheduler.config, timestep_spacing="trailing"),
                "lora_scale": 1.0,
                "num_inference_steps": 8 if not smash_config["agressive"] else 4,
                "guidance_scale": 0,
            },
            "sd3": {
                "lora_path": (
                    "Hyper-SD3-8steps-CFG-lora.safetensors"
                    if not smash_config["agressive"]
                    else "Hyper-SD3-4steps-CFG-lora.safetensors"
                ),
                "lora_scale": 0.125,
                "num_inference_steps": 8 if not smash_config["agressive"] else 4,
                "guidance_scale": 5.0,
            },
            "flux": {
                "lora_path": (
                    "Hyper-FLUX.1-dev-16steps-lora.safetensors"
                    if not smash_config["agressive"]
                    else "Hyper-FLUX.1-dev-8steps-lora.safetensors"
                ),
                "lora_scale": 0.125,
                "num_inference_steps": 16 if not smash_config["agressive"] else 8,
                "guidance_scale": 3.5,
            },
        }

        config = model_configs[self.get_model_type(model)]

        # Load LoRA weights
        with TqdmPositionContext():
            model.load_lora_weights(
                hf_hub_download("ByteDance/Hyper-SD", config["lora_path"]),
                adapter_name="hyper",
                lora_scale=config["lora_scale"],
            )

        # Set adapters
        adapter_weights = [config["lora_scale"]] + ([1.0] * len(existing_adapters) if existing_adapters else [])
        if existing_adapters:
            pruna_logger.info(
                f"Found existing adapters: {existing_adapters} we will use {adapter_weights} as weights by default"
            )
            model.set_adapters(["hyper"] + existing_adapters, adapter_weights=adapter_weights)
            pruna_logger.info(
                "Diffusers does not save adapters by default, "
                "make sure to re-attach existing adapters when saving and loading the model. "
                "Pruna will only re-attach the hyper adapters when loading the model."
            )
        else:
            model.set_adapters("hyper", adapter_weights=[config["lora_scale"]])

        # Set scheduler if specified
        if "scheduler" in config:
            model.scheduler = config["scheduler"](model)

        original_call = model.__call__
        # Wrap the __call__ method to set num_inference_steps to 10 if not provided

        @functools.wraps(model.__call__)
        def hyper_call_wrapper(*args, **kwargs) -> Any:
            """
            Wrapper for the __call__ method to set num_inference_steps to 10 if not provided.

            Parameters
            ----------
            *args : Any
                The arguments to the __call__ method.
            **kwargs : Any
                The keyword arguments to the __call__ method.

            Returns
            -------
            Any
                The result of the __call__ method.
            """
            if "num_inference_steps" not in kwargs:
                num_inference_steps = config["num_inference_steps"]
            else:
                num_inference_steps = kwargs["num_inference_steps"]
                if num_inference_steps != config["num_inference_steps"]:
                    pruna_logger.warning(
                        "'num_inference_steps' is set to %d, "
                        "but we recommend using num_inference_steps=%d with the Hyper distiller.",
                        num_inference_steps,
                        config["num_inference_steps"],
                    )
            kwargs["num_inference_steps"] = num_inference_steps

            if "guidance_scale" not in kwargs:
                guidance_scale = config["guidance_scale"]
            else:
                guidance_scale = kwargs["guidance_scale"]
                if guidance_scale != config["guidance_scale"]:
                    pruna_logger.warning(
                        "'guidance_scale' is set to %.2f, "
                        "but we recommend using guidance_scale=%.2f with the Hyper distiller.",
                        guidance_scale,
                        config["guidance_scale"],
                    )
            kwargs["guidance_scale"] = guidance_scale

            return original_call(*args, **kwargs)

        model.__call__ = hyper_call_wrapper
        return model

    def import_algorithm_packages(self) -> dict[str, Any]:
        """
        Import necessary algorithm packages.

        Returns
        -------
        dict
            An empty dictionary as no packages are imported in this implementation.
        """
        return dict()


class TqdmPositionContext:
    """Context manager for temporarily modifying the TQDM_POSITION environment variable."""

    def __init__(self) -> None:
        pass

    def __enter__(self) -> "TqdmPositionContext":
        """Enter into the TQDM-silenced context."""
        disable_progress_bars()
        return self

    def __exit__(
        self,
        exc_type: BaseException | None,
        exc_value: BaseException | None,
        traceback: Any | None,
    ) -> None:
        """
        Exit the context manager and restore or remove the TQDM_POSITION environment variable.

        This method is automatically called when exiting the context manager block.
        It handles the cleanup of the TQDM_POSITION environment variable by either
        restoring its previous value or removing it completely if it didn't exist before.

        Parameters
        ----------
        exc_type : BaseException or None
            The type of the exception that occurred, if any.
        exc_val : BaseException or None
            The instance of the exception that occurred, if any.
        exc_tb : Any or None
            The traceback of the exception that occurred, if any.

        Returns
        -------
        bool or None
            None or False to propagate exceptions, True to suppress them.
        """
        enable_progress_bars()

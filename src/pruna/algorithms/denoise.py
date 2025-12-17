from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from ConfigSpace import UniformFloatHyperparameter
from diffusers import AutoPipelineForImage2Image, DiffusionPipeline
from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.engine.save import SAVE_FUNCTIONS
from pruna.engine.utils import determine_dtype


class Img2ImgDenoise(PrunaAlgorithmBase):
    """
    Refines images using the model's own image-to-image capabilities.

    This enhancer takes the output images from a diffusion pipeline and refines them
    by smartly reusing the same pipeline. This assumes the base model is a diffusers
    pipeline supporting image-to-image.

    Attributes
    ----------
    algorithm_name : str
        The name identifier for this algorithm.
    references : dict[str, str]
        Dictionary containing references (optional).
    tokenizer_required : bool
        Whether a tokenizer is required (usually False, depends on pipeline).
    processor_required : bool
        Whether a processor is required (usually False, depends on pipeline).
    run_on_cpu : bool
        Whether this enhancer can run on CPU (depends on base model).
    run_on_cuda : bool
        Whether this enhancer can run on CUDA devices (depends on base model).
    dataset_required : bool
        Whether a dataset is required for this enhancer.
    compatible_algorithms : dict
        Dictionary of algorithms that are compatible with this enhancer.
    """

    algorithm_name: str = "img2img_denoise"
    group_tags: list[AlgorithmTag] = [] 
    save_fn = SAVE_FUNCTIONS.reapply
    references: dict[str, str] = {
        "Diffusers": "https://huggingface.co/docs/diffusers/using-diffusers/img2img",
    }
    tokenizer_required: bool = False
    processor_required: bool = False
    runs_on: list[str] = ["cpu", "cuda"]
    dataset_required: bool = False
    compatible_before: Iterable[str | AlgorithmTag] = [
        AlgorithmTag.CACHER,
        "torch_compile",
        "stable_fast",
        "hqq_diffusers",
        "diffusers_int8",
        "torchao",
        "qkv_diffusers",
        "ring_attn",
    ]

    def get_hyperparameters(self) -> list:
        """
        Get the hyperparameters for the Img2Img Denoise enhancer.

        Returns
        -------
        list
            A list of hyperparameters, including:
            - strength: Controls how much noise is added to the input image,
                        influencing how much it changes (0.0-1.0). Lower values
                        mean less change/more refinement.
        """
        return [
            UniformFloatHyperparameter(
                "strength",
                lower=0.0,
                upper=1.0,
                default_value=0.02,
                log=False,
                meta=dict(desc="Strength of the denoising/refinement. Lower values mean less change/more refinement."),
            ),
        ]

    def model_check_fn(self, model: Any) -> bool:
        """
        Check if the model is a diffusers pipeline with UNet or Transformer.

        Parameters
        ----------
        model : Any
            The model instance to check.

        Returns
        -------
        bool
            True if the model seems compatible, False otherwise.
        """
        if not isinstance(model, DiffusionPipeline) or not hasattr(model, "_name_or_path"):
            return False

        model_dtype = determine_dtype(model)

        # check if the model is supported in an img2img pipeline
        try:
            AutoPipelineForImage2Image.from_pretrained(
                model._name_or_path,
                transformer=getattr(model, "transformer", None),
                unet=getattr(model, "unet", None),
                vae=getattr(model, "vae", None),
                text_encoder=getattr(model, "text_encoder", None),
                torch_dtype=model_dtype,
                scheduler=getattr(model, "scheduler", None),
            )
        except Exception:
            return False
        return True

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Apply image-to-image denoising/refinement to the model's output.

        Parameters
        ----------
        model : Any
            The diffusers pipeline model to enhance.
        smash_config : SmashConfigPrefixWrapper
            The configuration containing hyperparameters like 'strength'.

        Returns
        -------
        Any
            The model with its output generation wrapped for refinement.
        """

        model_dtype = determine_dtype(model)

        refiner = AutoPipelineForImage2Image.from_pretrained(
            model._name_or_path,
            transformer=getattr(model, "transformer", None),
            unet=getattr(model, "unet", None),
            vae=getattr(model, "vae", None),
            text_encoder=getattr(model, "text_encoder", None),
            torch_dtype=model_dtype,
            scheduler=getattr(model, "scheduler", None),
        ).to(smash_config.device)

        denoise_strength = smash_config["strength"]

        model.denoise_helper = DenoiseHelper(
            model,
            refiner,
            strength=denoise_strength,
        )
        model.denoise_helper.enable()

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


class DenoiseHelper:
    """
    Helper class to wrap a pipeline's call for image-to-image refinement.

    Intercepts the output images and runs them through the same pipeline
    again using image-to-image mode with a specified strength.

    Parameters
    ----------
    model : Any
        The diffusers pipeline model being wrapped.
    refiner : Any
        The separate pipeline used for the refinement step.
    strength : float
        The strength parameter for the img2img refinement step.
    """

    def __init__(self, model: Any, refiner: Any, strength: float) -> None:
        if not (hasattr(model, "__call__") and callable(model.__call__)):
            raise TypeError("Model must have a callable __call__ method to be wrapped.")
        self.model = model
        self.refiner = refiner
        self.refiner.set_progress_bar_config(disable=True)
        self.original_pipe_call = self.model.__call__
        self.strength = strength
        # Store device for placing tensors if needed
        self.device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def _wrapped_pipe_call(self, *args, **kwargs) -> Any:
        """
        Wrap the pipeline call to apply img2img refinement to the output.

        Runs the original call, then takes the output images and processes
        them via the refiner pipeline using its img2img capability. Handles
        multiple output images if generated. Selectively forwards relevant
        arguments to the refiner.

        Parameters
        ----------
        *args : tuple
            Positional arguments for the original pipeline call (e.g., prompt).
        **kwargs : dict
            Keyword arguments for the original pipeline call.

        Returns
        -------
        Any
            The pipeline output containing images refined via img2img.
        """
        # Execute the original call (e.g., text-to-image)
        output = self.original_pipe_call(*args, **kwargs)

        # --- Refinement Step ---
        # Check if output has images and is not None
        if output is None or not hasattr(output, "images") or not output.images:
            return output  # Return original output if no images

        # Disable cache helper if it exists during refinement
        if hasattr(self.model, "cache_helper") and hasattr(self.model.cache_helper, "disable"):
            self.model.cache_helper.disable()

        refined_images = []

        kwargs.pop("num_images_per_prompt", None)
        # Process each image individually
        for img in output.images:
            # Ensure image is on the correct device/format if necessary (often handled by pipeline)
            refined_output_single = self.refiner(image=img, strength=self.strength, *args, **kwargs)
            if refined_output_single is not None and hasattr(refined_output_single, "images"):
                refined_images.extend(refined_output_single.images)
            else:
                # Handle cases where refinement might fail for a single image
                refined_images.append(img)  # Keep original if refinement fails

        # Re-enable cache helper if it exists
        if hasattr(self.model, "cache_helper") and hasattr(self.model.cache_helper, "enable"):
            self.model.cache_helper.enable()

        # Replace original images with refined ones in the output object
        output.images = refined_images

        return output

    def enable(self) -> None:
        """Enable the img2img refinement by replacing the pipeline call."""
        self.model.__call__ = self._wrapped_pipe_call

    def disable(self) -> None:
        """Disable refinement by restoring the original pipeline call."""
        self.model.__call__ = self.original_pipe_call

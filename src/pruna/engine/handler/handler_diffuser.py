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
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch

from pruna.engine.handler.handler_inference import InferenceHandler, validate_seed_strategy
from pruna.logging.logger import pruna_logger


class DiffuserHandler(InferenceHandler):
    """
    Handle inference arguments, inputs and outputs for diffusers models.

    A generator with a fixed seed (42) is passed as an argument to the model for reproducibility.
    The first element of the batch is passed as input to the model.
    The generated outputs are expected to have .images attribute.

    Parameters
    ----------
    call_signature : inspect.Signature
        The signature of the call to the model.
    model_args : Dict[str, Any]
        The arguments to pass to the model.
    """

    def __init__(self, call_signature: inspect.Signature, model_args: Optional[Dict[str, Any]] = None) -> None:
        self.call_signature = call_signature
        self.model_args = model_args if model_args else {}
        self.model_args["output_type"] = "pt"

    def prepare_inputs(
        self, batch: List[str] | torch.Tensor | Tuple[List[str] | torch.Tensor | dict[str, Any], ...] | dict[str, Any]
    ) -> Any:
        """
        Prepare the inputs for the model.

        Parameters
        ----------
        batch : List[str] | torch.Tensor | Tuple[List[str] | torch.Tensor | dict[str, Any], ...] | dict[str, Any]
            The batch to prepare the inputs for.

        Returns
        -------
        Any
            The prepared inputs.
        """
        if "prompt" in self.call_signature.parameters or "args" in self.call_signature.parameters:
            x, _ = batch
            return x
        else:  # Unconditional generation models
            return None

    def apply_per_sample_seed(self) -> None:
        """Generate and apply a new random seed derived from global_seed (only valid if seed_strategy="per_sample")."""
        if self.seed_strategy != "per_sample":
            raise ValueError("Seed strategy must be 'per_sample' to apply per sample seed.")
        seed = int(torch.randint(0, 2**31, (1,), generator=self.generator).item())
        self.model_args["generator"] = torch.Generator("cpu").manual_seed(seed)

    def process_output(self, output: Any) -> torch.Tensor:
        """
        Handle the output of the model.

        Parameters
        ----------
        output : Any
            The output to process.

        Returns
        -------
        torch.Tensor
            The processed images.
        """
        if hasattr(output, "images"):
            generated = output.images
        elif hasattr(output, "frames"):
            generated = output.frames
        else:
            # Maybe the user is calling the pipeline with return_dict = False,
            # which then directly returns the generated image / video.
            generated = output
        return generated

    def log_model_info(self) -> None:
        """Log information about the inference handler."""
        pruna_logger.info(
            "Detected diffusers model. Using DiffuserHandler.\n- The first element of the batch is passed as input.\n"
        )

    def configure_seed(
        self, seed_strategy: Literal["per_evaluation", "per_sample", "no_seed"], global_seed: int | None
    ) -> None:
        """
        Set the random seed according to the chosen strategy.

        - If `seed_strategy="per_evaluation"`, the same `global_seed` is applied
        once and reused for the entire generation run.
        - If `seed_strategy="per_sample"`, the `global_seed` is used as a base to derive a different seed for each sample
        This ensures reproducibility while still producing variation across samples,
        making it the preferred option for benchmarking.
        - If `seed_strategy="no_seed"`, no seed is set internally.
        The user is responsible for managing seeds if reproducibility is required.

        Parameters
        ----------
        seed_strategy : Literal["per_evaluation", "per_sample", "no_seed"]
            The seeding strategy to apply.
        global_seed : int | None
            The base seed value to use (if applicable).
        """
        self.seed_strategy = seed_strategy
        validate_seed_strategy(seed_strategy, global_seed)
        if global_seed is not None:
            self.global_seed = global_seed
            self.generator = torch.Generator("cpu").manual_seed(global_seed)
            # We also set the generator for the per_evaluation seed strategy already here.
            self.model_args["generator"] = torch.Generator("cpu").manual_seed(global_seed)

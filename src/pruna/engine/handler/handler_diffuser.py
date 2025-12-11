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

from pruna.engine.handler.handler_inference import InferenceHandler
from pruna.logging.logger import pruna_logger


class DiffuserHandler(InferenceHandler):
    """
    Handle inference arguments, inputs and outputs for diffusers models.

    Parameters
    ----------
    call_signature : inspect.Signature
        The signature of the call to the model.
    model_args : Dict[str, Any]
        The arguments to pass to the model.
    seed_strategy : Literal ["per_sample", "no_seed"]
        The strategy to use for the seed.
    global_seed : int | None
        The global seed to use for the model.
    """

    def __init__(
        self,
        call_signature: inspect.Signature,
        model_args: Optional[Dict[str, Any]] = None,
        seed_strategy: Literal["per_sample", "no_seed"] = "no_seed",
        global_seed: int | None = None,
    ) -> None:
        self.call_signature = call_signature
        self.model_args = model_args if model_args else {}
        # We want the default output type to be pytorch tensors.
        self.model_args["output_type"] = "pt"
        self.configure_seed(seed_strategy, global_seed)

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
        #  For video models.
        elif hasattr(output, "frames"):
            generated = output.frames
        else:
            # Maybe the user is calling the pipeline with return_dict = False,
            # which then returns the generated image / video in a tuple
            generated = output[0]
        return generated.float()

    def log_model_info(self) -> None:
        """Log information about the inference handler."""
        pruna_logger.info(
            "Detected diffusers model. Using DiffuserHandler.\n- The first element of the batch is passed as input.\n"
            "Inference outputs are expected to have either have an `images` attribute or a `frames` attribute."
            "Or be a tuple with the generated image / video as the first element."
        )

    def set_seed(self, seed: int) -> None:
        """
        Set the random seed for the current process.

        Parameters
        ----------
        seed : int
            The seed to set.
        """
        self.model_args["generator"] = torch.Generator("cpu").manual_seed(seed)

    def remove_seed(self) -> None:
        """Remove the seed from the current process."""
        self.model_args["generator"] = None

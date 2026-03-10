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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from pruna.engine.handler.handler_inference import InferenceHandler
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
        default_args = {"generator": torch.Generator("cpu").manual_seed(42)}
        self.call_signature = call_signature
        if model_args:
            default_args.update(model_args)
        self.model_args = default_args

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
        generated = getattr(output, "images", None) or getattr(output, "frames", None)
        if generated is None:
            raise AttributeError(f"{type(output).__name__} has neither 'images' nor 'frames' attribute.")
        # Video pipelines return nested lists of PIL images; flatten to a single list.
        if isinstance(generated, list) and generated and isinstance(generated[0], list):
            generated = [frame for clip in generated for frame in clip]
        to_tensor = transforms.PILToTensor()
        tensors = []
        for g in generated:
            if isinstance(g, np.ndarray):
                g = Image.fromarray(g)
            tensors.append(to_tensor(g))
        return torch.stack(tensors)

    def log_model_info(self) -> None:
        """Log information about the inference handler."""
        pruna_logger.info(
            "Detected diffusers model. Using DiffuserHandler with fixed seed.\n"
            "- The first element of the batch is passed as input.\n"
            "- The generated outputs are expected to have .images attribute."
        )

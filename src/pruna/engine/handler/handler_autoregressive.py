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

from typing import Any, Dict, List, Optional, Tuple, cast

import torch
import torchvision.transforms as transforms
import transformers
from transformers.feature_extraction_utils import BatchFeature

from pruna.config.smash_config import SmashConfig
from pruna.engine.handler.handler_inference import InferenceHandler
from pruna.logging.logger import pruna_logger


class AutoregressiveHandler(InferenceHandler):
    """
    Handle inference arguments, inputs and outputs for autoregressive models.

    Autoregressive models can switch between text and image generation modes.
    We assume this was set as a model argument before calling inference.

    This inferencer is based on Janus and should be adapted for other models.

    Parameters
    ----------
    model_args : Dict[str, Any]
        The arguments to pass to the model.
    """

    inference_function_name: str = "generate"

    def __init__(self, model: Any, smash_config: SmashConfig, model_args: Optional[Dict[str, Any]] = None) -> None:
        self.model = model
        self.smash_config = smash_config
        self.model_args = model_args if model_args else {}

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
        # make sure the generation mode is set as model argument
        if "generation_mode" not in self.model_args:
            raise ValueError("Generation mode must be set as a model argument")
        generation_mode = self.model_args["generation_mode"]
        if generation_mode not in ["image"]:
            raise ValueError("Unrecognized generation mode: expected 'image', other modes are not supported yet")

        if self.smash_config.processor is None:
            raise ValueError(
                "Processor must be set in order to use image generation. "
                "The SmashConfig did not contain a processor, you can set it by hand with "
                "`model.inference_handler.processor = ...`"
            )
        # recognize JanusProcessor by its class name because it requires transformers>=4.52.0
        elif self.smash_config.processor.__class__.__name__ != "JanusProcessor":
            raise ValueError(f"Expected a JanusProcessor, but got {type(self.smash_config.processor)}")

        # processor is a JanusProcessor, meaning transformers>=4.52.0, so we can now us it for type casting
        processor = cast(transformers.JanusProcessor, self.smash_config.processor)

        # image generation mode
        text, _ = batch
        text = cast(str, text)
        inputs = processor(text=text, generation_mode=generation_mode, return_tensors="pt")
        inputs = inputs.to(self.smash_config.device, dtype=self.model.dtype)
        if isinstance(inputs, BatchFeature):
            return dict(inputs)
        return inputs

    def process_output(self, output: Any) -> Any:
        """
        Handle the output of the model.

        Parameters
        ----------
        output : Any
            The output to process.

        Returns
        -------
        Any
            The processed output.
        """
        if "generation_mode" not in self.model_args:
            raise ValueError("Generation mode must be set as a model argument")
        generation_mode = self.model_args["generation_mode"]
        if generation_mode not in ["image"]:
            raise ValueError("Unrecognized generation mode: expected 'image', other modes are not supported yet")

        # image generation mode
        decoded_image = self.model.decode_image_tokens(output)
        processor = cast(transformers.JanusProcessor, self.smash_config.processor)
        # return_tensors="pt" leads to [width, height, 3] images instead of [3, width, height]
        # processing to PIL for consistency
        images = processor.postprocess(list(decoded_image.float()), return_tensors="PIL.Image.Image")
        images = torch.stack([transforms.PILToTensor()(img) for img in images["pixel_values"]])
        return images

    def log_model_info(self) -> None:
        """Log information about the inference handler."""
        pruna_logger.info(
            "Detected Janus model. Using AutoregressiveHandler.\n"
            "- The first element of the batch is passed as text input.\n"
            "- The generated outputs are expected to have pixel_values attribute."
        )

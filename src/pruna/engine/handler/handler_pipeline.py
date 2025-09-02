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

from typing import Any, Dict, List, Optional, Tuple

import torch

from pruna.engine.handler.handler_inference import InferenceHandler
from pruna.logging.logger import pruna_logger


class PipelineHandler(InferenceHandler):
    """
    Handle inference arguments, inputs and outputs for transformer pipelines.

    This handler is specifically designed for transformers pipelines that expect
    string inputs but receive tokenized tensor data from the evaluation pipeline.
    It converts tensor input_ids back to strings using the pipeline's tokenizer.

    Parameters
    ----------
    pipeline : Any
        The pipeline object to extract tokenizer from.
    model_args : Dict[str, Any]
        The arguments to pass to the model.
    """

    def __init__(self, pipeline: Any = None, model_args: Optional[Dict[str, Any]] = None) -> None:
        self.model_args = model_args if model_args else {}
        self.pipeline = pipeline
        self.tokenizer = getattr(pipeline, "tokenizer", None) if pipeline else None

        # Patch the pipeline's __call__ method to return logits for evaluation contexts
        if pipeline is not None:
            self._patch_pipeline_call(pipeline)

    def _patch_pipeline_call(self, pipeline: Any) -> None:
        """
        Patch the pipeline's __call__ method to return logits for evaluation contexts.

        When the input is tokenized tensors (evaluation context), bypass the pipeline's
        text generation and return raw logits needed for perplexity calculation.

        Parameters
        ----------
        pipeline : Any
            The pipeline whose __call__ method to patch.
        """
        original_call = pipeline.__call__

        def patched_call(*args, **kwargs):
            inputs = args[0] if len(args) > 0 else kwargs.get("inputs", kwargs.get("text_inputs"))

            # If input is tensor, return logits from the underlying model forward
            if hasattr(inputs, "shape") and hasattr(inputs, "dtype"):
                try:
                    if hasattr(pipeline, "model"):
                        with torch.no_grad():
                            outputs = pipeline.model(input_ids=inputs)
                        return outputs.logits if hasattr(outputs, "logits") else outputs
                except Exception as e:
                    pruna_logger.warning(f"Failed to get logits from model forward pass: {e}")
                    # Fallback to original pipeline behavior
                    pass

            return original_call(*args, **kwargs)

        pipeline.__call__ = patched_call

    def prepare_inputs(
        self, batch: List[str] | torch.Tensor | Tuple[List[str] | torch.Tensor | dict[str, Any], ...] | dict[str, Any]
    ) -> Any:
        """
        Prepare the inputs for the pipeline.

        For text generation pipelines, string inputs are passed through. For evaluation
        contexts (tensor token ids), tensors are passed through so the patched pipeline
        can return logits for metrics like perplexity.

        Parameters
        ----------
        batch : List[str] | torch.Tensor | Tuple[List[str] | torch.Tensor | dict[str, Any], ...] | dict[str, Any]
            The batch to prepare the inputs for.

        Returns
        -------
        Any
            The prepared inputs (tensors for evaluation, strings for normal generation).
        """
        x, _ = batch

        return x

    def process_output(self, output: Any) -> Any:
        """
        Handle the output of the pipeline.

        Normalize common pipeline outputs. Returns generated text when available.

        Parameters
        ----------
        output : Any
            The output to process.

        Returns
        -------
        Any
            The processed output - pass through since patched __call__ handles the logic.
        """
        # HuggingFace text-generation pipeline returns list[dict]
        if (
            isinstance(output, list)
            and len(output) > 0
            and isinstance(output[0], dict)
            and "generated_text" in output[0]
        ):
            return [o["generated_text"] for o in output]
        return output

    def log_model_info(self) -> None:
        """Log information about the inference handler."""
        pruna_logger.info(
            "Detected transformers pipeline. Using PipelineHandler.\n"
            "- Token ids will be decoded to strings for pipeline processing.\n"
            "- Pipeline outputs will be normalized to generated text when applicable."
        )

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

        # Patch the pipeline's model generate method if it exists to handle generation_config properly
        if pipeline and hasattr(pipeline, "model") and hasattr(pipeline.model, "generate"):
            self._patch_generate_method(pipeline.model)

        # Also patch the pipeline's __call__ method to handle evaluation contexts
        if pipeline:
            self._patch_pipeline_call(pipeline)

    def _patch_generate_method(self, model: Any) -> None:
        """
        Patch the model's generate method to extract parameters from generation_config.

        This handles the case where CausalLMGenerator expects max_new_tokens as a direct
        parameter but the pipeline passes it inside generation_config.

        Parameters
        ----------
        model : Any
            The model whose generate method to patch.
        """
        # Store the original generate method
        original_generate = model.generate

        def patched_generate(*args, **kwargs):
            # Extract generation_config if present
            generation_config = kwargs.get("generation_config")

            if generation_config:
                # Extract max_new_tokens from generation_config if not directly provided
                if "max_new_tokens" not in kwargs and hasattr(generation_config, "max_new_tokens"):
                    kwargs["max_new_tokens"] = generation_config.max_new_tokens

                # Extract other parameters that CausalLMGenerator might need
                if "temperature" not in kwargs and hasattr(generation_config, "temperature"):
                    kwargs["temperature"] = generation_config.temperature

                if "top_k" not in kwargs and hasattr(generation_config, "top_k"):
                    kwargs["top_k"] = generation_config.top_k

            # Call the original generate method with the extracted parameters
            return original_generate(*args, **kwargs)

        # Replace the generate method with our patched version
        model.generate = patched_generate

    def _patch_pipeline_call(self, pipeline: Any) -> None:
        """
        Patch the pipeline's __call__ method to return logits for evaluation contexts.

        When the input is tokenized tensors (evaluation context), we bypass the pipeline's
        text generation and return raw logits needed for perplexity calculation.

        Parameters
        ----------
        pipeline : Any
            The pipeline whose __call__ method to patch.
        """
        # Store the original __call__ method
        original_call = pipeline.__call__

        def patched_call(*args, **kwargs):
            # Check if we're being called with string inputs (normal generation) or tensor inputs (evaluation context)
            inputs = args[0] if len(args) > 0 else kwargs.get("inputs", kwargs.get("text_inputs"))

            # If input is tensor or we detect evaluation context, return logits
            if hasattr(inputs, "shape") and hasattr(inputs, "dtype"):
                # This is a tensor input - likely from evaluation pipeline
                # Use the model's forward pass to get logits instead of text generation
                try:
                    # Prepare inputs for the model
                    if hasattr(pipeline, "model") and hasattr(inputs, "to"):
                        device = next(pipeline.model.parameters()).device
                        inputs = inputs.to(device)

                        # Call the model's forward method directly to get logits
                        with torch.no_grad():
                            outputs = pipeline.model(input_ids=inputs)

                        # Return logits in the format expected by perplexity metric
                        if hasattr(outputs, "logits"):
                            return outputs.logits
                        else:
                            return outputs

                except Exception as e:
                    pruna_logger.warning(f"Failed to get logits from model forward pass: {e}")
                    # Fallback to original pipeline behavior
                    pass

            # For string inputs or fallback, use the original pipeline behavior
            return original_call(*args, **kwargs)

        # Replace the pipeline's __call__ method with our patched version
        pipeline.__call__ = patched_call

    def prepare_inputs(
        self, batch: List[str] | torch.Tensor | Tuple[List[str] | torch.Tensor | dict[str, Any], ...] | dict[str, Any]
    ) -> Any:
        """
        Prepare the inputs for the pipeline.

        For text generation pipelines, this normally converts tokenized tensors back to strings.
        However, for evaluation contexts (like perplexity), we keep tensors as-is since
        the patched pipeline will handle them directly.

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

        # If x is already a string or list of strings, return as is
        if isinstance(x, (str, list)) and all(isinstance(item, str) for item in (x if isinstance(x, list) else [x])):
            return x

        # If x is a tensor, we need to decide whether to convert to strings or keep as tensor
        if isinstance(x, torch.Tensor):
            # For evaluation contexts, we keep tensors as-is so the patched __call__
            # can return logits instead of generated text
            # The patched __call__ method will detect tensor inputs and handle accordingly
            return x

        # If no tokenizer available or x is not a tensor, return as-is
        return x

    def process_output(self, output: Any) -> Any:
        """
        Handle the output of the pipeline.

        With our patched pipeline, the output is either:
        - Logits tensor (for evaluation with tensor inputs)
        - Generated text (for normal generation with string inputs)

        Parameters
        ----------
        output : Any
            The output to process.

        Returns
        -------
        Any
            The processed output - pass through since patched __call__ handles the logic.
        """
        # The patched pipeline __call__ method already handles the tensor vs string logic
        # So we just pass through the output as-is
        return output

    def log_model_info(self) -> None:
        """Log information about the inference handler."""
        pruna_logger.info(
            "Detected transformers pipeline. Using PipelineHandler.\n"
            "- Tensor inputs will be converted to strings for pipeline processing.\n"
            "- Pipeline outputs will be processed to extract generated text."
        )

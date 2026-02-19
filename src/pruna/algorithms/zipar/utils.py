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

import copy
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers.generation.configuration_utils import GenerationMode
from transformers.generation.logits_process import (
    ClassifierFreeGuidanceLogitsProcessor,
    LogitsProcessorList,
)
from transformers.generation.utils import GenerateDecoderOnlyOutput

from pruna.logging.logger import pruna_logger


class ZipARGenerationGrid:
    """
    Configuration and state tracking for ZipAR fast decoding.

    Parameters
    ----------
    num_rows : int
        Number of rows in the grid.
    num_cols : int
        Number of columns in the grid.
    window_size : int
        Size of the sliding window for generation.
    dtype : torch.dtype
        The compute dtype.
    model_kwargs : dict
        Model keyword arguments.
    """

    def __init__(self, num_rows: int, num_cols: int, window_size: int, dtype: torch.dtype, model_kwargs: dict):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.window_size = window_size

        # keep the mask for the prompt, converting the 1s to 0s and the 0s to -inf
        # it will be used to explicitely define the the causal attention mask during generation
        self.prompt_mask = (1 - model_kwargs["attention_mask"]).to(dtype) * -torch.finfo(dtype).max

        # actual prompt length for each input in the batch (unconditional + conditional)
        self.prompt_lengths = model_kwargs["attention_mask"].sum(dim=1)

        # Decoding head tracking
        self.per_row_token_count = [0] * self.num_rows
        self.active_rows: List[int] = [0]
        self.completed_rows_needing_cache_update: List[int] = []
        self.completed_rows: List[int] = []

    def __len__(self) -> int:
        """Return the total number of positions in the grid."""
        return self.num_rows * self.num_cols

    def get_proxy_token_idx_for_new_row(self) -> int:
        """
        Return the index of the token in the last column of the last completed row.

        Returns
        -------
        int
            The index of the token in the last column of the last completed row.
        """
        # We get its index from the number of completed rows, i.e. the index of the first active row
        return self.active_rows[0] * self.num_cols - 1

    @property
    def total_tokens_generated(self) -> int:
        """Total number of tokens generated so far."""
        return sum(self.per_row_token_count)

    @property
    def is_complete(self) -> bool:
        """Check if all tokens have been generated."""
        return self.total_tokens_generated == len(self)

    def update_grid(self) -> None:
        """Update the grid by moving to the next generation step."""
        if self.is_complete:
            raise ValueError("Grid is complete, cannot update grid")

        # the first row is a special case, it should be generated first
        if self.active_rows[0] == 0:  # the first row is under generation
            self.per_row_token_count[0] += 1
            if self.per_row_token_count[0] == self.num_cols:  # the first row is completed
                self.completed_rows.append(0)
                self.active_rows.pop(0)
                self.active_rows.append(1)
            return

        # increment the token count for each active row
        for row in self.active_rows:
            self.per_row_token_count[row] += 1

        # Detect when to introduce a new decoding head
        last_active_row = self.active_rows[-1]
        has_last_decoded_reached_window = self.per_row_token_count[last_active_row] == self.window_size
        is_last_row = last_active_row == self.num_rows - 1
        if has_last_decoded_reached_window and not is_last_row:
            self.active_rows.append(last_active_row + 1)

        # Clear decoding heads for previously completed rows that have already gone through a generate step
        if self.completed_rows_needing_cache_update:
            self.completed_rows.append(self.completed_rows_needing_cache_update.pop(0))

        # Detect when to remove a decoding head: active_rows remains in the order we added the decoding heads
        # so the first one is the most advanced and therefore the only one we need to check
        if self.per_row_token_count[self.active_rows[0]] >= self.num_cols:
            # we store it separately to update the static cache for the last column of this row
            self.completed_rows_needing_cache_update.append(self.active_rows.pop(0))

    def get_model_kwargs_for_parallel_generation(
        self, batch_size: int, input_length: int, model_kwargs: dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return cache_positions, position_ids and attention_mask tensors.

        This method replaces the role of `_update_model_kwargs_for_generation` in the base Janus model.
        It computes the positions and attention masks based on the current state of the grid, to generate
        multiple tokens at once at the returned positions.

        Parameters
        ----------
        batch_size : int
            Number of batches.
        input_length : int
            Length of input sequence.
        model_kwargs : dict
            The model kwargs.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of (cache_positions, position_ids, attention_mask) tensors.
        """
        # we get both the active decoding heads, and those that need to go through the LM to update the static cache
        # completed ones first so they are in increasing row order, and to pop the completed ones after the LM has run
        decoding_heads = self.completed_rows_needing_cache_update + self.active_rows

        # We have a cache position for each decoding head. Cache positions are 1D, position_ids are 2D
        # For a single prompt, they are the same, but the position_ids are repeated along the batch dimension
        # For multiple prompts, they differ, as position_ids is adapted based on the real (without padding) prompt length
        cache_positions = torch.zeros(len(decoding_heads), dtype=torch.long)
        position_ids = torch.zeros(batch_size, len(decoding_heads), dtype=torch.long)

        # we build the attention mask, of shape (batch_size, 1, num_decoding_heads, max_cache_len),
        # where the second dimension stands for the head dimension
        # we initialize it with -inf, i.e. we do not attend to any position in the cache
        max_cache_len = model_kwargs["past_key_values"].max_cache_len
        dtype = self.prompt_mask.dtype
        attention_mask = torch.full(
            (batch_size, 1, len(decoding_heads), max_cache_len), -torch.finfo(dtype).max, dtype=dtype
        )
        # update the attention mask, so that we attend to the prompt, i.e. we attend to the first input_length positions
        attention_mask[:, :, :, :input_length] = self.prompt_mask.unsqueeze(1).unsqueeze(1)

        for i, row in enumerate(decoding_heads):
            # compute the position within the image, without the prompt
            local_position = row * self.num_cols + self.per_row_token_count[row]

            # the position within the image, taking into account the (longest) prompt
            global_position = input_length + local_position

            cache_positions[i] = global_position
            position_ids[:, i] = (
                self.prompt_lengths + local_position
            )  # the position within the image, with respect to the actual prompt length

            # it should attend all the positions, including the current position
            attention_mask[:, :, i, input_length : global_position + 1] = 0

        return cache_positions, position_ids, attention_mask

    def get_hidden_states(self, outputs: Any) -> torch.Tensor:
        """
        Get the hidden states from the outputs.

        Parameters
        ----------
        outputs : Any
            The outputs from the model.

        Returns
        -------
        torch.Tensor
            The hidden states.
        """
        if self.total_tokens_generated == 0:
            # last state [batch_size * 2, tokens_from_prompt, hidden_state_dim], get last token into expected shape
            hidden_states = outputs.last_hidden_state[:, -1, :].unsqueeze(1).clone()
        elif self.completed_rows_needing_cache_update:
            # this completed row went through the language model and updated the static cache
            assert len(self.completed_rows_needing_cache_update) == 1
            # last state [batch_size * 2, 1 + num_active_decoding_heads, hidden_state_dim]
            hidden_states = outputs.last_hidden_state[:, 1:, :].clone()
        else:
            # last state [batch_size * 2, num_decoding_heads, hidden_state_dim]
            hidden_states = outputs.last_hidden_state.clone()

        return hidden_states

    def place_tokens_in_image(self, next_tokens: torch.Tensor, generated_tokens: torch.Tensor) -> torch.Tensor:
        """
        Modify the generated tokens tensor, assigning the next tokens to the image.

        Parameters
        ----------
        next_tokens : torch.Tensor
            The next tokens to assign to the image.
        generated_tokens : torch.Tensor
            The generated tokens tensor.

        Returns
        -------
        torch.Tensor
            The generated tokens tensor, with the next tokens assigned to the image.
        """
        for i, active_row in enumerate(self.active_rows):
            vector_position = active_row * self.num_cols + self.per_row_token_count[active_row]
            generated_tokens[:, vector_position] = next_tokens[:, i]
        return generated_tokens

    def handle_next_tokens(self, next_tokens: torch.Tensor, generated_tokens: torch.Tensor) -> torch.Tensor:
        """
        Update the next tokens by introducing a proxy token when introducing a new decoding head.

        Parameters
        ----------
        next_tokens : torch.Tensor
            The next tokens to assign to the image.
        generated_tokens : torch.Tensor
            The generated tokens tensor.

        Returns
        -------
        torch.Tensor
            If the window size has been reached, the next tokens are augmented.
            Otherwise, the next tokens are returned unchanged.
        """
        # if the last decoding head has just started (0 tokens generated), we need to provide a proxy token
        # we exclude the rows 0 and 1 since their start uses the default generation
        start_new_row = (
            self.active_rows and self.active_rows[-1] > 1 and self.per_row_token_count[self.active_rows[-1]] == 0
        )
        if start_new_row:
            # We need a "last token" to pass to the language model in order to start this row
            # The previous row is not yet completed, so we use the most spatially adjacent token
            # as a proxy, following the logic in the ZipAR paper.
            proxy_token_idx = self.get_proxy_token_idx_for_new_row()
            proxy_token = generated_tokens[:, proxy_token_idx].unsqueeze(1)
            # proxy token is placed at the end, corresponding to the new decoding head
            next_tokens = torch.cat([next_tokens, proxy_token], dim=1)
        return next_tokens


class JanusZipARGenerator:
    """
    A class for generating images using a Janus model and ZipAR fast decoding.

    Parameters
    ----------
    model : Any
        The Janus model to use for generation.
    window_size : int
        Size of the sliding window for Z generation.
    """

    def __init__(self, model: Any, window_size: int):
        self.window_size = window_size
        self.model = model
        self.original_generate = self.model.generate
        self.num_image_tokens = self.model.model.vision_model.config.num_image_tokens
        self.row_size = int(np.round(np.sqrt(self.num_image_tokens)))

    def enable(self):
        """Activate ZipAR generation by monkey patching the model's generate method."""
        self.model.generate = self.generate

    def disable(self):
        """Deactivate ZipAR generation by restoring the original generate method."""
        self.model.generate = self.original_generate

    def generate_image_tokens(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        input_tokens: torch.Tensor,
        logits_processor: LogitsProcessorList,
        output_attentions: bool,
        output_hidden_states: bool,
        generation_config: Any,
        **model_kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """
        Generate the image tokens using the ZipAR algorithm.

        This implementation was inspired by the original implementation of ZipAR:
        https://github.com/ThisisBillhe/ZipAR/blob/2a5695ca2525872ac6ee38c9d62be38f0c9e985b/LlamaGen-ZipAR/autoregressive/models/generate_zipar.py.

        Parameters
        ----------
        input_ids : torch.LongTensor
            Input token IDs tensor, without CFG tokens.
        inputs_embeds : torch.Tensor
            Input embeddings tensor.
        input_tokens : torch.Tensor
            Input tokens tensor, with CFG tokens.
        logits_processor : LogitsProcessorList
            The logits processor for the input tokens.
        output_attentions : bool
            Whether to return the attentions.
        output_hidden_states : bool
            Whether to return the hidden states.
        generation_config : Any
            The generation config.
        **model_kwargs : Any
            Additional model keyword arguments.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]
            Tuple of (generated_tokens, scores, hidden_states, outputs).
        """
        generated_tokens = torch.zeros(
            (input_ids.shape[0], self.num_image_tokens), dtype=input_ids.dtype, device=input_ids.device
        )
        generation_grid = ZipARGenerationGrid(
            num_rows=self.row_size,
            num_cols=self.row_size,
            window_size=self.window_size,
            dtype=inputs_embeds.dtype,
            model_kwargs=model_kwargs,
        )

        while not generation_grid.is_complete:
            model_inputs = self.model.prepare_inputs_for_generation(
                inputs_embeds=inputs_embeds, input_ids=input_tokens, **model_kwargs
            )

            outputs = self.model.model.language_model(
                **model_inputs,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            # get the hidden states needed for prediction
            hidden_states = generation_grid.get_hidden_states(outputs)
            scores = self.model.model.generation_head(hidden_states)

            # Apply logits processing (classifier-free guidance, temperature, top_k, etc.)
            processed_scores = logits_processor(input_ids, scores)

            # sample the next tokens
            if generation_config.do_sample:
                probs = torch.softmax(processed_scores, dim=-1)
                # flatten the batch and sequence dimensions, sample, and reshape back
                next_tokens = torch.multinomial(probs.flatten(0, 1), num_samples=1).view(probs.shape[:2])
            else:
                next_tokens = processed_scores.argmax(dim=-1)

            # Place the tokens in the correct position in the generated tokens
            generated_tokens = generation_grid.place_tokens_in_image(next_tokens, generated_tokens)

            # update the grid by moving to the next generation step
            generation_grid.update_grid()

            # add a new token, in the case the window size is reached
            next_tokens = generation_grid.handle_next_tokens(next_tokens, generated_tokens)

            # double the batch size for CFG
            next_tokens_for_embeddings = torch.cat([next_tokens, next_tokens], dim=0)
            inputs_embeds = self.model.prepare_embeddings_for_image_generation(next_tokens_for_embeddings)

            # Update cache positions, position ids and attention mask to generate the next tokens in parallel
            cache_positions, position_ids, attention_mask = generation_grid.get_model_kwargs_for_parallel_generation(
                batch_size=input_tokens.shape[0], input_length=input_tokens.shape[1], model_kwargs=model_kwargs
            )
            model_kwargs["cache_position"] = cache_positions.to(inputs_embeds.device)
            model_kwargs["position_ids"] = position_ids.to(inputs_embeds.device)
            model_kwargs["attention_mask"] = attention_mask.to(inputs_embeds.device)

        return generated_tokens, scores, hidden_states, outputs

    def prepare_logits_processor(self, generation_config, input_ids, device, logits_processor):
        """
        Prepare (and merge) the logits processor.

        Parameters
        ----------
        generation_config : GenerationConfig
            The generation config.
        input_ids : torch.Tensor
            The input token ids that serve as the prompt.
        device : torch.device
            The device to use for the input tokens.
        logits_processor : LogitsProcessorList | None
            The logits processor for the input tokens.

        Returns
        -------
        LogitsProcessorList
            The logits processor.
        """
        # Initialize logit processors
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()

        # 4. Add CFG processor along with user passed logit processor.
        if generation_config.guidance_scale and generation_config.guidance_scale > 1:
            logits_processor.append(ClassifierFreeGuidanceLogitsProcessor(generation_config.guidance_scale))
            generation_config.guidance_scale = None  # Reset to prevent processor duplication.

        # 5. Prepare and merge logits processor
        logits_processor = self.model._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids.shape[1],
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=None,
            logits_processor=logits_processor,
            device=device,
        )

        return logits_processor

    def prepare_inputs_tokens(self, inputs, generation_config, model_kwargs, attention_mask):
        """
        Check inputs shapes, and setup special tokens and model kwargs.

        Parameters
        ----------
        inputs : torch.Tensor
            The input tokens.
        generation_config : GenerationConfig
            The generation config.
        model_kwargs : dict
            The model kwargs.
        attention_mask : torch.Tensor | None
            The attention mask.

        Returns
        -------
        tuple[torch.Tensor, dict, torch.dtype, torch.device]
            The input ids, model kwargs, dtype, and device.
        """
        input_ids, _, model_kwargs = self.model._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        dtype, device = input_ids.dtype, input_ids.device

        if len(input_ids.shape) != 2:
            raise ValueError(
                f"Expected input ids of shape (batch_size, seq_len), but got {input_ids.shape}. "
                "Passing `inputs embeds` is not supported currently."
            )

        # Prepare special tokens which will be used generate internally. Note that we drop attention mask.
        kwargs_has_attention_mask = attention_mask is not None

        self.model._prepare_special_tokens(
            generation_config, kwargs_has_attention_mask=kwargs_has_attention_mask, device=input_ids.device
        )
        # 6. Expand inputs for multiple image generations per prompt.
        input_ids, model_kwargs = self.model._expand_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            expand_size=generation_config.num_return_sequences,
            **model_kwargs,
        )

        return input_ids, model_kwargs, dtype, device

    def prepare_input_and_cache(self, input_ids, model_kwargs, generation_config, device):
        """
        Setup input tokens, mask and cache.

        Prepare the input tokens, inputs embeddings and model_kwargs for ZipAR fast decoding.
        Differs from the original implementation, since we drop attention mask and force the use of static caching.

        Parameters
        ----------
        input_ids : torch.Tensor
            The input token ids that serve as the prompt.
        model_kwargs : dict
            The model kwargs.
        generation_config : GenerationConfig
            The generation config.
        device : torch.device
            The device to use for the input tokens.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, dict]
            The input tokens, inputs embeddings, and model kwargs.
        """
        batch_size, seq_len = input_ids.shape

        # Double batch size for conditional/unconditional logits
        input_tokens = input_ids.repeat(2, 1)
        attention_mask = model_kwargs.pop("attention_mask", None)
        attention_mask = attention_mask.repeat(2, 1)
        model_kwargs["attention_mask"] = attention_mask

        # Mask all the tokens that are neither BOS nor BOI with pad token in the unconditional logits
        mask = (input_tokens[batch_size:, :] != generation_config.bos_token_id) & (
            input_tokens[batch_size:, :] != generation_config.generation_kwargs["boi_token_id"]
        )
        input_tokens[batch_size:, :].masked_fill_(mask, generation_config.pad_token_id)

        inputs_embeds = self.model.get_input_embeddings()(input_tokens)

        model_kwargs = self.model._get_initial_cache_position(seq_len, device, model_kwargs)

        # Force the use of static cache
        user_cache_implementation = getattr(generation_config, "cache_implementation", None)
        if user_cache_implementation is not None and user_cache_implementation != "static":
            pruna_logger.warning(
                f"ZipAR fast decoding requires static caching. User specified '{user_cache_implementation}' "
                "Setting to 'static'."
            )
        # Ignore past_key_values, since we will setup our own cache
        if model_kwargs.get("past_key_values", None) is not None:
            pruna_logger.warning("past_key_values will be ignored for ZipAR fast decoding.")
            model_kwargs.pop("past_key_values")

        generation_config.cache_implementation = "static"

        model_kwargs["past_key_values"] = self.model._get_cache(
            cache_implementation=generation_config.cache_implementation,
            # batch_size should account for both conditional/unconditional input; hence multiplied by 2
            batch_size=batch_size * 2,
            max_cache_len=self.num_image_tokens + seq_len,
            device=device,
            model_kwargs=model_kwargs,
        )

        return input_tokens, inputs_embeds, model_kwargs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        **kwargs: Any,
    ) -> Union[torch.Tensor, GenerateDecoderOnlyOutput]:
        """
        Generate images using the model.

        The code is an adaptation from:
        https://github.com/huggingface/transformers/blob/34133d0a790787739bfc9a42603985de3728ede4/src/transformers/models/janus/modeling_janus.py#L1254.

        Parameters
        ----------
        inputs : Optional[torch.Tensor], optional
            Input tensor, by default None.
        attention_mask : Optional[torch.LongTensor], optional
            Attention mask tensor, by default None.
        logits_processor : Optional[LogitsProcessorList], optional
            Logits processor, by default None.
        **kwargs : Any
            Additional generation parameters.

        Returns
        -------
        Union[torch.Tensor, GenerateDecoderOnlyOutput]
            Generated tokens or output object.
        """
        generation_mode = kwargs.pop("generation_mode", "text")  # original default is "text"
        # ZipAR only works for image generation, so we fall back to the original generate method for text generation
        if generation_mode != "image":
            return self.original_generate(
                inputs=inputs,
                attention_mask=attention_mask,
                logits_processor=logits_processor,
                generation_mode=generation_mode,
                **kwargs,
            )

        # Numbered comments are based on the original implementation.
        # 1. Handle generation config and model kwargs
        generation_config = kwargs.pop("generation_config", self.model.generation_config)
        generation_config = copy.deepcopy(generation_config)

        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs

        # Validate generation mode
        if generation_config.get_generation_mode() not in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            raise ValueError(
                "Got incompatible mode for Image Generation, should be one of greedy or sampling. "
                "Ensure that beam search is de-activated by setting `num_beams=1` and `num_beam_groups=1`."
            )

        # Validate the configuration and model kwargs
        generation_config.validate()
        self.model._validate_model_kwargs(model_kwargs.copy())

        # 2. Skipped: Initialize logit processors (we use a custom function)

        # Set `use_cache=True` as we will be using input embeds for generation.
        model_kwargs["use_cache"] = True

        # Check if guidance_scale is provided.
        if generation_config.guidance_scale is None:
            pruna_logger.warning("`guidance_scale` is required for CFG but not provided. Setting to default value of 5.")
            generation_config.guidance_scale = 5
        model_kwargs["guidance_scale"] = generation_config.guidance_scale

        # 3. & 6. Prepare model inputs (with custom function)
        input_ids, model_kwargs, dtype, device = self.prepare_inputs_tokens(
            inputs, generation_config, model_kwargs, attention_mask
        )

        # 4. & 5. Prepare logits processors
        logits_processor = self.prepare_logits_processor(
            generation_config=generation_config,
            input_ids=input_ids,
            device=device,
            logits_processor=logits_processor,
        )
        # 6. was done with 3.

        # 7. Prepare input and model caches. This differs from the original implementation, since we drop attention mask.
        input_tokens, inputs_embeds, model_kwargs = self.prepare_input_and_cache(
            input_ids, model_kwargs, generation_config, device
        )

        # 8. init attention / hidden states / scores tuples
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate

        raw_scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None

        # Apply ZipAR parallel decoding (rewrites the original token-generating for loop)
        generated_tokens, scores, hidden_state, outputs = self.generate_image_tokens(
            input_ids,
            inputs_embeds,
            input_tokens,
            logits_processor,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            generation_config=generation_config,
            **model_kwargs,
        )

        # Return the results.
        if return_dict_in_generate:
            if output_scores:
                raw_scores = tuple(raw_scores) + (scores,) if raw_scores is not None else (scores,)  # type: ignore
            if output_logits:
                raw_logits = (
                    tuple(raw_logits) + (hidden_state.float(),) if raw_logits is not None else (hidden_state.float(),)  # type: ignore
                )
            if output_attentions:
                decoder_attentions = (
                    tuple(decoder_attentions) + (outputs.attentions,)
                    if decoder_attentions is not None
                    else (outputs.attentions,)
                )
            if output_hidden_states:
                decoder_hidden_states = (
                    tuple(decoder_hidden_states) + (outputs.hidden_states,)
                    if decoder_hidden_states is not None
                    else (outputs.hidden_states,)
                )
            return GenerateDecoderOnlyOutput(
                sequences=generated_tokens,  # type: ignore
                scores=scores,  # type: ignore
                logits=raw_logits,  # type: ignore
                attentions=decoder_attentions,  # type: ignore
                hidden_states=decoder_hidden_states,  # type: ignore
                past_key_values=outputs.past_key_values,
            )
        else:
            return generated_tokens

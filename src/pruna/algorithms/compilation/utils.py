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

import contextlib
import copy
from typing import Optional

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.cache_utils import Cache, StaticCache
from transformers.generation import ClassifierFreeGuidanceLogitsProcessor, GenerationMode, LogitsProcessorList
from transformers.generation.utils import GenerateDecoderOnlyOutput

from pruna.logging.logger import pruna_logger


class CausalLMGenerator:
    """
    A class for generating text using a Hugging Face model, and using torch.compile.

    The code is adapted from # https://gist.github.com/ArthurZucker/5dc54a3fb443e979fac437e5df7c800b
    and https://github.com/mobiusml/hqq/blob/1f052eb5a0aab0572d380d48b708ae1c74936d23/hqq/utils/generation_hf.py.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The Hugging Face model to use for text generation.
    max_kv_cache_size : int
        The maximum size of the key-value cache used during generation.
    temperature : float, default=0.6
        The sampling temperature to use for text generation. Higher values increase randomness.
    top_k : int, default=5
        The number of highest probability vocabulary tokens to keep for top-k filtering.
    compile_mode : str, default='reduce-overhead'
        The compilation mode to use with torch.compile(). Options include 'reduce-overhead', 'max-autotune', etc.
    compile_fullgraph : bool, default=True
        Whether to compile the full computation graph or use partial graph compilation.
    batch_size : int, default=1
        The batch size to use for text generation.
    device : str, default='cuda'
        The device to use for text generation.
    """

    def __init__(
        self,
        model,
        max_kv_cache_size: int,
        temperature: float = 0.6,
        top_k: int = 5,
        compile_mode: str = "reduce-overhead",
        compile_fullgraph: bool = True,
        batch_size: int = 1,
        device: str = "cuda",
    ):
        """
        Initialize the CausalLMGenerator.

        Parameters
        ----------
        model : transformers.PreTrainedModel
            The Hugging Face model to use for text generation.
        max_kv_cache_size : int
            The maximum size of the key-value cache used during generation.
        temperature : float
            The sampling temperature to use for text generation. Higher values increase randomness.
        top_k : int
            The number of highest probability vocabulary tokens to keep for top-k filtering.
        compile_mode : str
            The compilation mode to use with torch.compile(). Options include 'reduce-overhead', 'max-autotune', etc.
        compile_fullgraph : bool
            Whether to compile the full computation graph or use partial graph compilation.
        batch_size : int
            The batch size to use for text generation.

        Returns
        -------
        None
        """
        super().__init__()

        torch._dynamo.config.capture_scalar_outputs = True
        torch._inductor.config.fx_graph_cache = True
        with contextlib.suppress(Exception):
            torch._dynamo.config.inline_inbuilt_nn_modules = False  # torch 2.5.0 fix

        self.model = model
        self.device = device
        self.temperature = temperature
        self.top_k = top_k
        self.use_cache = True
        self.compile_mode = compile_mode
        self.compile_fullgraph = compile_fullgraph
        self.batch_size = batch_size
        self.cache_batch_size = batch_size
        self.cache_size = max_kv_cache_size
        self.eos_token_id = getattr(model.config, "eos_token_id", None)
        if self.eos_token_id is None:
            pruna_logger.warning("Warning: eos_token_id is None. This may affect generation stopping criteria.")

        self.setup_cache()

        self.decode_one_token = torch.compile(  # type: ignore
            self.decode_one_token, mode=self.compile_mode, fullgraph=self.compile_fullgraph
        )

        self.init()

        ############################
        # Cuda Graph section
        self.static_input = torch.zeros((1, 1), device=self.device, dtype=torch.int32)
        self.static_output = torch.zeros((1, 1), device=self.device, dtype=torch.int32)
        self.original_gen_next_token = self.gen_next_token
        self.cuda_graph = None
        self.do_capture_graph = False
        ############################

    @torch.no_grad()
    def setup_cache(self):
        """
        Setup the Static cache for the model.

        Returns
        -------
        None
            This method initializes the static cache for the model but does not return a value.
        """
        self.past_key_values = StaticCache(
            self.model.config, self.batch_size, self.cache_size, self.model.device, self.model.dtype
        )

    @torch.no_grad()
    def reset_cache(self):
        """
        Reset the Static cache for the model.

        Returns
        -------
        None
            This method resets the internal static cache but does not return any value.
        """
        self.past_key_values.reset()

    def init(self) -> None:
        """
        Initialize the model.

        Returns
        -------
        None
            This method initializes the model by setting it to evaluation mode and configuring
            the cache implementation and usage settings.
        """
        self.model.eval()
        self.model.generation_config.cache_implementation = "static"
        self.model.config.use_cache = True

    def multinomial_sample_one_no_sync(self, probs_sort: torch.Tensor) -> torch.Tensor:
        """
        Sample one token from the model.

        Parameters
        ----------
        probs_sort : torch.Tensor
            The probabilities to sample from.

        Returns
        -------
        torch.Tensor
            The sampled token index.
        """
        q = torch.empty_like(probs_sort).exponential_(1)
        return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

    def logits_to_probs(self, logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        """
        Convert logits to probabilities.

        Parameters
        ----------
        logits : torch.Tensor
            The logits to convert.
        temperature : float
            The temperature to use.
        top_k : int | None
            The top-k value to use.

        Returns
        -------
        torch.Tensor
            The probabilities after applying temperature scaling and optional top-k filtering.
        """
        logits = logits / max(temperature, 1e-5)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            pivot = v.select(-1, -1).unsqueeze(-1)
            logits = torch.where(logits < pivot, -float("Inf"), logits)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs

    def sample(
        self, logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample one token from the model.

        Parameters
        ----------
        logits : torch.Tensor
            The logits to sample from.
        temperature : float
            The temperature to use.
        top_k : int | None
            The top-k value to use.

        Returns
        -------
        idx_next : torch.Tensor
            The next token.
        probs : torch.Tensor
            The probabilities.
        """
        probs = self.logits_to_probs(logits[:, -1], temperature, top_k)
        idx_next = self.multinomial_sample_one_no_sync(probs)
        return idx_next, probs

    def decode_one_token(
        self,
        cur_token: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: torch.Tensor,
        temperature: float = 0.6,
        top_k: int = 5,
    ) -> torch.Tensor:
        """
        Decode one token sampled from the model.

        Parameters
        ----------
        cur_token : torch.Tensor
            The current token.
        cache_position : torch.Tensor
            The cache position.
        past_key_values : torch.Tensor
            The past key values.
        temperature : float
            The temperature to use.
        top_k : int
            The top-k value to use.

        Returns
        -------
        torch.Tensor
            The next token sampled from the model.
        """
        # run the model with the current token, cache position, past key values.
        # (kv cache will be updated internally by the model)
        out = self.model(
            cur_token,
            cache_position=cache_position,
            past_key_values=past_key_values,
            return_dict=True,
            use_cache=self.use_cache,
        )
        # get the logits and the past key values from the output.
        logits, self.past_key_values = out.logits, out.past_key_values
        # sample the next token from the logits.
        new_token = self.sample(logits, temperature=temperature, top_k=top_k)[0]
        return new_token

    def setup(self, inputs: torch.Tensor, max_new_tokens: int):
        """
        Setup the inputs for the model.

        Parameters
        ----------
        inputs : torch.Tensor
            The inputs to the model.
        max_new_tokens : int
            The maximum number of new tokens to generate.

        Returns
        -------
        None
            This method initializes internal state for generation but does not return a value.
        """
        new_batch_size = inputs.shape[0]

        # Check if batch size changed compared to the cache configuration
        # Round up max_new_tokens to the nearest 1000 for better memory allocation
        rounded_cache_size = ((inputs.shape[1] + max_new_tokens + 999) // 1000) * 1000
        if new_batch_size != self.cache_batch_size or self.cache_size != rounded_cache_size:
            pruna_logger.info(
                f"Cache size changed from {self.cache_batch_size}x{self.cache_size} to "
                f"{new_batch_size}x{rounded_cache_size}. Re-initializing StaticCache."
            )
            self.batch_size = new_batch_size
            self.cache_batch_size = new_batch_size
            self.cache_size = rounded_cache_size
            self.setup_cache()

            # If CUDA graph was used, recompile the graph
            if hasattr(self, "cuda_graph") and self.cuda_graph is not None:
                pruna_logger.warning(
                    "CUDA graph is invalidated due to batch size or cache size change. Recompiling the graph."
                )
                self.enable_cuda_graph(max_kv_cache_size=self.cache_size)

        # Reset cache contents (does not change shape)
        self.reset_cache()

        self.inputs = inputs
        self.batch_size, self.seq_length = self.inputs.shape
        self.cache_position = torch.arange(self.seq_length, device=self.device)
        # initialize the generated ids with zeros
        self.generated_ids = torch.zeros(
            self.batch_size,
            self.seq_length + max_new_tokens + 1,
            dtype=torch.int,
            device=self.device,
        )
        # copy the input ids to the generated ids
        self.generated_ids[:, self.cache_position] = self.inputs.to(torch.int)

    def prefill(self) -> torch.Tensor:
        """
        Prefill the model.

        Compute the prefill phase of the LLM. No compilation here because it's only run once.

        Returns
        -------
        torch.Tensor
            The next token generated during prefill.
        """
        out = self.model(
            self.inputs,
            cache_position=self.cache_position,
            past_key_values=self.past_key_values,
            return_dict=True,
            use_cache=self.use_cache,
        )
        logits, self.past_key_values = out.logits, out.past_key_values
        next_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
        self.generated_ids[:, self.seq_length] = next_token[:, 0]
        self.cache_position = torch.tensor([self.seq_length], device=self.device, dtype=torch.long)
        self.begin_gen_position = self.cache_position.item()
        return next_token

    def gen_next_token(self, current_token: torch.Tensor) -> torch.Tensor:
        """
        Generate the next token.

        Parameters
        ----------
        current_token : torch.Tensor
            The current token.

        Returns
        -------
        torch.Tensor
            The next token generated by the model.
        """
        next_token = self.decode_one_token(
            current_token.clone(),
            cache_position=self.cache_position + 1,
            past_key_values=self.past_key_values,
            temperature=self.temperature,
            top_k=self.top_k,
        )
        self.cache_position += 1
        self.generated_ids[:, self.cache_position] = next_token.int()
        return next_token

    def enable_cuda_graph(
        self,
        iters: int = 2,
        prompt_tokenized: list[int] = [596, 8830, 315, 6913, 19476, 11, 1778, 439, 279, 12939],
        max_kv_cache_size: int = 1024,
    ) -> None:
        """
        Enable the CUDA graph and capture the graph on random prompt.

        Parameters
        ----------
        iters : int
            The number of iterations to run.
        prompt_tokenized : list[int]
            The prompt tokenized.
        max_kv_cache_size : int
            The maximum KV cache size.

        Returns
        -------
        None
            This method modifies the internal state of the generator to use CUDA graphs
            but does not return any value.
        """
        _ = self.generate(
            torch.tensor(prompt_tokenized, device=self.model.device).unsqueeze(0), max_new_tokens=max_kv_cache_size
        )
        for _ in range(iters):
            # need to reset the graph before capturing it at each iteration
            # to avoid block/thread errors.
            self.do_capture_graph = True
            self.gen_next_token = self.gen_next_token_withgraph  # type: ignore
            _ = self.generate(
                torch.tensor(prompt_tokenized, device=self.model.device).unsqueeze(0), max_new_tokens=max_kv_cache_size
            )

    def gen_next_token_withgraph(self, current_token: torch.Tensor) -> torch.Tensor:
        """
        Generate the next token with the CUDA graph.

        Parameters
        ----------
        current_token : torch.Tensor
            The current token.

        Returns
        -------
        torch.Tensor
            The next token generated using the CUDA graph.
        """
        self.static_input.copy_(current_token)

        if self.do_capture_graph:
            self.cuda_graph = torch.cuda.CUDAGraph()  # type: ignore
            with torch.cuda.graph(self.cuda_graph), sdpa_kernel([SDPBackend.MATH]):
                self.static_output = self.decode_one_token(
                    self.static_input.clone(),
                    cache_position=self.cache_position + 1,
                    past_key_values=self.past_key_values,
                    temperature=self.temperature,
                    top_k=self.top_k,
                )
        else:
            if self.cuda_graph is not None:
                self.cuda_graph.replay()

        self.do_capture_graph = False
        next_token = self.static_output

        self.cache_position += 1
        self.generated_ids[:, self.cache_position] = next_token.int()
        return next_token

    def next_token_iterator(
        self, current_token: torch.Tensor, max_new_tokens: int, cleanup: bool = True
    ) -> torch.Tensor:
        """
        Generate the next token, stopping at max_new_tokens or EOS for each sequence in the batch.

        Parameters
        ----------
        current_token : torch.Tensor
            The current token tensor of shape (batch_size, 1).
        max_new_tokens : int
            The maximum number of new tokens to generate.
        cleanup : bool
            Whether to cleanup the inputs, generated ids, and cache position after generation.

        Returns
        -------
        torch.Tensor
            The generated tokens tensor of shape (batch_size, seq_length + generated_length),
            including the input prompt and potentially EOS tokens. Sequences that finish early
            will have EOS followed by padding (initial zeros).
        """
        # Keep track of sequences that haven't finished yet (encountered EOS)
        # Assumes initial state is unfinished for all sequences in the batch
        unfinished_sequences = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)

        # Loop for a maximum of max_new_tokens - 1 steps (as prefill generates the first)
        for i in range(1, max_new_tokens):
            # Generate the next token for all sequences
            current_token = self.gen_next_token(current_token)  # Updates self.generated_ids internally

            # Check if the generated token is the EOS token for any currently unfinished sequence
            if self.eos_token_id is not None:
                # Check which sequences produced the EOS token THIS step
                # current_token shape is (batch_size, 1), squeeze to (batch_size,)
                # Only consider sequences that were previously unfinished
                finished_this_step = (current_token.squeeze(-1) == self.eos_token_id) & unfinished_sequences
                # Update the overall tracker for unfinished sequences
                unfinished_sequences &= ~finished_this_step

            # Stop generation if all sequences in the batch have finished
            if not unfinished_sequences.any():
                break

        # Determine the actual length generated (up to the current cache position)
        # .item() is safe as cache_position should be a 0-dim tensor
        final_seq_len = self.cache_position.item() + 1
        # Clone the relevant part of generated_ids before potential cleanup
        output_tokens = self.generated_ids[:, : int(final_seq_len)].clone()

        if cleanup:
            # Delete internal state tensors, but not output_tokens which is returned
            del self.inputs, self.generated_ids, self.cache_position
            torch.cuda.empty_cache()

        return output_tokens

    @torch.no_grad()
    def generate(self, *args, **kwargs) -> torch.Tensor:
        """
        Generate tokens using the model.

        Parameters
        ----------
        *args : tuple
            Variable length argument list (not used directly).
        **kwargs : dict
            Keyword arguments dictionary that must contain:
            - input_ids : torch.Tensor
                The input token ids that serve as the prompt.
            - max_new_tokens : int
                The maximum number of new tokens to generate.

        Returns
        -------
        torch.Tensor
            The generated tokens, including the input prompt and potentially an EOS token.
        """
        # Extract parameters from kwargs with defaults from instance variables
        self.temperature = kwargs.pop("temperature", self.temperature)
        self.top_k = kwargs.pop("top_k", self.top_k)
        self.use_cache = kwargs.pop("use_cache", self.use_cache)

        # Log any kwargs that are not explicitly handled
        unhandled_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ["input_ids", "max_new_tokens", "temperature", "top_k", "batch_size"]
        }
        if unhandled_kwargs:
            pruna_logger.warning(f"Unhandled kwargs in generate method: {unhandled_kwargs}")

        # Update instance variables with any provided values
        self.setup(
            inputs=kwargs["input_ids"] if "input_ids" in kwargs else args[0],
            max_new_tokens=kwargs["max_new_tokens"] if "max_new_tokens" in kwargs else args[1],
        )
        return self.next_token_iterator(self.prefill(), kwargs["max_new_tokens"])


class JanusGenerator:
    """
    A class for generating images using a Janus model, and using torch.compile.

    The code is adapted from # https://github.com/huggingface/transformers/blob/4542086db764080c4333beef7b9f4327b4f8ff64/src/transformers/models/janus/modular_janus.py#L1147.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The Hugging Face model to use for text generation.
    temperature : float, default=0.6
        The sampling temperature to use for text generation. Higher values increase randomness.
    top_k : int, default=5
        The number of highest probability vocabulary tokens to keep for top-k filtering.
    compile_mode : str, default='reduce-overhead'
        The compilation mode to use with torch.compile(). Options include 'reduce-overhead', 'max-autotune', etc.
    compile_fullgraph : bool, default=True
        Whether to compile the full computation graph or use partial graph compilation.
    compile_backend : str, default='inductor'
        The backend to use for compilation. Options include 'inductor', 'cudagraphs', etc.
    """

    def __init__(
        self,
        model,
        temperature: float = 0.6,
        top_k: int = 5,
        compile_mode: str = "reduce-overhead",
        compile_fullgraph: bool = True,
        compile_backend: str = "inductor",
    ):
        """
        Initialize the JanusGenerator.

        Parameters
        ----------
        model : transformers.PreTrainedModel
            The Hugging Face model to use for image generation.
        temperature : float
            The sampling temperature to use for image generation. Higher values increase randomness.
        top_k : int
            The number of highest probability vocabulary tokens to keep for top-k filtering.
        compile_mode : str
            The compilation mode to use with torch.compile(). Options include 'reduce-overhead', 'max-autotune', etc.
        compile_fullgraph : bool
            Whether to compile the full computation graph or use partial graph compilation.
        compile_backend : str
            The backend to use for compilation. Options include 'inductor', 'cudagraphs', etc.

        Returns
        -------
        None
        """
        super().__init__()

        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.compile_mode = compile_mode
        self.compile_fullgraph = compile_fullgraph
        self.compile_backend = compile_backend

        self.compiled_language_model = torch.compile(
            self.model.model.language_model,
            mode=self.compile_mode,
            fullgraph=self.compile_fullgraph,
            backend=self.compile_backend,
        )

        self.model.eval()

    def validate_config_and_model_kwargs(self, generation_config, model_kwargs):
        """
        Validate the generation config and model kwargs.

        This function is adapted from the `_validate_model_kwargs` function in the `transformers` library.

        Parameters
        ----------
        generation_config : GenerationConfig
            The generation config.
        model_kwargs : dict
            The model kwargs.
        """
        generation_config.validate()
        self.model._validate_model_kwargs(model_kwargs.copy())

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

        # Add CFG processor along with user passed logit processor.
        if generation_config.guidance_scale and generation_config.guidance_scale > 1:
            logits_processor.append(ClassifierFreeGuidanceLogitsProcessor(generation_config.guidance_scale))
            generation_config.guidance_scale = None  # Reset to prevent processor duplication.

        # Prepare and merge logits processor
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
                f"Expected input ids of shape (batch_size, seq_len), but got {input_ids.shape}"
                "Passing `inputs embeds` is not supported currently."
            )

        # Prepare special tokens which will be used generate internally.
        kwargs_has_attention_mask = attention_mask is not None
        self.model._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=input_ids.device)

        # Expand inputs for multiple image generations per prompt.
        input_ids, model_kwargs = self.model._expand_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            expand_size=generation_config.num_return_sequences,
            **model_kwargs,
        )

        return input_ids, model_kwargs, dtype, device

    def get_initial_cache_position(self, input_ids, model_kwargs):
        """
        Get the initial cache position for the model.

        This function is adapted from the `get_initial_cache_position` function in the `transformers` library.

        Parameters
        ----------
        input_ids : torch.Tensor
            The input token ids that serve as the prompt.
        model_kwargs : dict
            The model kwargs.

        Returns
        -------
        dict
            The model kwargs with the initial cache position.
        """
        # `torch.compile`-friendly `torch.arange` from a shape -- the lines below are equivalent to `torch.arange`
        if "inputs_embeds" in model_kwargs:
            cache_position = torch.ones_like(model_kwargs["inputs_embeds"][0, :, 0], dtype=torch.int64).cumsum(0) - 1
        elif "decoder_inputs_embeds" in model_kwargs:
            cache_position = (
                torch.ones_like(model_kwargs["decoder_inputs_embeds"][0, :, 0], dtype=torch.int64).cumsum(0) - 1
            )
        else:
            cache_position = torch.ones_like(input_ids[0, :], dtype=torch.int64).cumsum(0) - 1

        past_length = 0
        if model_kwargs.get("past_key_values") is not None:
            cache = model_kwargs["past_key_values"]
            past_length = 0
            if not isinstance(cache, Cache):
                past_length = cache[0][0].shape[2]
            elif hasattr(cache, "get_seq_length") and cache.get_seq_length() is not None:
                past_length = cache.get_seq_length()

            cache_position = cache_position[past_length:]

        model_kwargs["cache_position"] = cache_position
        return model_kwargs

    def prepare_input_and_cache(self, input_ids, model_kwargs, attention_mask, generation_config, device):
        """
        Setup input tokens, mask and cache.

        Prepare the input tokens, inputs embeddings, model kwargs, batch size, the number of image tokens,
        and setup the KV cache.

        Parameters
        ----------
        input_ids : torch.Tensor
            The input token ids that serve as the prompt.
        model_kwargs : dict
            The model kwargs.
        attention_mask : torch.Tensor | None
            The attention mask.
        generation_config : GenerationConfig
            The generation config.
        device : torch.device
            The device to use for the input tokens.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, dict, int, int]
            The input tokens, inputs embeddings, model kwargs, batch size, and number of image tokens.
        """
        num_image_tokens = self.model.model.vision_model.config.num_image_tokens
        batch_size, seq_len = input_ids.shape

        input_tokens = input_ids.repeat(2, 1)  # Double batch size for conditional/unconditional logits
        attention_mask = model_kwargs.pop("attention_mask", None)
        attention_mask = attention_mask.repeat(2, 1)  # type: ignore
        model_kwargs["attention_mask"] = attention_mask

        # Mask all the tokens that are neither BOS nor BOI with pad token in the unconditional logits.
        mask = (input_tokens[batch_size:, :] != generation_config.bos_token_id) & (
            input_tokens[batch_size:, :] != generation_config.generation_kwargs["boi_token_id"]
        )
        input_tokens[batch_size:, :].masked_fill_(mask, generation_config.pad_token_id)

        inputs_embeds = self.model.get_input_embeddings()(input_tokens)

        model_kwargs = self.get_initial_cache_position(input_ids, model_kwargs)

        if model_kwargs.get("past_key_values", None) is None:
            # Prepare cache if not provided.
            model_kwargs["past_key_values"] = self.model._get_cache(
                cache_implementation=generation_config.cache_implementation or "static",
                # batch_size should account for both conditional/unconditional input; hence multiplied by 2.
                batch_size=batch_size * 2,
                # we should have at least a cache len of seq_len + num_image_tokens.
                max_cache_len=max(generation_config.max_length, num_image_tokens + seq_len),
                device=device,
                model_kwargs=model_kwargs,
            )

        return input_tokens, inputs_embeds, model_kwargs, batch_size, num_image_tokens

    def loop_over_latent_tokens(
        self,
        input_tokens,
        input_ids,
        model_kwargs,
        num_image_tokens,
        output_attentions,
        output_hidden_states,
        inputs_embeds,
        generated_tokens,
        logits_processor,
        generation_config,
    ):
        """
        Loop over the latent tokens.

        Parameters
        ----------
        input_tokens : torch.Tensor
            The input token ids that serve as the prompt.
        input_ids : torch.Tensor
            The input token ids that serve as the prompt.
        model_kwargs : dict
            The model kwargs.
        num_image_tokens : int
            The number of image tokens.
        output_attentions : bool
            Whether to output attentions.
        output_hidden_states : bool
            Whether to output hidden states.
        inputs_embeds : torch.Tensor
            The input embeddings.
        generated_tokens : torch.Tensor
            The generated tokens.
        logits_processor : LogitsProcessorList
            The logits processor.
        generation_config : GenerationConfig
            The generation config.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            The scores, hidden state, and outputs.
        """
        for i in range(num_image_tokens):
            model_inputs = self.model.prepare_inputs_for_generation(
                inputs_embeds=inputs_embeds, input_ids=input_tokens, **model_kwargs
            )

            model_inputs["attention_mask"] = model_inputs["attention_mask"].to(inputs_embeds.device)
            model_inputs["cache_position"] = model_inputs["cache_position"].to(inputs_embeds.device)

            # Pad attention mask to max length to avoid dynamic shapes error during compilation.
            max_length = model_inputs["past_key_values"].get_max_cache_shape()
            current_length = model_inputs["attention_mask"].shape[1]
            if current_length < max_length:
                padding = torch.zeros(
                    (model_inputs["attention_mask"].shape[0], max_length - current_length),
                    dtype=model_inputs["attention_mask"].dtype,
                    device=model_inputs["attention_mask"].device,
                )
                model_inputs["attention_mask"] = torch.cat([model_inputs["attention_mask"], padding], dim=1)

            # no compilation for the prefill.
            if i == 0:
                outputs = self.model.model.language_model(
                    **model_inputs,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
            # compilation for the decoding phase (one token at a time).
            else:
                outputs = self.compiled_language_model(
                    **model_inputs,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )

            # Update model_kwargs like cache_position for next generation.
            model_kwargs = self.model._update_model_kwargs_for_generation(outputs, model_kwargs)
            hidden_state = outputs.last_hidden_state[:, -1, :].clone()

            # Generate scores using the generation head (Not using above defined LM Head)
            scores = self.model.model.generation_head(hidden_state)
            next_token_scores = logits_processor(input_ids, scores) if logits_processor is not None else scores

            # Sample next token.
            if generation_config.do_sample:
                probs = torch.softmax(next_token_scores, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_token = torch.argmax(next_token_scores, dim=-1)

            generated_tokens[:, i] = next_token

            # Prepare embeddings for the next step.
            next_token = torch.cat([next_token, next_token])
            next_token = next_token.unsqueeze(-1)

            inputs_embeds = self.model.prepare_embeddings_for_image_generation(next_token)

        return scores, hidden_state, outputs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        **kwargs,
    ) -> torch.Tensor | GenerateDecoderOnlyOutput:
        """
        Generate latent tokens using the model.

        Parameters
        ----------
        inputs : torch.Tensor | None
            The input token ids that serve as the prompt.
        attention_mask : torch.LongTensor | None
            The attention mask for the input tokens.
        logits_processor : LogitsProcessorList | None
            The logits processor for the input tokens.
        **kwargs : dict
            Keyword arguments dictionary.

        Returns
        -------
        torch.Tensor | GenerateDecoderOnlyOutput
            The generated latent tokens.
        """
        # Extract parameters from kwargs with defaults from instance variables
        self.temperature = kwargs.pop("temperature", self.temperature)
        self.top_k = kwargs.pop("top_k", self.top_k)

        # 1. Handle generation config and model kwargs
        generation_config = kwargs.pop("generation_config", self.model.generation_config)
        generation_config = copy.deepcopy(generation_config)

        # Default to "text" generation if mode isn't provided
        generation_mode = kwargs.pop("generation_mode", "text")
        if generation_mode == "text":
            # Set guidance_scale=None to prevent running UnbatchedCFG processor.
            return self.model.generate(
                inputs=inputs,
                attention_mask=attention_mask,
                generation_config=generation_config,
                guidance_scale=None,
                **kwargs,
            )

        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs

        # Validate generation mode
        if generation_config.get_generation_mode() not in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            raise ValueError(
                "Got incompatible mode for Image Generation, should be one of greedy or sampling. "
                "Ensure that beam search is de-activated by setting `num_beams=1` and `num_beam_groups=1`."
            )

        # Validate the configuration and model kwargs
        self.validate_config_and_model_kwargs(generation_config, model_kwargs)

        # Set `use_cache=True` as we will be using input embeds for generation.
        model_kwargs["use_cache"] = True

        # Check if guidance_scale is provided.
        if generation_config.guidance_scale is None:
            pruna_logger.warning("`guidance_scale` is required for CFG but not provided. Setting to default value of 5.")
            generation_config.guidance_scale = 5
        model_kwargs["guidance_scale"] = generation_config.guidance_scale

        # 2. Prepare model inputs shapes, and check special tokens.
        input_ids, model_kwargs, dtype, device = self.prepare_inputs_tokens(
            inputs, generation_config, model_kwargs, attention_mask
        )

        # 3. Prepare logits processor
        logits_processor = self.prepare_logits_processor(generation_config, input_ids, device, logits_processor)

        # 4. Prepare input and model caches
        input_tokens, inputs_embeds, model_kwargs, batch_size, num_image_tokens = self.prepare_input_and_cache(
            input_ids,
            model_kwargs,
            attention_mask,
            generation_config,
            device,
        )

        # Placeholder for generated tokens.
        generated_tokens = torch.zeros((batch_size, num_image_tokens), dtype=dtype, device=device)

        # 5. init attention / hidden states / scores tuples
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate

        raw_scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None

        # 6. Loop over the latent tokens.
        scores, hidden_state, outputs = self.loop_over_latent_tokens(
            input_tokens,
            input_ids,
            model_kwargs,
            num_image_tokens,
            output_attentions,
            output_hidden_states,
            inputs_embeds,
            generated_tokens,
            logits_processor,
            generation_config,
        )

        # 7. Return the results.
        if return_dict_in_generate:
            if output_scores:
                raw_scores = tuple(raw_scores) + (scores,) if raw_scores is not None else (scores,)
            if output_logits:
                raw_logits = (
                    tuple(raw_logits) + (hidden_state.float(),) if raw_logits is not None else (hidden_state.float(),)
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

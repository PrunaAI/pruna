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

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.cache_utils import StaticCache

from pruna.logging.logger import pruna_logger


class TransformersGenerator:
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
        Initialize the TransformersGenerator.

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
        if new_batch_size != self.cache_batch_size:
            pruna_logger.info(
                f"Batch size changed from {self.cache_batch_size} to {new_batch_size}. Re-initializing StaticCache."
            )
            self.batch_size = new_batch_size
            self.cache_batch_size = new_batch_size
            self.setup_cache()

            # If CUDA graph was used, it's now invalid
            if hasattr(self, "cuda_graph") and self.cuda_graph is not None:
                pruna_logger.warning("CUDA graph is invalidated due to batch size change. Disabling CUDA graph usage.")
                self.cuda_graph = None
                self.gen_next_token = self.original_gen_next_token
                self.do_capture_graph = False

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
        output_tokens = self.generated_ids[:, :final_seq_len].clone()

        if cleanup:
            # Delete internal state tensors, but not output_tokens which is returned
            del self.inputs, self.generated_ids, self.cache_position
            torch.cuda.empty_cache()

        return output_tokens

    @torch.inference_mode()
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

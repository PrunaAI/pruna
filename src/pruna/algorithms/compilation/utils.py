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
from typing import Callable

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.cache_utils import StaticCache


class HFGenerator:
    """
    A class for generating text using a Hugging Face model, and using torch.compile.

    The code is adapted from # https://gist.github.com/ArthurZucker/5dc54a3fb443e979fac437e5df7c800b
    and https://github.com/mobiusml/hqq/blob/1f052eb5a0aab0572d380d48b708ae1c74936d23/hqq/utils/generation_hf.py.
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
    ):
        """
        Initialize the HFGenerator.

        Parameters
        ----------
        model: The Hugging Face model to use.
        max_kv_cache_size: The maximum size of the KV cache.
        temperature: The temperature to use for the generation.
        top_k: The top-k value to use for the generation.
        compile_mode: The mode to use for the compilation.
        compile_fullgraph: Whether to compile the full graph.
        batch_size: The batch size to use for the generation.

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
        self.device = model.device
        self.temperature = temperature
        self.top_k = top_k
        self.use_cache = True
        self.compile_mode = compile_mode
        self.compile_fullgraph = compile_fullgraph
        self.batch_size = batch_size
        self.decode_one_token = self.decode_one_token_sampled
        self.cache_size = max_kv_cache_size

        self.setup_cache()

        self.is_compiled = False

        self.compile_partial(self.decode_one_token)

        self.init()

        ############################
        # Cuda Graph section
        self.static_input = torch.zeros((1, 1), device=self.device, dtype=torch.int32)
        self.static_output = torch.zeros((1, 1), device=self.device, dtype=torch.int32)
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
        """
        self.past_key_values.reset()

    # Ideally only compile this, but it creates issues with generation: https://github.com/huggingface/transformers/issues/30351
    def compile_partial(self, decode_one_token: Callable):
        """
        Compile the partial function for the model.

        Parameters
        ----------
        decode_one_token : Callable
            Function to compile that takes a token and returns the next predicted token.
            This is typically the model's token decoding function.

        Returns
        -------
        None
            This function modifies the internal state by compiling and setting the decode_one_token
            function but does not return anything.
        """
        self.decode_one_token = torch.compile(decode_one_token, mode=self.compile_mode, fullgraph=self.compile_fullgraph)
        self.is_compiled = True

    def next_multiple(self, val: int) -> int:  # next power of 2
        """
        Get the next multiple of 2 for the given value.

        Parameters
        ----------
        val : int
            The value to get the next multiple of 2 for.

        Returns
        -------
        new_val : int
            The next multiple of 2 for the given value.
        """
        vals = [2**i for i in range(5, 20)]  # [32, 64, ...]
        new_val = vals[[i for i in range(len(vals)) if (vals[i] - val) > 0][0]]
        return new_val

    def init(self) -> None:
        """
        Initialize the model.

        Returns
        -------
        None
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
        idx_next : torch.Tensor
            The next token.
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
        probs : torch.Tensor
            The probabilities.
        """
        logits = logits / max(temperature, 1e-5)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            pivot = v.select(-1, -1).unsqueeze(-1)
            logits = torch.where(logits < pivot, -float("Inf"), logits)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs

    def sample(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int | None = None
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

    def decode_one_token_sampled(
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
        new_token : torch.Tensor
            The next token.
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
        """
        self.reset_cache()
        self.inputs = inputs
        self.batch_size, self.seq_length = self.inputs.shape
        self.cache_position = torch.arange(self.seq_length, device=self.device)
        # initialize the generated ids with zeros of the shape (batch_size, seq_length + max_new_tokens + 1)
        self.generated_ids = torch.zeros(
            self.batch_size,
            self.seq_length + max_new_tokens + 1,
            dtype=torch.int,
            device=self.device,
        )
        # copy the input ids to the generated ids at the cache position.
        self.generated_ids[:, self.cache_position] = self.inputs.to(torch.int)

    def prefill(self) -> torch.Tensor:
        """
        Prefill the model.

        Compute the prefill phase of the LLM. No compilation here because it's only run once.

        Returns
        -------
        next_token : torch.Tensor
            The next token.
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

    def gen_next_token_raw(self, next_token: torch.Tensor) -> torch.Tensor:
        """
        Generate the next token.

        Parameters
        ----------
        next_token : torch.Tensor
            The current token.

        Returns
        -------
        next_token : torch.Tensor
            The next token.
        """
        next_token = self.decode_one_token(
                next_token.clone(),
                cache_position=self.cache_position + 1,
                past_key_values=self.past_key_values,
                temperature=self.temperature,
                top_k=self.top_k,
        )
        self.cache_position += 1
        self.generated_ids[:, self.cache_position] = next_token.int()
        return next_token

    def gen_next_token(self, next_token: torch.Tensor) -> torch.Tensor:
        """
        Generate the next token.

        Parameters
        ----------
        next_token : torch.Tensor
            The current token.

        Returns
        -------
        next_token : torch.Tensor
            The next token.
        """
        return self.gen_next_token_raw(next_token)

    def enable_cuda_graph_(self) -> None:
        """Enable the CUDA graph."""
        self.do_capture_graph = True
        self.gen_next_token = self.gen_next_token_withgraph  # type: ignore

    def enable_cuda_graph(
        self,
        iters: int = 2,
        prompt_tokenized: list[int] = [596, 8830, 315, 6913, 19476, 11, 1778, 439, 279, 12939],
        max_kv_cache_size: int = 1024
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
        """
        _ = self.generate(
            torch.tensor(prompt_tokenized, device=self.model.device).unsqueeze(0),
            max_new_tokens=max_kv_cache_size
        )
        for _ in range(iters):
            self.enable_cuda_graph_()
            _ = self.generate(
                torch.tensor(prompt_tokenized, device=self.model.device).unsqueeze(0),
                max_new_tokens=max_kv_cache_size
            )

    def gen_next_token_withgraph(self, next_token: torch.Tensor) -> torch.Tensor:
        """
        Generate the next token with the CUDA graph.

        Parameters
        ----------
        next_token : torch.Tensor
            The current token.

        Returns
        -------
        next_token : torch.Tensor
            The next token.
        """
        self.static_input.copy_(next_token)

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

    def next_token_iterator(self, next_token: torch.Tensor, max_new_tokens: int, cleanup: bool = True) -> torch.Tensor:
        """
        Generate the next token.

        Parameters
        ----------
        next_token : torch.Tensor
            The current token.
        max_new_tokens : int
            The maximum number of new tokens to generate.
        cleanup : bool
            Whether to cleanup the inputs, generated ids, and cache position.

        Returns
        -------
        output_tokens : torch.Tensor
            The generated tokens.
        """
        for i in range(1, max_new_tokens):
            next_token = self.gen_next_token(next_token)
        output_tokens = self.generated_ids

        if cleanup:
            del self.inputs, self.generated_ids, self.cache_position
            torch.cuda.empty_cache()

        return output_tokens

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100) -> torch.Tensor:
        """
        Generate the tokens.

        Parameters
        ----------
        input_ids : torch.Tensor
            The input ids.
        max_new_tokens : int
            The maximum number of new tokens to generate.

        Returns
        -------
        output_tokens : torch.Tensor
            The generated tokens.
        """
        self.setup(inputs=input_ids, max_new_tokens=max_new_tokens)
        return self.next_token_iterator(self.prefill(), max_new_tokens)

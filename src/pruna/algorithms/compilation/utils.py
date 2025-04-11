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
# This script is inspired from https://github.com/chu-tianxiang/QuIP-for-all
from __future__ import annotations

from typing import Optional

import torch


def multinomial_sample_one_no_sync(probs_sort):
    """
    Does multinomial sampling without a cuda synchronization.

    Parameters:
    ----------
        probs_sort (torch.Tensor): The probabilities to sample from

    Returns:
    -------
        torch.Tensor: The sampled index
    """
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    """
    Convert logits to probabilities.

    Parameters:
    ----------
        logits (torch.Tensor): The logits to convert
        temperature (float): The temperature to use for the softmax
        top_k (int): The number of top k to use for the softmax

    Returns:
    -------
        torch.Tensor: The probabilities
    """
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


@torch.compile
def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    """
    Sample a token from the logits.

    Parameters:
    ----------
        logits (torch.Tensor): The logits to sample from
        temperature (float): The temperature to use for the softmax
        top_k (int): The number of top k to use for the softmax

    Returns:
    -------
        torch.Tensor: The sampled index
        torch.Tensor: The probabilities
    """
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


@torch.no_grad()
def decode_one_token(model, cur_token, past_kv, cache_position, temperature, top_k):
    """
    Decode one token from the model.

    Parameters:
    ----------
        model (torch.nn.Module): The model to decode from
        cur_token (torch.Tensor): The current token to decode from
        past_kv (torch.Tensor): The past key values to decode from
        cache_position (torch.Tensor): The cache position to decode from
        temperature (float): The temperature to use for the softmax
        top_k (int): The number of top k to use for the softmax

    Returns:
    -------
        torch.Tensor: The sampled index
        torch.Tensor: The logits
    """
    logits = model(cur_token, past_key_values=past_kv, cache_position=cache_position)[0]
    new_token = sample(logits, temperature=temperature, top_k=top_k)[0]
    return new_token, logits


def create_generate_fn(model, top_k, temperature, past_kv, compiled_decoding):
    """
    Create a generate function for the model.

    Parameters:
    ----------
        model (torch.nn.Module): The model to generate from
        top_k (int): The number of top k to use for the softmax
        temperature (float): The temperature to use for the softmax
        past_kv (torch.Tensor): The past key values to generate from
        compiled_decoding (torch.compile): The compiled decoding function

    Returns:
    -------
        Callable: The generate function
    """

    @torch.no_grad()
    def generate(input_ids, max_new_tokens):
        """
        Generate a sequence from the model using autoregressive generation.

        This function takes an initial sequence of tokens and generates additional tokens
        one at a time up to max_new_tokens. It uses the model's past key/values cache
        for efficient generation (as transformers does by default) and applies temperature and top-k sampling.
        By now, we only support greedy decoding. Other sampling strategies are left for future work.

        Parameters:
        ----------
            input_ids (torch.Tensor): Initial sequence of token ids to continue from, shape [batch_size, seq_length]
            max_new_tokens (int): Number of new tokens to generate beyond the input sequence

        Returns:
        -------
            torch.Tensor: Full sequence of generated tokens including the input sequence,
                         shape [batch_size, seq_length + max_new_tokens]
        """
        # Get input dimensions
        batch_size, seq_length = input_ids.shape
        # Initialize position tracking for the KV cache
        cache_position = torch.arange(seq_length, device=model.device)
        # Initialize tensor to store full generated sequence
        generated_ids = torch.zeros(batch_size, seq_length + max_new_tokens, dtype=torch.int, device=0)
        # Copy input sequence into generated sequence
        generated_ids[:, cache_position] = input_ids.int()
        # Get initial logits from model using input sequence
        # during this step, the kv cache is updated
        logits = model(input_ids, past_key_values=past_kv, cache_position=cache_position)[0]
        # Sample first new token
        next_token, _ = sample(logits, temperature=temperature, top_k=top_k)
        # Add first generated token to sequence
        generated_ids[:, seq_length] = next_token
        # Update cache position for next token
        cache_position = torch.tensor([seq_length + 1], device=model.device)
        # Main generation loop
        for _ in range(1, max_new_tokens):
            # Generate next token using compiled function
            next_token, logits = compiled_decoding(
                model, next_token.clone(), past_kv, cache_position, temperature, top_k
            )
            # Add new token to sequence
            generated_ids[:, cache_position] = next_token.int()
            cache_position += 1

        return generated_ids

    return generate

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

from typing import Optional

import torch


def multinomial_sample_one_no_sync(probs_sort):
    """
    Does multinomial sampling without a cuda synchronization.

    Args:
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

    Args:
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

    Args:
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
def decode_one_token(model, cur_token, past_kv, cache_position, temperature: float = 1.0, top_k: Optional[int] = None):
    """
    Decode one token from the model.

    Args:
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


@torch.no_grad()
def generate(input_ids, max_new_tokens, model, top_k, temperature, past_kv, compiled_decoding):
    """
    Generate a sequence from the model.

    Args:
        model (torch.nn.Module): The model to generate from
        input_ids (torch.Tensor): The input ids to generate from
        max_new_tokens (int): The maximum number of new tokens to generate
        top_k (int): The number of top k to use for the softmax
        temperature (float): The temperature to use for the softmax
        past_kv (torch.Tensor): The past key values to generate from
        compiled_decoding (torch.compile): The compiled decoding function

    Returns:
    -------
        torch.Tensor: The generated ids
    """
    batch_size, seq_length = input_ids.shape
    cache_position = torch.arange(seq_length, device=model.device)
    generated_ids = torch.zeros(batch_size, seq_length + max_new_tokens, dtype=torch.int, device=0)
    generated_ids[:, cache_position] = input_ids.int()
    logits = model(input_ids, past_key_values=past_kv, cache_position=cache_position)[0]

    next_token, _ = sample(logits, temperature=temperature, top_k=top_k)

    generated_ids[:, seq_length] = next_token

    cache_position = torch.tensor([seq_length + 1], device=model.device)
    for _ in range(1, max_new_tokens):
        next_token, logits = compiled_decoding(model, next_token.clone(), past_kv, cache_position, temperature, top_k)
        generated_ids[:, cache_position] = next_token.int()
        cache_position += 1

    return generated_ids

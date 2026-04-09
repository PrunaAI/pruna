# Copyright 2025 - Pruna AI GmbH. All rights reserved.
#
# Vendored from OneIG-Benchmark (commit 41b49831e79e6dde5323618c164da1c4cf0f699d).

import importlib.metadata
from typing import cast

import torch
from packaging import version
from peft import PeftModel
from torch import nn
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
)
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.utils import logging
from transformers.utils.import_utils import _is_package_available

logger = logging.get_logger(__name__)


def is_transformers_attn_greater_or_equal_4_38() -> bool:
    """Return whether the installed ``transformers`` package is at least 4.38.0.

    Returns:
    -------
        True if ``transformers`` is installed and its version is >= 4.38.0; False otherwise.
    """
    if not _is_package_available("transformers"):
        return False
    return version.parse(importlib.metadata.version("transformers")) >= version.parse("4.38.0")


def is_transformers_attn_greater_or_equal_4_40() -> bool:
    """Return whether the installed ``transformers`` package is at least 4.40.0.

    Returns:
    -------
        True if ``transformers`` is installed and its version is >= 4.40.0; False otherwise.
    """
    if not _is_package_available("transformers"):
        return False
    return version.parse(importlib.metadata.version("transformers")) >= version.parse("4.40.0")


class ModifiedLlamaDecoderLayer(LlamaDecoderLayer):
    """Decoder layer with non-causal self-attention when supported by the attention module."""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        if hasattr(self.self_attn, "is_causal"):
            self.self_attn.is_causal = False


class LlamaBiModel(LlamaModel):
    """Bidirectional Llama backbone for MNTP-style training (transformers >= 4.38)."""

    _no_split_modules = ["ModifiedLlamaDecoderLayer"]

    def __init__(self, config: LlamaConfig):
        if not is_transformers_attn_greater_or_equal_4_38():
            raise ValueError(
                "The current implementation of LlamaBiModel follows modeling_llama.py "
                "of transformers version >= 4.38.0"
            )
        LlamaPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.layers = nn.ModuleList(
            [ModifiedLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()

    def _update_causal_mask(
        self,
        attention_mask,
        input_tensor,
        cache_position,
        past_seen_tokens=None,
        output_attentions=False,
    ):
        attn_impl = getattr(
            self.config, "_attn_implementation", getattr(self.config, "attn_implementation", "eager")
        )
        if attn_impl == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]

        if hasattr(getattr(self.layers[0], "self_attn", {}), "past_key_value"):
            target_length = self.config.max_position_embeddings
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else (
                    cache_position[-1] + 1
                    if not is_transformers_attn_greater_or_equal_4_40()
                    else past_seen_tokens + sequence_length + 1
                )
            )

        causal_mask = torch.zeros((sequence_length, target_length), dtype=dtype, device=device)

        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)

        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
            elif attention_mask.dim() == 4:
                offset = (
                    cache_position[0]
                    if attention_mask.shape[-2] < cache_position[0] + sequence_length
                    else 0
                )
                mask_shape = attention_mask.shape
                mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
                causal_mask[
                    : mask_shape[0],
                    : mask_shape[1],
                    offset : mask_shape[2] + offset,
                    : mask_shape[3],
                ] = mask_slice

        attn_impl = getattr(
            self.config, "_attn_implementation", getattr(self.config, "attn_implementation", "eager")
        )
        if (
            attn_impl == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            causal_mask = AttentionMaskConverter._unmask_unattended(
                cast(torch.FloatTensor, causal_mask.to(dtype=torch.float32)),
                min_dtype,
            )

        return causal_mask


class LlamaBiForMNTP(LlamaForCausalLM):
    """Causal LM wrapper around :class:`LlamaBiModel` for MNTP with optional PEFT."""

    def __init__(self, config: LlamaConfig):
        LlamaPreTrainedModel.__init__(self, config)
        self.model = LlamaBiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_model_for_peft(self) -> LlamaBiModel | PeftModel:
        """Return the inner model for PEFT wrapping (base or wrapped).

        Returns:
        -------
            ``self.model``, either a :class:`LlamaBiModel` or a :class:`peft.PeftModel`.
        """
        return self.model

    def set_model_for_peft(self, model: PeftModel) -> None:
        """Replace the inner model with a PEFT-wrapped model.

        Args:
            model: A :class:`peft.PeftModel` whose base matches the expected backbone.
        """
        self.model = model

    def save_peft_model(self, path: str) -> None:
        """Save the (possibly PEFT-wrapped) inner model to ``path``.

        Args:
            path: Directory path passed to ``save_pretrained`` on the inner model.
        """
        self.model.save_pretrained(path)

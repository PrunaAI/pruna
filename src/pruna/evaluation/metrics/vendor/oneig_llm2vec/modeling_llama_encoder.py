# Copyright 2025 - Pruna AI GmbH. All rights reserved.
#
# Derived from McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp ``modeling_llama_encoder.py``
# (Hugging Face Hub). Upstream requires ``flash_attention_2`` only; this copy allows ``eager``
# and ``sdpa`` so ``oneig_reasoning`` can run on CPU without ``flash_attn``. See
# ``NOTICE.oneig_llm2vec`` in the parent ``vendor`` directory.

import importlib.metadata

from packaging import version
from torch import nn
from transformers import LlamaConfig, LlamaModel, LlamaPreTrainedModel
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.utils import logging
from transformers.utils.import_utils import _is_package_available

logger = logging.get_logger(__name__)


def is_transformers_attn_greater_or_equal_4_56_2() -> bool:
    if not _is_package_available("transformers"):
        return False
    return version.parse(importlib.metadata.version("transformers")) >= version.parse("4.56.2")


class ModifiedLlamaAttention(LlamaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        GradientCheckpointingLayer.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = ModifiedLlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class LlamaEncoderModel(LlamaModel):
    def __init__(self, config: LlamaConfig) -> None:
        if not is_transformers_attn_greater_or_equal_4_56_2():
            raise ValueError(
                "The current implementation of LlamaEncoderModel follows modeling_llama.py "
                "of transformers version >= 4.56.2"
            )
        LlamaPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [ModifiedLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        attn_impl = getattr(config, "_attn_implementation", getattr(config, "attn_implementation", "eager"))
        self._use_sdpa = attn_impl == "sdpa"
        self._use_flash_attention_2 = attn_impl == "flash_attention_2"
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

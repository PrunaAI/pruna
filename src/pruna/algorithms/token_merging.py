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

import math
from typing import Any, Callable

import torch
from ConfigSpace import UniformIntegerHyperparameter
from transformers import ImageClassificationPipeline

from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag as tags
from pruna.config.hyperparameters import Boolean
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.engine.save import SAVE_FUNCTIONS
from pruna.logging.logger import pruna_logger

# ---------------------------------------------------------------------------
# Token merging utility functions (adapted from facebook/ToMe)
# ---------------------------------------------------------------------------


def _do_nothing(x: torch.Tensor, mode: str | None = None) -> torch.Tensor:
    """Identity function used as a no-op merge / unmerge."""
    return x


def _parse_r(num_layers: int, r: int | list[int] | tuple[int, float]) -> list[int]:
    """
    Process a constant *r* or *r* schedule into a per-layer list.

    Parameters
    ----------
    num_layers : int
        Number of transformer blocks.
    r : int | list[int] | tuple[int, float]
        Token reduction amount. Can be a constant ``int``, a ``(r, inflection)``
        tuple, or an explicit per-layer list.
        Inflection describes the trend of the r value over layers.
        It can increase (+1), decrease (-1), or stay constant (0).
        Any value between -1 and +1 is accepted.

    Returns
    -------
    list[int]
        A list of length ``num_layers`` with the number of tokens to merge in
        each layer.
    """
    inflect = 0
    if isinstance(r, list):
        if len(r) < num_layers:
            r = r + [0] * (num_layers - len(r))
        return list(r)
    elif isinstance(r, tuple):
        r, inflect = r

    min_val = int(r * (1.0 - inflect))
    max_val = 2 * r - min_val
    step = (max_val - min_val) / (num_layers - 1)
    return [int(min_val + step * i) for i in range(num_layers)]


def _bipartite_soft_matching(
    tokens: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> tuple[Callable, Callable]:
    """
    Apply ToMe with a balanced matching set (50 %, 50 %).

    Parameters
    ----------
    tokens : torch.Tensor
        Token tensor of shape ``[batch, tokens, channels]``.
    r : int
        Number of tokens to remove (at most 50 % of tokens).
    class_token : bool
        Whether a class token is present (will not be merged).
    distill_token : bool
        Whether a distillation token is present (will not be merged).

    Returns
    -------
    tuple[Callable, Callable]
        ``(merge, unmerge)`` callables.
    """
    protected = int(class_token) + int(distill_token)
    t = tokens.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return _do_nothing, _do_nothing

    with torch.no_grad():
        tokens = tokens / tokens.norm(dim=-1, keepdim=True)
        a, b = tokens[..., ::2, :], tokens[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged tokens
        src_idx = edge_idx[..., :r, :]  # Merged tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        """Merge tokens by scattering sources into their matched destinations."""
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        """Reverse a previous merge operation (approximate inverse)."""
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, tokens.shape[1], c, device=x.device, dtype=x.dtype)
        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


def _merge_wavg(merge: Callable, x: torch.Tensor, size: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Merge via weighted average based on token size.

    Parameters
    ----------
    merge : Callable
        The merge function returned by ``_bipartite_soft_matching``.
    x : torch.Tensor
        Token tensor to merge.
    size : torch.Tensor | None
        Current token sizes.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(merged_x, new_size)``.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")
    x = x / size
    return x, size


def _merge_source(merge: Callable, x: torch.Tensor, source: torch.Tensor | None = None) -> torch.Tensor:
    """
    Track merge sources as an adjacency matrix.

    Parameters
    ----------
    merge : Callable
        The merge function returned by ``_bipartite_soft_matching``.
    x : torch.Tensor
        Token tensor (used to infer shape when *source* is ``None``).
    source : torch.Tensor | None
        Existing source adjacency matrix, or ``None`` to initialise.

    Returns
    -------
    torch.Tensor
        Updated source adjacency matrix.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)
    return merge(source, mode="amax")


# ---------------------------------------------------------------------------
# ToMe-aware HuggingFace ViT modules (module-level for picklability)
# ---------------------------------------------------------------------------

try:
    from transformers.models.vit.modeling_vit import ViTLayer as _HFViTLayer
    from transformers.models.vit.modeling_vit import ViTSelfAttention as _HFViTSelfAttention

    class ToMeViTSelfAttention(_HFViTSelfAttention):
        """
        Self-attention with proportional attention and key-metric side-output for ToMe.

        Modifications over the base HuggingFace ``ViTSelfAttention``:
        - Uses eager attention to inject proportional attention weighting when
          ``self._tome_info["prop_attn"]`` is ``True`` and a token ``size`` is available.
        - Stores the mean of *k* over heads in ``self._tome_info["metric"]`` so that
          the enclosing ``ToMeViTLayer`` can use it for bipartite matching without
          requiring changes to the intermediate ``ViTAttention`` wrapper.
        """

        _tome_info: dict[str, Any]

        def forward(
            self,
            hidden_states: torch.Tensor,
            head_mask: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Forward pass with proportional attention and key-metric storage."""
            batch_size = hidden_states.shape[0]
            new_shape = (batch_size, -1, self.num_attention_heads, self.attention_head_size)

            key_layer = self.key(hidden_states).view(*new_shape).transpose(1, 2)
            value_layer = self.value(hidden_states).view(*new_shape).transpose(1, 2)
            query_layer = self.query(hidden_states).view(*new_shape).transpose(1, 2)

            # Eager attention so we can inject the proportional-attention term.
            attn_weights = (query_layer @ key_layer.transpose(-2, -1)) * self.scaling

            # Proportional attention: bias scores by log(token_size).
            if self._tome_info["prop_attn"] and self._tome_info["size"] is not None:
                attn_weights = attn_weights + self._tome_info["size"].log()[:, None, None, :, 0]

            if head_mask is not None:
                attn_weights = attn_weights + head_mask

            attn_weights = attn_weights.softmax(dim=-1)
            attn_probs = torch.nn.functional.dropout(attn_weights, p=self.dropout_prob if self.training else 0.0)

            context_layer = (attn_probs @ value_layer).transpose(1, 2)
            context_layer = context_layer.reshape(batch_size, -1, self.all_head_size)

            # Store the key mean as the similarity metric for token merging.
            self._tome_info["metric"] = key_layer.mean(1)

            return context_layer, attn_weights

    class ToMeViTLayer(_HFViTLayer):
        """
        ViT encoder layer that applies Token Merging between attention and MLP.

        After the attention sub-layer and its residual connection, this layer
        performs bipartite soft matching on the key-metric stored in
        ``self._tome_info["metric"]`` and merges the ``r`` most similar token
        pairs before proceeding to the MLP sub-layer.
        """

        _tome_info: dict[str, Any]

        def forward(
            self,
            hidden_states: torch.Tensor,
            head_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Forward pass with token merging between attention and MLP."""
            # --- self-attention + first residual ---
            hidden_states_norm = self.layernorm_before(hidden_states)
            attention_output = self.attention(hidden_states_norm, head_mask)
            hidden_states = attention_output + hidden_states

            # --- token merging ---
            r = self._tome_info["r"].pop(0)
            if r > 0:
                metric = self._tome_info["metric"]
                merge, _ = _bipartite_soft_matching(
                    metric,
                    r,
                    self._tome_info["class_token"],
                    self._tome_info["distill_token"],
                )
                if self._tome_info["trace_source"]:
                    self._tome_info["source"] = _merge_source(merge, hidden_states, self._tome_info["source"])
                hidden_states, self._tome_info["size"] = _merge_wavg(merge, hidden_states, self._tome_info["size"])

            # --- MLP + second residual ---
            layer_output = self.layernorm_after(hidden_states)
            layer_output = self.intermediate(layer_output)
            layer_output = self.output(layer_output, hidden_states)

            return layer_output

except ImportError:
    ToMeViTSelfAttention = None
    ToMeViTLayer = None


# ---------------------------------------------------------------------------
# Picklable model wrapper
# ---------------------------------------------------------------------------


class ToMeModelWrapper(torch.nn.Module):
    """
    Wrapper that initialises ``_tome_info`` on every forward call.

    This class is defined at module level so that the wrapped model can be
    pickled and unpickled without issues.  On each forward pass it resets the
    per-layer ``r`` schedule and clears any accumulated token-size / source
    state before delegating to the underlying model.

    Parameters
    ----------
    model : torch.nn.Module
        A HuggingFace ViT model (already patched with ``ToMeViTLayer`` /
        ``ToMeViTSelfAttention``).
    r : int
        The number of tokens to merge per layer.
    tome_info : dict
        The shared mutable state dict read/written by all ``ToMeViTLayer``
        instances.
    num_layers : int
        The number of transformer layers in the model (used by ``_parse_r``).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        r: int,
        tome_info: dict,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.r = r
        self._tome_info = tome_info
        self.num_layers = num_layers

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Initialise ToMe state and forward through the wrapped model."""
        self._tome_info["r"] = _parse_r(self.num_layers, self.r)
        self._tome_info["size"] = None
        self._tome_info["source"] = None
        self._tome_info["metric"] = None
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped model for convenience."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------


class TokenMerging(PrunaAlgorithmBase):
    """
    Apply Token Merging (ToMe) to HuggingFace Vision Transformer models.

    Token Merging progressively merges similar tokens between the attention
    and MLP stages of each transformer block, reducing the total number of
    tokens and therefore speeding up inference with minimal quality loss.
    """

    algorithm_name: str = "token_merging"
    group_tags: list[str] = [tags.KERNEL]
    save_fn = SAVE_FUNCTIONS.reapply
    references: dict[str, str] = {
        "Paper": "https://arxiv.org/abs/2210.09461",
        "GitHub": "https://github.com/facebookresearch/ToMe",
    }
    tokenizer_required: bool = False
    processor_required: bool = False
    runs_on: list[str] = ["cpu", "cuda"]
    dataset_required: bool = False

    def model_check_fn(self, model: Any) -> bool:
        """
        Check whether *model* contains HuggingFace ``ViTLayer`` blocks.

        Parameters
        ----------
        model : Any
            The model to check.

        Returns
        -------
        bool
            ``True`` if the model contains at least one ``ViTLayer``.
        """
        try:
            from transformers.models.vit.modeling_vit import ViTLayer
        except ImportError:
            pruna_logger.warning("Transformers library not found. Token merging will not be applied.")
            return False

        return any(isinstance(m, ViTLayer) for m in model.model.modules())

    def import_algorithm_packages(self) -> dict[str, Any]:
        """
        Import the required HuggingFace ViT classes.

        Returns
        -------
        Dict[str, Any]
            Dictionary with ``ViTLayer`` and ``ViTSelfAttention``.
        """
        from transformers.models.vit.modeling_vit import ViTLayer, ViTSelfAttention

        return dict(ViTLayer=ViTLayer, ViTSelfAttention=ViTSelfAttention)

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Apply Token Merging to a HuggingFace ViT model.

        For every ``ViTLayer`` in the model, swaps its class to
        ``ToMeViTLayer`` (which performs bipartite token merging after
        self-attention).  For every ``ViTSelfAttention``, swaps its class to
        ``ToMeViTSelfAttention`` (which uses eager attention with proportional
        weighting and stores the key metric).  The model is then wrapped in a
        ``ToMeModelWrapper`` that resets the shared ``_tome_info`` state
        before every forward pass.

        Parameters
        ----------
        model : Any
            A HuggingFace ViT model (e.g. ``ViTForImageClassification``
            or ``ViTModel``).
        smash_config : SmashConfigPrefixWrapper
            Algorithm configuration providing ``r``, ``trace_source``, and
            ``prop_attn``.

        Returns
        -------
        ToMeModelWrapper
            The wrapped model with Token Merging applied.
        """
        if isinstance(model, ImageClassificationPipeline):
            model = model.model

        imported = self.import_algorithm_packages()
        vit_layer_cls = imported["ViTLayer"]
        vit_self_attn_cls = imported["ViTSelfAttention"]

        r = smash_config["r"]
        trace_source = smash_config["trace_source"]
        prop_attn = smash_config["prop_attn"]

        # Shared mutable state dict – every ToMe module reads from / writes to this.
        tome_info: dict[str, Any] = {
            "r": r,
            "size": None,
            "source": None,
            "metric": None,
            "trace_source": trace_source,
            "prop_attn": prop_attn,
            "class_token": True,
            "distill_token": False,
        }

        # Swap every ViTLayer / ViTSelfAttention to the ToMe-aware variants.
        num_layers = 0
        for module in model.modules():
            if isinstance(module, vit_layer_cls):
                module.__class__ = ToMeViTLayer
                module._tome_info = tome_info
                num_layers += 1
            elif isinstance(module, vit_self_attn_cls):
                module.__class__ = ToMeViTSelfAttention
                module._tome_info = tome_info

        return ToMeModelWrapper(model, r, tome_info, num_layers)

    def get_hyperparameters(self) -> list:
        """
        Return the algorithm-specific hyperparameters.

        Returns
        -------
        list
            A list containing:
            - ``r`` – number of tokens to merge per layer (int, 0–128).
            - ``trace_source`` – whether to track merge provenance (bool).
            - ``prop_attn`` – whether to use proportional attention (bool).
        """
        return [
            UniformIntegerHyperparameter(
                "r",
                lower=0,
                upper=128,
                default_value=16,
                meta=dict(
                    desc=(
                        "Number of tokens to merge per transformer layer. "
                        "Higher values speed up inference but may reduce accuracy."
                    )
                ),
            ),
            Boolean(
                name="trace_source",
                default=False,
                meta=dict(desc="Track the source of each merged token (useful for visualisation)."),
            ),
            Boolean(
                name="prop_attn",
                default=True,
                meta=dict(desc="Use proportional attention weights based on token size."),
            ),
        ]

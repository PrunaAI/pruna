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

"""Utility functions for Token Merging (ToMe) algorithm."""

from typing import Callable, Tuple

import torch
import torch.nn.functional as F


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Perform bipartite soft matching to merge tokens.

    This is the core algorithm from the ToMe paper. It splits tokens into two sets,
    computes similarity between them, and merges the most similar pairs.

    Parameters
    ----------
    metric : torch.Tensor
        Similarity metric between tokens. Shape: [B, N, C] where B is batch size,
        N is number of tokens, C is feature dimension.
    r : int
        Number of tokens to remove (merge).
    class_token : bool
        Whether the first token is a class token (should not be merged).
    distill_token : bool
        Whether there is a distillation token (should not be merged).

    Returns
    -------
    Tuple[Callable, Callable]
        merge function and unmerge function
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens (since we split into two sets)
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        # No merging needed
        return do_nothing, do_nothing

    with torch.no_grad():
        # Compute similarity matrix
        metric = metric / (metric.norm(dim=-1, keepdim=True) + 1e-8)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        # Handle protected tokens (class/distill tokens)
        if protected > 0:
            scores[..., :protected, :] = -float("inf")
            scores[..., :, :protected] = -float("inf")

        # Find best matches using max along each axis
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        # Only keep top r matches
        unm_idx = edge_idx[..., r:, :]  # Unmerged tokens
        src_idx = edge_idx[..., :r, :]  # Source tokens to merge
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)  # Destination tokens

    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        """
        Merge tokens based on the computed matching.

        Parameters
        ----------
        x : torch.Tensor
            Token features to merge. Shape: [B, N, C]
        mode : str
            Merging mode: 'mean' for averaging, 'sum' for summing

        Returns
        -------
        torch.Tensor
            Merged tokens
        """
        src, dst = x[..., ::2, :], x[..., 1::2, :]

        # Extract tokens
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        # Handle protected tokens
        if protected > 0:
            # Prepend class/distill tokens
            return torch.cat([x[..., :protected, :], dst, unm], dim=-2)
        else:
            return torch.cat([dst, unm], dim=-2)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        """
        Unmerge tokens to original size.

        This is mainly for reconstruction purposes - duplicates merged tokens
        to restore original token count.

        Parameters
        ----------
        x : torch.Tensor
            Merged tokens

        Returns
        -------
        torch.Tensor
            Unmerged tokens (approximately restored to original)
        """
        # For simplicity, we duplicate the merged tokens
        # In practice, this is only needed for certain architectures
        protected_tokens = x[..., :protected, :]
        n, t, c = x.shape
        dst_tokens = x[..., protected : t - (unm_idx.shape[-2]), :]
        unm_tokens = x[..., t - (unm_idx.shape[-2]) :, :]

        # Restore original ordering (approximately)
        out = torch.zeros(n, src_idx.shape[-2], c, device=x.device, dtype=x.dtype)
        out[..., :r, :] = dst_tokens  # if needed
        out.scatter_(dim=-2, index=src_idx.expand(n, r, c), src=dst_tokens)


        out[..., 1::2, :] = dst_tokens
        # Duplicate merged tokens for source positions
        out.scatter_(dim=-2, index=(src_idx * 2).expand(n, r, c), src=dst_tokens.gather(-2, dst_idx.expand(n, r, c)))

        # Restore unmerged tokens
        out.scatter_(dim=-2, index=(unm_idx * 2).expand(n, unm_idx.shape[-2], c), src=unm_tokens)

        if protected > 0:
            return torch.cat([protected_tokens, out], dim=-2)
        return out

    return merge, unmerge


def do_nothing(x: torch.Tensor, mode: str = None) -> torch.Tensor:
    """Pass-through function when no merging is needed."""
    return x


def apply_tome_to_vit(model: torch.nn.Module, r: int = 0) -> torch.nn.Module:
    """
    Apply Token Merging to a Vision Transformer model.

    This patches the model's attention blocks to perform token merging.

    Parameters
    ----------
    model : torch.nn.Module
        The Vision Transformer model to patch
    r : int
        Number of tokens to merge per layer (0-100, representing percentage)

    Returns
    -------
    torch.nn.Module
        The patched model

    Raises
    ------
    ValueError
        If r is negative or the model has no compatible blocks
    """
    if r < 0:
        raise ValueError(f"Reduction ratio must be non-negative, got {r}")

    # Store the reduction ratio in the model
    model._tome_r = r

    # If r is 0, no merging needed
    if r == 0:
        return model

    patched_count = 0

    # Find and patch attention blocks
    for name, module in model.named_modules():
        # More specific checks for transformer blocks
        is_transformer_block = (
            ("blocks." in name and "block" in name.lower()) or  # ViT style: model.blocks.0
            ("layers." in name and "layer" in name.lower()) or  # Alternative: model.layers.0
            ("encoder.layer." in name.lower()) or  # BERT style
            hasattr(module, "attn") or  # Has attention attribute
            hasattr(module, "self_attn")  # Has self-attention
        )

        # Only patch if it's a transformer block and has forward method
        if is_transformer_block and hasattr(module, "forward"):
            original_forward = module.forward

            def make_forward_with_merging(original_fn, r_value):
                def forward_with_merging(x, *args, **kwargs):
                    # Validate input shape before merging
                    if not isinstance(x, torch.Tensor):
                        return original_fn(x, *args, **kwargs)

                    # Only merge if input is 3D [B, N, C]
                    if x.dim() != 3:
                        return original_fn(x, *args, **kwargs)

                    B, N, C = x.shape
                    num_to_merge = int(N * r_value / 100.0)

                    # Apply token merging
                    if num_to_merge > 0 and N > 1:  # Need at least 2 tokens to merge
                        try:
                            # Use the token features as the metric
                            merge, unmerge = bipartite_soft_matching(
                                x.detach(), r=num_to_merge, class_token=True
                            )
                            x = merge(x)
                        except Exception:
                            # If merging fails, continue without it
                            # This ensures robustness
                            pass

                    # Run the original forward pass
                    return original_fn(x, *args, **kwargs)

                return forward_with_merging

            module.forward = make_forward_with_merging(original_forward, r)
            patched_count += 1

    # Verify we actually patched something
    if patched_count == 0:
        raise ValueError(
            f"No compatible transformer blocks found in {model.__class__.__name__}. "
            "Token merging requires Vision Transformer-style architectures with block/layer modules."
        )

    return model



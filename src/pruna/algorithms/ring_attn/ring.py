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
import functools
from collections.abc import Iterable
from types import ModuleType
from typing import Any, List, Optional, Union

import torch
import torch.distributed as dist
from ConfigSpace import CategoricalHyperparameter
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from torch.distributed.tensor.device_mesh import DeviceMesh

from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag
from pruna.algorithms.ring_attn.utils.ring_utils import RingDistributedContext
from pruna.config.hyperparameters import Boolean
from pruna.config.smash_config import SmashConfig, SmashConfigPrefixWrapper
from pruna.engine.save import SAVE_FUNCTIONS

ring_attention: ModuleType | None = None

with contextlib.suppress(ImportError):
    # see "import_algorithm_packages" for further explanation
    import torch.distributed.tensor.experimental._attention as ring_attention


class RingAttn(PrunaAlgorithmBase):
    """
    Distributed attention on multiple GPUs computation by using the torch native ring attention implementation.

    Each GPU stores only its own slice of Q/K/V and participates in a Ring Attention shuffle that lets every query
    attend to every key/value. The result is lower KV-cache/activation memory per GPU and higher arithmetic intensity.
    """

    algorithm_name: str = "ring_attn"
    group_tags: list[AlgorithmTag] = [AlgorithmTag.KERNEL]
    save_fn = SAVE_FUNCTIONS.reapply
    references = {
        "Implementation": "https://docs.pytorch.org/tutorials/prototype/context_parallel.html",
        "Paper": "https://arxiv.org/pdf/2310.01889",
    }
    tokenizer_required: bool = False
    processor_required: bool = False
    runs_on: list[str] = ["cuda"]
    dataset_required: bool = False
    compatible_before: Iterable[str | AlgorithmTag] = [
        "qkv_diffusers",
    ]
    compatible_after: Iterable[str | AlgorithmTag] = ["torch_compile"]

    def get_hyperparameters(self) -> list:
        """
        Get the hyperparameters for the RingAttn.

        Returns
        -------
        list
            A list of hyperparameters.
        """
        return [
            Boolean(
                "convert_to_f32",
                default=True,
                meta=dict(desc="Allowing intermediate computations in the attention mechanism to be upcast to 32-bit."),
            ),
            CategoricalHyperparameter(
                "rotate_method",
                default_value="ALL_TO_ALL",
                meta=dict(desc="The method to use for rotating the computations."),
                choices=["ALL_TO_ALL", "ALL_GATHER"],
            ),
        ]

    def model_check_fn(self, model: Any) -> bool:
        """
        Check if the model is supported by the RingAttn.

        Parameters
        ----------
        model : Any
            The model to check.

        Returns
        -------
        bool
            True if the model is supported, False otherwise.
        """
        if torch.cuda.device_count() < 2:
            raise ValueError("RingAttn requires at least 2 GPUs")

        return hasattr(model, "transformer") and isinstance(
            model.transformer, (FluxTransformer2DModel, WanTransformer3DModel)
        )

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:

        # configure the ring attention hyperparameters
        _cp_options = ring_attention._cp_options  # type: ignore
        _cp_options.convert_to_f32 = smash_config["convert_to_f32"]
        _cp_options.enable_load_balance = False
        _cp_options.rotate_method = getattr(ring_attention._RotateMethod, smash_config["rotate_method"])  # type: ignore

        wrap_pipeline_call(model, torch.cuda.device_count())

        mesh = dist.init_device_mesh("cuda", (torch.cuda.device_count(),), mesh_dim_names=("ring_dim",))
        rank = dist.get_rank()
        world_size = torch.cuda.device_count()

        if isinstance(model.transformer, FluxTransformer2DModel):
            wrap_flux2d_transformer_forward(
                model.transformer,
                world_size,
                smash_config._base_config,
                rank,
                mesh,
                cache_helper=getattr(model, "cache_helper", None),
            )
        elif isinstance(model.transformer, WanTransformer3DModel):
            wrap_wan3d_transformer_forward(model.transformer, world_size, smash_config._base_config, rank, mesh)
        else:
            raise ValueError(f"Unsupported transformer type: {type(model.transformer)}")

        return model

    def import_algorithm_packages(self) -> dict[str, Any]:
        """
        Import the algorithm packages.

        Returns
        -------
        dict[str, Any]
            The algorithm packages.
        """
        # even though it is a torch import we isolate it, as experimental modules can often change the interface
        # we import the package even though we dont use it directly to make sure it is available
        # additionally, we can not pass it as module to the distributed setup (not picklable)
        # nor as a string (the import massively irritates torch.compile)
        # we import it on the top of the file if available
        import torch.distributed.tensor.experimental._attention as ring_attention  # noqa: F401

        return dict()


def wrap_wan3d_transformer_forward(
    model: Any,
    world_size: int,
    smash_config: Union[SmashConfig, SmashConfigPrefixWrapper],
    rank: int,
    mesh: DeviceMesh,
) -> Any:
    """
    Wrap the transformer forward pass to chunk the inputs and intercept the torch attention function.

    Parameters
    ----------
    model : Any
        The transformer model to wrap.
    world_size : int
        The number of GPUs to distribute the model on.
    smash_config : SmashConfig
        The SmashConfig to use.
    rank : int
        The rank of the current process.
    mesh : DeviceMesh
        The mesh to use for the distributed attention.
    """
    for i, block in enumerate(model.blocks):
        block_original = block.forward

        @functools.wraps(block_original)
        def block_forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            temb: torch.Tensor,
            rotary_emb: torch.Tensor,
            _block_ref=block,
            _original_forward=block_original,
            _layer_id=i,
            _num_layers=len(model.blocks),
        ) -> torch.Tensor:
            # on the first layer, we chunk the hidden states
            if _layer_id == 0:
                hidden_states = hidden_states.chunk(world_size, dim=-2)[rank]

            rotary_emb = rotary_emb.chunk(world_size, dim=-2)[rank]

            # Use compiled version if available, otherwise use original (not the wrapped one!)
            forward_to_call = getattr(_block_ref, "compiled_forward", _original_forward)

            with RingDistributedContext(mesh, smash_config):
                hidden_states = forward_to_call(hidden_states, encoder_hidden_states, temb, rotary_emb)

            # on the last layer, we sync back the hidden states
            if _layer_id == _num_layers - 1:
                return sync_tensor(hidden_states, dim=-2, group=dist.distributed_c10d._get_default_group())

            return hidden_states

        block.original_forward = block_original
        block.forward = block_forward.__get__(block)  # type: ignore


def wrap_pipeline_call(model: Any, world_size: int) -> Any:
    """
    Wrap the model forward pass to set up a generator with rank-specific device.

    Parameters
    ----------
    model : Any
        The model to wrap.
    world_size : int
        The number of GPUs to distribute the model on.
    """
    # Set up generator with rank-specific device, if it is not explicitly specified the different
    # processes might sample different seeds, we have to sync this
    original_forward = model.__call__

    @functools.wraps(original_forward)
    def new_forward(
        *args,
        **kwargs,
    ):
        rank = kwargs.pop("rank") if "rank" in kwargs else dist.get_rank()
        if "generator" not in kwargs:
            # if we distributed manually, we can not use "dist" to get the rank, in this case we pass the rank ourselves
            seed_t = torch.randint(0, torch.iinfo(torch.int64).max, [1], dtype=torch.int64, device=f"cuda:{rank}")
            seed_t = sync_tensor(seed_t, dim=0, group=None)
            seed_t = seed_t.chunk(world_size, dim=0)[0]
            seed = seed_t.item()
            seed -= torch.iinfo(torch.int64).min
            generator = torch.Generator(f"cuda:{rank}").manual_seed(seed)
            kwargs["generator"] = generator

        return original_forward(*args, **kwargs)

    model.__call__ = new_forward  # type: ignore


def wrap_flux2d_transformer_forward(
    model: Any,
    world_size: int,
    smash_config: Union[SmashConfig, SmashConfigPrefixWrapper],
    rank: int,
    mesh: DeviceMesh,
    cache_helper: Any | None = None,
) -> Any:
    """
    Wrap the transformer forward pass to chunk the inputs and intercept the torch attention function.

    Parameters
    ----------
    model : Any
        The transformer model to wrap.
    world_size : int
        The number of GPUs to distribute the model on.
    smash_config : SmashConfig
        The SmashConfig to use.
    rank : int
        The rank of the current process.
    mesh : DeviceMesh
        The mesh to use for the distributed attention.
    cache_helper : Any | None
        The cache helper if one is present in the pipe.
    """
    original_forward = model.forward

    @functools.wraps(original_forward)
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        img_ids: torch.Tensor | None = None,
        txt_ids: torch.Tensor | None = None,
        *args,
        **kwargs,
    ):
        # split all input tensors along the sequence length dimension and get chunk for this process (rank)
        # we do the forward pass on two separate chunks and only "sync" when the attention is computed
        # for intuition: number of chunks = number of GPUs
        hidden_states = hidden_states.chunk(world_size, dim=1)[rank]
        encoder_hidden_states = (
            encoder_hidden_states.chunk(world_size, dim=1)[rank] if encoder_hidden_states is not None else None
        )
        img_ids = img_ids.chunk(world_size, dim=0)[rank] if img_ids is not None else None
        txt_ids = txt_ids.chunk(world_size, dim=0)[rank] if txt_ids is not None else None

        # this context basically intercepts any call to F.scaled_dot_product_attention
        # and replaces it with the ring attention implementation
        with RingDistributedContext(mesh, smash_config):
            output = self.inner_forward(
                hidden_states,
                encoder_hidden_states,
                *args,
                img_ids=img_ids,
                txt_ids=txt_ids,
                **kwargs,
            )

        # before we output the result, we attach the separate chunks together again
        sample = output[0]
        sample = sync_tensor(sample, dim=-2, group=dist.distributed_c10d._get_default_group())
        return (sample, *output[1:])

    model.forward = new_forward.__get__(model)  # type: ignore
    model.inner_forward = original_forward.__get__(model if cache_helper is None else cache_helper)  # type: ignore


def sync_tensor(tensor: torch.Tensor, dim: int, group: dist.ProcessGroup | None) -> torch.Tensor:
    """
    Sync a tensor across a process group.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to sync.
    dim : int
        The dimension to sync along.
    group : dist.ProcessGroup | None
        The process group to sync across.

    Returns
    -------
    torch.Tensor
        The synced tensor.
    """
    tensor = tensor.transpose(0, dim).contiguous()

    if group is None:
        group = dist.distributed_c10d._get_default_group()

    if isinstance(group, dist.ProcessGroup):
        pg: Union[dist.ProcessGroup, List[dist.ProcessGroup]] = group
    else:
        pg = group.get_group()

    x_shape = tensor.shape
    tensor = tensor.flatten()
    x_numel = tensor.numel()  # type: ignore
    tensor = dist._functional_collectives.all_gather_tensor(tensor, group=pg, gather_dim=0)  # type: ignore
    if isinstance(tensor, dist._functional_collectives.AsyncCollectiveTensor):
        tensor.wait()
    x_shape = list(x_shape)  # type: ignore
    x_shape[0] *= tensor.numel() // x_numel  # type: ignore
    tensor = tensor.reshape(x_shape)  # type: ignore
    tensor = tensor.transpose(0, dim)
    return tensor

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

import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from ConfigSpace import CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter
from transformers.modeling_outputs import ImageClassifierOutput
from transformers.models.llama.modeling_llama import LlamaForCausalLM as Llama
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.opt.modeling_opt import OPTForCausalLM as Opt

from pruna.algorithms.pruning import PrunaPruner
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.config.smash_space import Boolean
from pruna.engine.model_checks import is_causal_lm
from pruna.engine.save import SAVE_FUNCTIONS
from pruna.logging.logger import pruna_logger

_QKV_PAT = re.compile(r"(q|k|v).*proj|query|key|value", re.I)
_Q_HEAD_ATTRS = ("num_attention_heads", "num_heads", "n_heads")
_KV_HEAD_ATTRS = ("num_key_value_heads",)
_HEAD_DIM_ATTRS = ("head_dim", "attention_head_size")
_EMBED_DIM_ATTRS = ("all_head_size", "embed_dim", "hidden_size")

is_gradient_based = {"TaylorImportance", "HessianImportance"}


class TorchStructuredPruner(PrunaPruner):
    """
    Implement structured pruning using torch.

    Structured pruning removes entire units like neurons, channels, or filters from a network, leading to a more compact
    and computationally efficient model while preserving a regular structure that standard hardware can easily optimize.
    """

    algorithm_name: str = "torch_structured"
    references: dict[str, str] = {"GitHub": "https://github.com/pytorch/pytorch"}
    # when performing structured pruning, the tensor sizes can change and disrupt normal saving
    save_fn = SAVE_FUNCTIONS.pickled
    tokenizer_required: bool = False
    processor_required: bool = False
    runs_on: list[str] = ["cpu", "cuda"]
    dataset_required: bool = True
    compatible_algorithms: dict[str, list[str]] = dict(quantizer=["half"])

    def get_hyperparameters(self) -> List:
        """
        Get the hyperparameters for the pruner.

        Returns
        -------
        List
            The hyperparameters for the pruner.
        """
        return [
            CategoricalHyperparameter(
                "type",
                choices=[
                    "RandomImportance",
                    "MagnitudeImportance",
                    "LAMPImportance",
                    "TaylorImportance",
                    "HessianImportance",
                ],
                default_value="MagnitudeImportance",
                meta=dict(desc="Importance criterion for pruning."),
            ),
            UniformIntegerHyperparameter(
                "calibration_samples",
                1,
                256,
                default_value=64,
                meta=dict(desc="Number of calibration samples for importance computation."),
            ),
            Boolean("prune_head_dims", meta=dict(desc="Whether to prune head dimensions.")),
            Boolean("prune_num_heads", meta=dict(desc="Whether to prune number of heads.")),
            Boolean("global_pruning", meta=dict(desc="Whether to perform global pruning.")),
            UniformFloatHyperparameter(
                "sparsity",
                0.0,
                1.0,
                default_value=0.1,
                meta=dict(desc="Sparsity level up to which to prune."),
            ),
            UniformFloatHyperparameter(
                "head_sparsity",
                0.0,
                1.0,
                default_value=0.0,
                meta=dict(desc="Sparsity level up to which to prune heads."),
            ),
            UniformIntegerHyperparameter(
                "it_steps",
                1,
                10,
                default_value=1,
                meta=dict(desc="Number of iterations for pruning."),
            ),
        ]

    def model_check_fn(self, model: Any) -> bool:  # noqa: PLR0911
        """
        Check if the model is supported by the pruner.

        Parameters
        ----------
        model : Any
            The model to check.

        Returns
        -------
        bool
            True if the model is supported, False otherwise.
        """
        # Pruning is still unstable for LLMs when they are *the* target artefact; skip for now
        if is_causal_lm(model):
            return False
        imported = self.import_algorithm_packages()
        # Simple heuristics – extend as needed
        if isinstance(model, (imported["Opt"], imported["Llama"], imported["ViT"])):
            return True
        if isinstance(model, imported["timm"].models.convnext.ConvNeXt):
            return True
        if isinstance(model, imported["torchvision"].models.resnet.ResNet):
            return True
        if isinstance(model, imported["GLiNER"]):
            return True
        return isinstance(model, imported["timm"].models.resnet.ResNet)

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Prune the model.

        Parameters
        ----------
        model : Any
            The model to prune.
        smash_config : SmashConfigPrefixWrapper
            The configuration for the pruning.

        Returns
        -------
        Any
            The pruned model.
        """
        imported = self.import_algorithm_packages()

        device = smash_config["device"]
        model = model.to(device)
        model.eval()

        # model forward does not work on half precision on cpu
        if device == "cpu":
            model.float()

        # Retrieve the importance function or class from the mapping based on the pruning type
        importance_function = getattr(imported_modules["tp"].importance, smash_config["type"])

        # Get the example input and move to device correctly if it's a dict
        batch = next(iter(smash_config.train_dataloader()))
        if isinstance(batch, dict):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            example_input = batch
        else:
            example_input = batch[0][:1, :].to(device)  # type: ignore[arg-type]

        # Get the target module to prune, and it's boundary and exterior

        target_module = get_target_module(model, imported, smash_config)
        dg = tp.DependencyGraph().build_dependency(model, example_input)
        boundary, exterior = get_boundary_and_exterior(target_module, model, dg)

        # Get the ignored layers
        ignored_layers = get_ignored_layers(boundary, exterior, model, imported)

        # Get the number of heads
        num_heads = find_attention_blocks(target_module)

        iterative_steps = smash_config["it_steps"]

        pruner = imported_modules["tp"].pruner.MetaPruner(
            model,
            example_input,
            importance=importance_function(),
            iterative_steps=iterative_steps,
            pruning_ratio=smash_config["sparsity"],
            ignored_layers=ignored_layers,
            num_heads=num_heads,
            prune_head_dims=smash_config["prune_head_dims"],
            prune_num_heads=smash_config["prune_num_heads"],
            head_pruning_ratio=smash_config["head_sparsity"],
            global_pruning=smash_config["global_pruning"],
        )

        for _ in range(iterative_steps):
            if smash_config["type"] in is_gradient_based:
                model = compute_loss_and_accumulate_gradients(
                    model,
                    # presence of dataloader is ensured beforehand
                    smash_config.train_dataloader(),  # type: ignore[arg-type]
                    device=device,
                    smash_config=smash_config,
                    calibration_data_size=smash_config["calibration_samples"],
                )
            pruner.step()

        for p in model.parameters():
            p.requires_grad = False

        model = patch_heads(model, pruner.num_heads)
        return model

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Provide a algorithm packages for the algorithm.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        try:
            import timm
            import torch_pruning as tp
            import torchvision
            from gliner import GLiNER
            from timm.models.mvitv2 import MultiScaleAttention
            from timm.models.mvitv2 import MultiScaleVit as MViT
            from transformers.models.llama.modeling_llama import LlamaForCausalLM as Llama
            from transformers.models.opt.modeling_opt import OPTForCausalLM as Opt
            from transformers.models.vit.modeling_vit import ViTForImageClassification as ViT
        except ImportError:
            pruna_logger.error("TorchStructuredPruner: You need the GPU version of Pruna (timm, torchvision).")
            raise
        return dict(
            timm=timm,
            torchvision=torchvision,
            Opt=Opt,
            Llama=Llama,
            MultiScaleAttention=MultiScaleAttention,
            MViT=MViT,
            tp=tp,
            ViT=ViT,
            GLiNER=GLiNER,
        )


def get_boundary_and_exterior(target_module, model, dg):
    """
    Get the boundary and exterior for a target module.

    Returns (boundary, exterior) where
       - boundary: modules inside target_module that exchange tensors
                    with modules outside target_module
       - exterior: modules that are NOT in the target_module subtree

    Parameters
    ----------
    target_module : nn.Module
        The target module to prune.
    model : nn.Module
        The full model.
    dg : tp.DependencyGraph
        The dependency graph of the model.

    Returns
    -------
    Tuple[Set[nn.Module], Set[nn.Module]]
        The boundary and exterior modules.
    """
    if target_module == model:  # If we are pruning the entire model, there is no boundary or exterior
        return set(), set()

    real_nodes = dg.module2node  # includes all modules, parameters, buffers, even internals like autograd nodes
    name_of = dg._module2name  # modules and parameter with actual names
    # We only want the modules and parameters that have actual names
    real_named = set(real_nodes.keys()) & set(name_of.keys())

    # Get the modules and parameters that are inside the target module
    enc_inside_prms = set(target_module.parameters())
    enc_inside_modules = set(target_module.modules())
    enc_inside = enc_inside_prms | enc_inside_modules  # We want both the parameters and the modules

    interior = real_named & enc_inside  # interior is intersection of entire model and target module
    exterior = real_named - interior  # exterior is the rest of the modules

    def touches_exterior(node):
        stack, seen = node.inputs + node.outputs, set()
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            mod = n.module
            # We don't want the auxiliary nodes like autograd nodes
            # So we only want the modules that have actual names
            if mod in real_named:
                return mod in exterior  # If the input or output is in the exterior, then the module touches the exterior
            stack.extend(n.inputs)
            stack.extend(n.outputs)
        return False

    boundary = {m for m in interior if touches_exterior(real_nodes[m])}

    return boundary, exterior


def _get_q_head_attr_value(m: nn.Module) -> Optional[int]:
    """
    Get the head count attribute value from the module.

    Parameters
    ----------
    m : nn.Module
        The module to get the head count attribute value from.

    Returns
    -------
    Optional[int]
        The head count attribute value.
    """
    for attr in _Q_HEAD_ATTRS:
        if hasattr(m, attr):
            return getattr(m, attr)
    return None


def _get_kv_head_attr_value(m: nn.Module) -> Optional[int]:
    """
    Get the KV head count attribute value from the module if it exists.

    Parameters
    ----------
    m : nn.Module
        The module to get the KV head count attribute value from.

    Returns
    -------
    Optional[int]
        The KV head count attribute value.
    """
    for attr in _KV_HEAD_ATTRS:
        if hasattr(m, attr):
            return getattr(m, attr)
    return None


def _infer_qkv_linears(m: nn.Module) -> List[Tuple[str, nn.Linear]]:
    """
    Infer the QKV layers from the module.

    Parameters
    ----------
    m : nn.Module
        The module to infer the QKV layers from.

    Returns
    -------
    List[Tuple[str, nn.Linear]]
        The QKV name and layer pairs.
    """
    return [(name, sub) for name, sub in m.named_children() if isinstance(sub, nn.Linear) and _QKV_PAT.fullmatch(name)]


def find_attention_blocks(model: nn.Module) -> Dict[nn.Linear, int]:
    """
    Return the projection to head_count map.

    Any module that exposes a head‑count and contains three QKV layers
    is treated as an attention block.

    Parameters
    ----------
    model : nn.Module
        The model to find the attention blocks in.

    Returns
    -------
    Dict[nn.Linear, int]
        The projection to head_count map.
    """
    mapping: Dict[nn.Linear, int] = {}
    for m in model.modules():
        # Is m an attention block? Keep going if not.
        if (heads := _get_q_head_attr_value(m)) is None:
            continue
        # Do we have a separate KV head count?
        kv_heads = _get_kv_head_attr_value(m)
        # Get the QKV layers.
        qkv = _infer_qkv_linears(m)
        if len(qkv) < 3:
            continue
        # Map every projection to the original head count
        for name, proj in qkv:
            is_kv = name.lower().startswith(("k", "v"))
            mapping[proj] = kv_heads if is_kv and kv_heads is not None else heads
    return mapping


def patch_heads(model: nn.Module, new_heads: Dict[nn.Linear, int]):
    """
    Update head count attributes and derived sizes after pruning.

    Parameters
    ----------
    model : nn.Module
        The model to patch the heads for.
    new_heads : Dict[nn.Linear, int]
        The new head count map.

    Returns
    -------
    nn.Module
        The patched model.
    """
    for m in model.modules():
        if (_get_q_head_attr_value(m)) is None:
            continue

        q_proj = (
            getattr(m, "q_proj", None) or getattr(m, "query", None) or getattr(m, "query_proj", None)
        )  # Maybe this should be the QKV pattern instead.
        if q_proj not in new_heads:
            continue  # This block was ignored

        new_h = new_heads[q_proj]
        out_features = q_proj.out_features  # type: ignore[union-attr]
        head_dim = out_features // new_h

        # Update head count.
        for attr in _Q_HEAD_ATTRS:
            if hasattr(m, attr):
                setattr(m, attr, new_h)

        # Update head_dim
        for attr in _HEAD_DIM_ATTRS:
            if hasattr(m, attr):
                setattr(m, attr, head_dim)

        # Update embed_dim
        for attr in _EMBED_DIM_ATTRS:
            if hasattr(m, attr):
                setattr(m, attr, out_features)

        # Update num_key_value_heads
        if hasattr(m, "num_key_value_heads"):
            k_proj = getattr(m, "k_proj", None) or getattr(m, "key", None) or getattr(m, "key_proj", None)
            if k_proj in new_heads:
                new_h_kv = new_heads[k_proj]
                setattr(m, "num_key_value_heads", new_h_kv)

        if isinstance(m, LlamaRotaryEmbedding):
            m.forward = _llama_rotary_embedding_forward.__get__(m, LlamaRotaryEmbedding)
    return model


def _llama_rotary_embedding_forward(self: Any, x: torch.Tensor, seq_len: Optional[int] = None):
    if seq_len > self.max_seq_len_cached:
        self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
    return (
        self.cos_cached[:seq_len, : x.shape[-1]].to(dtype=x.dtype),
        self.sin_cached[:seq_len, : x.shape[-1]].to(dtype=x.dtype),
    )


def get_target_module(model: Any, imported: Dict[str, Any], smash_config: SmashConfigPrefixWrapper) -> nn.Module:
    """
    Returns the target submodule of a model to be used for pruning.

    If a target module is explicitly provided via the Smash config (experimental mode), it is returned directly.

    Otherwise, the function attempts to select a meaningful submodule based on the model type.
    If no known model type is detected, the entire model is returned.

    Parameters
    ----------
    model : Any
        The model to prune.
    imported : Dict[str, Any]
        The imported modules.
    smash_config : SmashConfigPrefixWrapper
        The Smash config.

    Returns
    -------
    nn.Module
        The target module to prune.
    """
    if smash_config._target_module is not None:
        return smash_config._target_module
    if isinstance(model, imported["timm"].models.convnext.ConvNeXt):
        return model.stages
    if isinstance(model, imported["ViT"]):
        return model.vit.embeddings
    if isinstance(model, imported["GLiNER"]):
        return model.model.token_rep_layer.bert_layer
    return model


def get_ignored_layers(boundary, exterior, model, imported: Dict[str, Any]) -> List[nn.Module]:
    """
    Returns a list of layers to ignore during pruning.

    Combines the boundary and exterior of the target module into the ignored set.
    In a few model-specific cases, key layers are also added.

    Parameters
    ----------
    boundary : Set[nn.Module]
        Modules at the boundary of the pruning scope.
    exterior : Set[nn.Module]
        Modules explicitly outside the pruning target scope.
    model : nn.Module
        The full model from which ignored layers are determined.
    imported : Dict[str, Any]
        Dictionary of imported model references used for type checking.

    Returns
    -------
    List[nn.Module]
        A list of layers that should be ignored during pruning.
    """
    ignored_layers = boundary.union(exterior)
    if isinstance(model, (imported["torchvision"].models.resnet.ResNet, imported["timm"].models.resnet.ResNet)):
        return ignored_layers.union([model.conv1, model.bn1, model.fc])
    if isinstance(model, (imported["Opt"], imported["Llama"])):
        return ignored_layers.union([model.lm_head])
    return ignored_layers


def add_grad_checkpointing(model: Union[Opt, Llama], pruning_device: torch.device) -> Union[Opt, Llama]:
    """
    Enable gradient checkpointing for the given model.

    Parameters
    ----------
    model : nn.Module
        The model to enable gradient checkpointing for.
    pruning_device : torch.device
        The device to use for pruning. Only applicable for certain models.

    Returns
    -------
    nn.Module
        The model with gradient checkpointing enabled.
    """
    is_llm = isinstance(model, (Opt, Llama))
    if is_llm and pruning_device == "cuda":
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def compute_loss_and_accumulate_gradients(
    model: Union[Opt, Llama],
    calibration_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    smash_config: SmashConfigPrefixWrapper,
    calibration_data_size: int = 4096,
) -> Union[Opt, Llama]:
    """
    Calculate loss and perform backpropagation for the given model.

    Parameters
    ----------
    model : nn.Module
        The model to calculate loss and perform backpropagation on.
    calibration_dataloader : torch.utils.data.DataLoader
        The dataloader for calibration data.
    device : torch.device
        The device to perform calculations on.
    smash_config : SmashConfigPrefixWrapper
        A dictionary containing pruning and other configuration parameters.
    calibration_data_size : int,
        The number of calibration data samples to use, by default 4096.

    Returns
    -------
    nn.Module
        The updated model after backpropagation.
    """
    dataloader_iter = iter(calibration_dataloader)
    for p in model.parameters():
        p.requires_grad = True

    # default to CrossEntropyLoss
    loss_fn = torch.nn.CrossEntropyLoss()

    # add gradient checkpointing if LLM is pruned on cuda
    model = add_grad_checkpointing(model, device)
    model.train()

    is_llm = "CausalLM" in type(model).__name__
    for _ in range(calibration_data_size):
        batch_data, batch_labels = next(dataloader_iter)
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        if is_llm:
            # Huggingface has integrated loss computation for CasualLMs
            # handles shifting inputs to make labels
            loss = model(batch_data, labels=batch_data).loss
        else:
            logits = model(batch_data)
            if isinstance(logits, ImageClassifierOutput):
                logits = logits.logits
            loss = loss_fn(logits, batch_labels)
        loss.backward()
    return model

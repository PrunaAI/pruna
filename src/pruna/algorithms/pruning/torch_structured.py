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
from transformers.models.opt.modeling_opt import OPTForCausalLM as Opt

from pruna.algorithms.pruning import PrunaPruner
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.config.smash_space import Boolean
from pruna.engine.save import SAVE_FUNCTIONS
from pruna.logging.logger import pruna_logger

_QKV_PAT = re.compile(r"(q|k|v).*proj|query|key|value", re.I)  # To capture all QKV layers
_Q_HEAD_ATTRS = ("num_attention_heads", "num_heads", "n_heads")  # To capture all Q head attributes
_KV_HEAD_ATTRS = ("num_key_value_heads",)  # To capture all KV head attributes
_HEAD_DIM_ATTRS = ("head_dim", "attention_head_size")  # To capture all head dimension attributes
_EMBED_DIM_ATTRS = ("all_head_size", "embed_dim", "hidden_size")  # To capture all embed dimension attributes

is_gradient_based = {"TaylorImportance", "HessianImportance"}


class TorchStructuredPruner(PrunaPruner):
    """
    Implement structured pruning using torch.

    Structured pruning removes entire units like neurons, channels, or filters from a network, leading to a more compact
    and computationally efficient model while preserving a regular structure that standard hardware can easily optimize.

    Note: If you would like to prune a target module,
    you can by setting the target_module parameter in the smash config.
    Please set the experimental flag to True to use this feature.
    """

    algorithm_name: str = "torch_structured"
    references: dict[str, str] = {"GitHub": "https://github.com/pytorch/pytorch"}
    # when performing structured pruning, the tensor sizes can change and disrupt normal saving
    save_fn = SAVE_FUNCTIONS.pickled
    tokenizer_required: bool = False
    processor_required: bool = False
    runs_on: list[str] = ["cpu", "cuda"]
    dataset_required: bool = True
    compatible_algorithms: dict[str, list[str]] = dict(quantizer=["half", "torchao", "hqq"], compiler=["torch_compile"])

    def get_hyperparameters(self) -> list:
        """
        Configure all algorithm-specific hyperparameters with ConfigSpace.

        Returns
        -------
        list
            The hyperparameters.
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
                name="calibration_samples",
                lower=1,
                upper=256,
                default_value=64,
                meta=dict(desc="Number of calibration samples for importance computation."),
            ),
            Boolean("prune_head_dims", meta=dict(desc="Whether to prune head dimensions.")),
            Boolean("prune_num_heads", meta=dict(desc="Whether to prune number of heads.")),
            Boolean("global_pruning", meta=dict(desc="Whether to perform global pruning.")),
            UniformFloatHyperparameter(
                "sparsity",
                lower=0.0,
                upper=1.0,
                default_value=0.1,
                meta=dict(desc="Sparsity level up to which to prune."),
            ),
            UniformFloatHyperparameter(
                "head_sparsity",
                lower=0.0,
                upper=1.0,
                default_value=0.0,
                meta=dict(desc="Sparsity level up to which to prune heads."),
            ),
            UniformIntegerHyperparameter(
                name="it_steps",
                lower=1,
                upper=10,
                default_value=1,
                meta=dict(desc="Number of iterations for pruning."),
            ),
        ]

    def model_check_fn(self, model: Any) -> bool:
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
        imported_modules = self.import_algorithm_packages()
        # Simple heuristics – extend as needed
        if isinstance(model, (imported_modules["Opt"], imported_modules["ViT"])):
            return True
        if isinstance(model, imported_modules["timm"].models.convnext.ConvNeXt):
            return True
        if isinstance(model, imported_modules["torchvision"].models.resnet.ResNet):
            return True
        if isinstance(model, imported_modules["GLiNER"]):
            return True
        return isinstance(model, imported_modules["timm"].models.resnet.ResNet)

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
        importance_function = getattr(imported["tp"].importance, smash_config["type"])

        # Get the example input and move to device correctly if it's a dict
        batch = next(iter(smash_config.train_dataloader()))[0]
        if isinstance(batch, dict):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            example_input = batch
        else:
            # PrunaDataModule always returns a tuple of two tensors, the first is the input.
            example_input = batch[:1, :].to(device)  # type: ignore[arg-type]

        # Get the target module to prune, and it's boundary and exterior

        target_module = get_target_module(model, imported, smash_config)
        dependency_graph = (
            imported["tp"]
            .DependencyGraph()
            .build_dependency(model, example_input, output_transform=safe_output_transform)
        )
        boundary, exterior = get_boundary_and_exterior(target_module, model, dependency_graph)

        # Get the ignored layers
        ignored_layers = get_ignored_layers(boundary, exterior, model, imported)

        # Get the number of heads
        num_heads = get_qkv_head_counts(target_module)

        iterative_steps = smash_config["it_steps"]

        pruner = imported["tp"].pruner.MetaPruner(
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
            output_transform=safe_output_transform,
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

        model = update_heads_attribute(model, pruner.num_heads)
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


def get_boundary_and_exterior(target_module, model, dependency_graph):
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
    dependency_graph : tp.DependencyGraph
        The dependency graph of the model.

    Returns
    -------
    Tuple[Set[nn.Module], Set[nn.Module]]
        The boundary and exterior modules.
    """
    if target_module == model:  # If we are pruning the entire model, there is no boundary or exterior
        return set(), set()

    real_nodes = (
        dependency_graph.module2node
    )  # includes all modules, parameters, buffers, even internals like autograd nodes
    name_of = dependency_graph._module2name  # modules and parameter with actual names
    # We only want the modules and parameters that have actual names
    real_named = set(real_nodes.keys()) & set(name_of.keys())

    # Get the modules and parameters that are inside the target module
    inside_prms = set(target_module.parameters())
    inside_modules = set(target_module.modules())
    inside = inside_prms | inside_modules  # We want both the parameters and the modules

    interior = real_named & inside  # interior is intersection of entire model and target module
    exterior = real_named - interior  # exterior is the rest of the modules

    def touches_exterior(node):
        """
        Check if the module has any inputs or outputs that are in the exterior.

        This would mean that the module is on the boundary of the pruning scope.
        """
        stack, seen = node.inputs + node.outputs, set()
        while stack:
            n = stack.pop()
            if n not in seen:
                seen.add(n)
                mod = n.module
                # We don't want the auxiliary nodes like autograd nodes
                # So we only want the modules that have actual names
                if mod in real_named:
                    return (
                        mod in exterior
                    )  # If the input or output is in the exterior, then the module touches the exterior
                stack.extend(n.inputs)
                stack.extend(n.outputs)
        return False

    boundary = {m for m in interior if touches_exterior(real_nodes[m])}

    return boundary, exterior


def _pick_attr(module: nn.Module, names: Tuple[str, ...], default: Optional[int] = None) -> Optional[Any]:
    """
    Return the first attribute (or config attribute) from the possible candidates that exists.

    Parameters
    ----------
    module : nn.Module
        The module to pick the attribute from.
    names : Tuple[str, ...]
        The ordered list of candidate names.
    default : value to return if nothing is found

    Returns
    -------
    Optional[Any]
        The first attribute that exists.
    """
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
        cfg = getattr(module, "config", None)  # Huggingface models
        if cfg is not None and hasattr(cfg, name):
            return getattr(cfg, name)
    return default


def get_num_q_heads(module: nn.Module) -> int | None:
    """
    Return the number of Q heads for the module.

    Parameters
    ----------
    module : nn.Module
        The module to get the Q heads for.

    Returns
    -------
    Optional[int]
        The number of Q heads.
    """
    return _pick_attr(module, _Q_HEAD_ATTRS)


def get_num_kv_heads(module: nn.Module) -> int | None:
    """
    Return the number of KV heads for the module.

    Parameters
    ----------
    module : nn.Module
        The module to get the KV heads for.

    Returns
    -------
    Optional[int]
        The number of KV heads.
    """
    return _pick_attr(module, _KV_HEAD_ATTRS)


def _infer_qkv_linears(module: nn.Module) -> List[Tuple[str, nn.Linear]]:
    """
    Infer the QKV layers from the module.

    Parameters
    ----------
    module : nn.Module
        The module to infer the QKV layers from.

    Returns
    -------
    List[Tuple[str, nn.Linear]]
        The QKV name and layer pairs.
    """
    return [
        (name, sub) for name, sub in module.named_children() if isinstance(sub, nn.Linear) and _QKV_PAT.fullmatch(name)
    ]


def get_qkv_head_counts(module: nn.Module) -> dict[nn.Linear, int]:
    """
    Return the projection to head_count map.

    Any module that exposes a head‑count and contains three QKV layers
    is treated as an attention block.

    Parameters
    ----------
    module : nn.Module
        The model to find the attention blocks in.

    Returns
    -------
    Dict[nn.Linear, int]
        The projection to head_count map.
    """
    mapping: Dict[nn.Linear, int] = {}
    for m in module.modules():
        # Is m an attention block? Keep going if not.
        if (num_q_heads := get_num_q_heads(m)) is not None:
            # Do we have a separate KV head count?
            num_kv_heads = get_num_kv_heads(m)
            # Get the QKV layers.
            qkv = _infer_qkv_linears(m)
            if len(qkv) >= 3:
                # Map every projection to the original head count
                for name, proj in qkv:
                    is_kv = name.lower().startswith(("k", "v"))
                    mapping[proj] = num_kv_heads if is_kv and num_kv_heads is not None else num_q_heads
    return mapping


def update_heads_attribute(model: nn.Module, new_num_heads_map: Dict[nn.Linear, int]):
    """
    Update head count attributes and derived sizes after pruning.

    Parameters
    ----------
    model : nn.Module
        The model to patch the heads for.
    new_num_heads_map : Dict[nn.Linear, int]
        The new head count map.

    Returns
    -------
    nn.Module
        The patched model.
    """
    for m in model.modules():
        if get_num_q_heads(m) is not None:
            q_proj = getattr(m, "q_proj", None) or getattr(m, "query", None) or getattr(m, "query_proj", None)
            if q_proj in new_num_heads_map:
                new_num_head = new_num_heads_map[q_proj]
                new_num_out_features = q_proj.out_features  # type: ignore[union-attr]
                new_num_head_dim = new_num_out_features // new_num_head

                # Update head count.
                for attr in _Q_HEAD_ATTRS:
                    if hasattr(m, attr):
                        setattr(m, attr, new_num_head)

                # Update head_dim
                for attr in _HEAD_DIM_ATTRS:
                    if hasattr(m, attr):
                        setattr(m, attr, new_num_head_dim)

                # Update embed_dim
                for attr in _EMBED_DIM_ATTRS:
                    if hasattr(m, attr):
                        setattr(m, attr, new_num_out_features)

                # Update num_key_value_heads
                if hasattr(m, "num_key_value_heads"):
                    k_proj = getattr(m, "k_proj", None) or getattr(m, "key", None) or getattr(m, "key_proj", None)
                    if k_proj in new_num_heads_map:
                        new_num_kv_head = new_num_heads_map[k_proj]
                        setattr(m, "num_key_value_heads", new_num_kv_head)

    return model


def get_target_module(model: Any, imported: Dict[str, Any], smash_config: SmashConfigPrefixWrapper) -> nn.Module:
    """
    Return the target submodule of a model to be used for pruning.

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
    Return a list of layers to ignore during pruning.

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


def safe_output_transform(output) -> torch.Tensor:
    """
    Extract a tensor from the model output.

    Parameters
    ----------
    output : Any
        The model output.

    Returns
    -------
    torch.Tensor
        The tensor from the model output.
    """
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "logits"):  # Huggingface models
        return output.logits
    if isinstance(output, dict) and "logits" in output:  # Huggingface models
        return output["logits"]
    if isinstance(output, (tuple, list)):  # Huggingface models
        for o in output:
            if isinstance(o, torch.Tensor):
                return o
    raise ValueError("Could not extract a tensor from model output for dependency tracing.")

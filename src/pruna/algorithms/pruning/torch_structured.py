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

from typing import Any, Dict, List, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch_pruning as tp
from ConfigSpace import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from transformers import ViTForImageClassification as ViT
from transformers.modeling_outputs import ImageClassifierOutput
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding
from transformers.models.llama.modeling_llama import LlamaForCausalLM as Llama
from transformers.models.opt.modeling_opt import OPTAttention
from transformers.models.opt.modeling_opt import OPTForCausalLM as Opt
from transformers.models.vit.modeling_vit import ViTSelfAttention

from pruna.algorithms.pruning import PrunaPruner
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.config.smash_space import Boolean
from pruna.engine.model_checks import is_causal_lm
from pruna.engine.save import SAVE_FUNCTIONS
from pruna.logging.logger import pruna_logger

is_gradient_based = ["TaylorImportance", "HessianImportance"]


class TorchStructuredPruner(PrunaPruner):
    """
    Implement structured pruning using torch.

    Structured pruning removes entire units like neurons, channels, or filters from a network, leading to a more compact
    and computationally efficient model while preserving a regular structure that standard hardware can easily optimize.
    """

    algorithm_name = "torch_structured"
    references = {"GitHub": "https://github.com/pytorch/pytorch"}
    # when performing structured pruning, the tensor sizes can change and disrupt normal saving
    save_fn = SAVE_FUNCTIONS.pickled
    tokenizer_required = False
    processor_required = False
    run_on_cpu = True
    run_on_cuda = True
    dataset_required = True
    compatible_algorithms = dict(quantizer=["half"])

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
                log=False,
                default_value=0.1,
                meta=dict(desc="Sparsity level up to which to prune."),
            ),
            UniformFloatHyperparameter(
                "head_sparsity",
                lower=0.0,
                upper=1.0,
                log=False,
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
        Check if the model is a torch.nn.Module.

        Parameters
        ----------
        model : Any
            The model to check.

        Returns
        -------
        bool
            True if the model is a torch.nn.Module, False otherwise.
        """
        # Torch structured pruning is currently broken for LLMs, to be fixed
        if is_causal_lm(model):
            return False
        imported_modules = self.import_algorithm_packages()
        if isinstance(model, (Opt, Llama, ViT)):
            return True
        if isinstance(model, imported_modules["timm"].models.convnext.ConvNeXt):
            return True
        if isinstance(model, imported_modules["torchvision"].models.resnet.ResNet):
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
        imported_modules = self.import_algorithm_packages()

        device = smash_config["device"]
        model = model.to(device)
        model.eval()

        # model forward does not work on half precision on cpu
        if device == "cpu":
            model.float()

        # Retrieve the importance function or class from the mapping based on the pruning type
        importance_function = getattr(tp.importance, smash_config["type"])

        ch_groups, num_heads, ignored_layers = get_prunable_layers(model, smash_config, imported_modules)

        # Initialize a pruner
        example_input = next(iter(smash_config.train_dataloader()))[0][:1, :].to(device)  # type: ignore[arg-type]
        iterative_steps = smash_config["it_steps"]

        pruner = tp.pruner.MetaPruner(
            model,
            example_input,
            importance=importance_function(),
            iterative_steps=iterative_steps,
            pruning_ratio=smash_config["sparsity"],
            ignored_layers=ignored_layers,
            out_channel_groups=ch_groups,
            num_heads=num_heads,
            prune_head_dims=smash_config["prune_head_dims"],
            prune_num_heads=smash_config["prune_num_heads"],
            head_pruning_ratio=smash_config["head_sparsity"],
            global_pruning=smash_config["global_pruning"],
            round_to=64,
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

        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        model = update_dimensions_post_pruning(model, pruner, imported_modules)
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
            import torchvision
            from timm.models.mvitv2 import MultiScaleAttention
            from timm.models.mvitv2 import MultiScaleVit as MViT
        except ImportError:
            pruna_logger.error(
                "You are trying to use the Pruna Pruner, but timm is not installed. "
                "This is likely because you did not install the GPU version of Pruna."
            )
            raise

        return dict(timm=timm, torchvision=torchvision, MultiScaleAttention=MultiScaleAttention, MViT=MViT)


def get_prunable_layers(
    model: nn.Module, smash_config: SmashConfigPrefixWrapper, imported_modules: Dict
) -> Tuple[Dict, Dict, List]:
    """
    Get the parameters for performing tensor pruning on the given model.

    Parameters
    ----------
    model : nn.Module
        The model object to perform tensor pruning on.
    smash_config : SmashConfigPrefixWrapper
        A dictionary containing the pruning parameters.
    imported_modules : Dict
        Dictionary containing the imported modules.

    Returns
    -------
    tuple
        A tuple containing:
        - ch_groups (dict): Channel groups.
        - num_heads (dict): Number of attention heads.
        - ignored_layers (list): Ignored layers during pruning.
    """
    if isinstance(model, Opt):
        ch_groups, num_heads, ignored_layers = get_opt_params(model, smash_config)

    elif isinstance(model, Llama):
        ch_groups, num_heads, ignored_layers = get_llama_params(model, smash_config)

    elif isinstance(model, ViT):
        ch_groups, num_heads, ignored_layers = get_vit_params(model, smash_config)

    elif isinstance(model, imported_modules["timm"].models.convnext.ConvNeXt):
        ignored_layers = [model.stem, model.head]
        ch_groups, num_heads = dict(), dict()

    elif isinstance(
        model, (imported_modules["torchvision"].models.resnet.ResNet, imported_modules["timm"].models.resnet.ResNet)
    ):
        ignored_layers = [model.conv1, model.bn1, model.fc]
        ch_groups, num_heads = dict(), dict()
    else:
        ch_groups, num_heads, ignored_layers = dict(), dict(), []

    return ch_groups, num_heads, ignored_layers


def get_opt_params(model: nn.Module, smash_config: SmashConfigPrefixWrapper) -> Tuple[Dict, Dict, List]:
    """
    Get the optimization parameters for the Opt model.

    Parameters
    ----------
    model : nn.Module
        The Opt model to extract optimization parameters from.
    smash_config : SmashConfigPrefixWrapper
        A dictionary containing pruning parameters.

    Returns
    -------
    tuple
        A tuple containing:
        - params (dict): A dictionary of optimization parameters.
        - num_heads (dict): A dictionary mapping attention projection layers to the number of heads.
        - ignored_layers (list): A list of ignored layers during pruning.
    """
    num_heads = dict()
    for m in model.modules():
        if isinstance(m, OPTAttention):
            num_heads[m.q_proj] = m.num_heads
            num_heads[m.k_proj] = m.num_heads
            num_heads[m.v_proj] = m.num_heads

    ignored_layers = [model.lm_head]
    return dict(), num_heads, ignored_layers


def get_llama_params(model: nn.Module, smash_config: SmashConfigPrefixWrapper) -> Tuple[Dict, Dict, List]:
    """
    Get the parameters related to the LlamaAttention module in the model.

    Parameters
    ----------
    model : nn.Module
        The model containing the LlamaAttention module.
    smash_config : SmashConfigPrefixWrapper
        A dictionary containing pruning parameters.

    Returns
    -------
    tuple
        A tuple containing:
        - params (dict): An empty dictionary.
        - num_heads (dict): A dictionary mapping the projection layers to the number of heads.
        - ignored_layers (list): A list of layers to be ignored during pruning.
    """
    num_heads = dict()
    for m in model.modules():
        if isinstance(m, LlamaAttention):
            num_heads[m.q_proj] = m.num_heads
            num_heads[m.k_proj] = m.num_key_value_heads
            num_heads[m.v_proj] = m.num_key_value_heads

    ignored_layers = [model.lm_head]
    return dict(), num_heads, ignored_layers


def get_vit_params(model: nn.Module, smash_config: SmashConfigPrefixWrapper) -> Tuple[Dict, Dict, List]:
    """
    Get the parameters related to the Vision Transformer (ViT) model.

    Parameters
    ----------
    model : nn.Module
        The Vision Transformer model.
    smash_config : SmashConfigPrefixWrapper
        A dictionary containing pruning parameters.

    Returns
    -------
    tuple
        A tuple containing:
        - ch_groups (dict): Channel groups mapped to corresponding layers.
        - num_heads (dict): Attention heads mapped to corresponding layers.
        - ignored_layers (list): Layers to be ignored during pruning.
    """
    ch_groups: Dict[Any, Any] = dict()
    num_heads: Dict[Any, Any] = dict()
    ignored_layers: List[Any] = [model.vit.embeddings, model.classifier]
    for m in model.modules():
        if isinstance(m, ViTSelfAttention):
            num_heads[m.query] = m.num_attention_heads
            num_heads[m.key] = m.num_attention_heads
            num_heads[m.value] = m.num_attention_heads

    return ch_groups, num_heads, ignored_layers


def add_grad_checkpointing(model: nn.Module, pruning_device: torch.device) -> nn.Module:
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
    model: nn.Module,
    calibration_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    smash_config: SmashConfigPrefixWrapper,
    calibration_data_size: int = 4096,
) -> nn.Module:
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


def update_dimensions_post_pruning(model: nn.Module, pruner: Any, imported_modules: Dict) -> nn.Module:
    """
    Update the pruned model by updating the attention parameters based on the pruner's configuration.

    Parameters
    ----------
    model : nn.Module
        The pruned model.
    pruner : Any
        The pruner object containing the configuration for pruning.
    imported_modules : Dict
        The imported modules.

    Returns
    -------
    nn.Module
        The postprocessed pruned model with updated attention parameters.
    """
    if isinstance(model, imported_modules["MViT"]):
        for m in model.modules():
            if isinstance(m, imported_modules["MultiScaleAttention"]):
                m.num_heads = pruner.num_heads[m.qkv]

    elif isinstance(model, ViT):
        for m in model.modules():
            if isinstance(m, ViTSelfAttention):
                m.num_attention_heads = pruner.num_heads[m.query]
                m.attention_head_size = m.query.out_features // m.num_attention_heads
                m.all_head_size = m.query.out_features

    elif isinstance(model, Opt):
        for m in model.modules():
            if isinstance(m, OPTAttention):
                m.num_heads = pruner.num_heads[m.q_proj]
                m.head_dim = m.q_proj.out_features // m.num_heads
                m.embed_dim = m.head_dim * m.num_heads

    elif isinstance(model, Llama):
        for n, m in model.named_modules():
            if isinstance(m, LlamaAttention):
                # override attention parameters to handle pruned model
                # handles both prune_num_heads and prune_head_dims
                m.num_heads = cast(int, pruner.num_heads[m.q_proj])  # type: ignore[assignment]
                m.num_key_value_heads = pruner.num_heads[m.k_proj]
                m.head_dim = m.q_proj.out_features // m.num_heads
                m.hidden_size = m.head_dim * m.num_heads

            elif isinstance(m, LlamaRotaryEmbedding):
                # override forward function to handle pruned head dimension
                m.forward = llama_rotary_embedding_forward.__get__(m, LlamaRotaryEmbedding)
    return model


def llama_rotary_embedding_forward(
    self: Any, x: torch.Tensor, seq_len: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the llama_rotary_embedding_forward.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor of shape [bs, num_attention_heads, seq_len, head_size].
    seq_len : int, optional
        The length of the sequence. If None, the length is determined from the input tensor.

    Returns
    -------
    tuple
        A tuple containing:
        - cos_cached (torch.Tensor): Cosine tensor of shape [seq_len, head_size].
        - sin_cached (torch.Tensor): Sine tensor of shape [seq_len, head_size].
    """
    if seq_len > self.max_seq_len_cached:
        self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
    return (
        self.cos_cached[:seq_len, : x.shape[-1]].to(dtype=x.dtype),
        self.sin_cached[:seq_len, : x.shape[-1]].to(dtype=x.dtype),
    )

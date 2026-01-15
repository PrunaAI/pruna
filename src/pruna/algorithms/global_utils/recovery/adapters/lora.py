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

from typing import Any

import torch
from ConfigSpace import CategoricalHyperparameter, Constant, OrdinalHyperparameter
from diffusers.models import (
    FluxTransformer2DModel,
    SanaTransformer2DModel,
    UNet2DConditionModel,
)
from peft import LoraConfig, PeftMixedModel, PeftModel, get_peft_model
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities.seed import isolate_rng

from pruna.algorithms.global_utils.recovery.adapters import PrunaAdapter
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.engine.model_checks import is_causal_lm
from pruna.engine.utils import get_device
from pruna.logging.logger import pruna_logger


class LoraAdapter(PrunaAdapter):
    """Adapter for LoRA finetuning."""

    adapter_prefix = "lora"

    @classmethod
    def get_hyperparameters(cls, task_name: str, **override_defaults: Any) -> list:
        """
        Configure all algorithm-specific hyperparameters with ConfigSpace.

        Parameters
        ----------
        task_name : str
            The name of the task, e.g. "text-to-image" or "text-to-text".
        **override_defaults : Any
            Values used to override the default hyperparameters when using multiple finetuners together.

        Returns
        -------
        list
            The hyperparameters.
        """
        if task_name == "text_to_text":
            # default values are based on
            # https://github.com/huggingface/smollm/blob/6f2fbbb76f77c2f0db355a9d3cd2167ae2a11854/finetuning/train.py,
            default_hyperparameters = {
                "r": 8,
                "alpha_r_ratio": 2.0,
                "target_modules": None,  # None is handled by the peft package in peft.utils.constants
                # see TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
                "dropout": 0.05,
                "variant": "lora",
            }
            default_hyperparameters.update(override_defaults)
            return [
                OrdinalHyperparameter(
                    "r",
                    sequence=[4, 8, 16, 32, 64, 128],
                    default_value=default_hyperparameters["r"],
                    meta=dict(desc="Rank of the LoRA layers."),
                ),
                OrdinalHyperparameter(
                    "alpha_r_ratio",
                    sequence=[0.5, 1.0, 2.0],
                    default_value=default_hyperparameters["alpha_r_ratio"],
                    meta=dict(desc="Alpha/Rank ratio of the LoRA layers."),
                ),
                CategoricalHyperparameter(
                    "target_modules",
                    choices=[None, "all-linear"],
                    default_value=default_hyperparameters["target_modules"],
                    meta=dict(desc="Target modules for the LoRA layers."),
                ),
                Constant(
                    "dropout",
                    default_hyperparameters["dropout"],
                    meta=dict(desc="Dropout rate of the LoRA layers during training."),
                ),
                CategoricalHyperparameter(
                    "variant",
                    choices=["lora", "pissa"],
                    default_value=default_hyperparameters["variant"],
                    meta=dict(desc="Variant of the LoRA adapter."),
                ),
            ]

        elif task_name == "text_to_image":
            # default values are based on
            # https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py,
            default_hyperparameters = {
                "r": 4,
                "alpha_r_ratio": 1.0,
                "target_modules": None,
                "dropout": 0.0,
                "variant": "lora",
            }
            default_hyperparameters.update(override_defaults)
            return [
                OrdinalHyperparameter(
                    "r",
                    sequence=[4, 8, 16, 32, 64, 128],
                    default_value=default_hyperparameters["r"],
                    meta=dict(desc="Rank of the LoRA layers."),
                ),
                OrdinalHyperparameter(
                    "alpha_r_ratio",
                    sequence=[0.5, 1.0, 2.0],
                    default_value=default_hyperparameters["alpha_r_ratio"],
                    meta=dict(desc="Alpha/Rank ratio of the LoRA layers."),
                ),
                Constant(
                    "target_modules", default_hyperparameters["target_modules"]
                ),  # default choice depends on the model, allow user to choose in future
                Constant("dropout", default_hyperparameters["dropout"]),
                CategoricalHyperparameter(
                    "variant",
                    choices=["lora", "pissa"],
                    default_value=default_hyperparameters["variant"],
                    meta=dict(desc="Variant of the LoRA adapter."),
                ),
            ]
        else:
            raise ValueError(f"Task '{task_name}' is not yet supported for LoRA recovery.")

    @classmethod
    def activate(
        cls,
        model: torch.nn.Module,
        smash_config: SmashConfigPrefixWrapper,
        seed: int | None = None,
    ) -> tuple[torch.nn.Module, int, int]:
        """
        Create LoRA layers.

        Parameters
        ----------
        model : torch.nn.Module
            The model to attach LoRA layers to.
        smash_config : SmashConfigPrefixWrapper
            The configuration to use for defining LoRA layers.
        seed : int | None
            The seed used to reproducibly initialize the LoRA layers.

        Returns
        -------
        torch.nn.Module
            The model with the LoRA layers activated.
        int
            The number of trainable LoRA parameters.
        int
            The number of skipped LoRA parameters.
        """
        # save active parameters, device and dtype to restore after getting peft model
        active_parameters = [param for param in model.parameters() if param.requires_grad]
        device = get_device(model)

        if smash_config["variant"] == "lora":
            # define LoRA layers
            target_modules = smash_config["target_modules"]
            if target_modules is None:
                target_modules = cls.get_default_target_modules(model)
            lora_config = LoraConfig(
                r=smash_config["r"],
                lora_alpha=int(smash_config["alpha_r_ratio"] * smash_config["r"]),
                lora_dropout=float(smash_config["dropout"]),
                target_modules=target_modules,
                bias="none",
            )
            model = _get_peft_model_with_seed(model, lora_config, seed=seed)
        elif smash_config["variant"] == "pissa":
            model = PeftModel.from_pretrained(
                model, smash_config.cache_dir / "pissa_weights", is_trainable=True, torch_device=device
            )
        else:
            raise ValueError(f"Invalid LoRA variant: {smash_config['variant']}")
        model.to(device=device)

        # count trainable LoRA parameters
        num_lora_params = sum(p.numel() for name, p in model.named_parameters() if "lora" in name and p.requires_grad)

        # restore active parameters
        for param in active_parameters:
            param.requires_grad = True
        return model, num_lora_params, 0

    @classmethod
    def pre_smash_hook(
        cls, model: torch.nn.Module, smash_config: SmashConfigPrefixWrapper, seed: int | None = None
    ) -> None:
        """
        Compute LoRA weights before smashing in case of a variant that requires the original weights.

        PiSSA initilization involves changing the base weights of the model,
        which we want to apply adapters to during smashing.
        Therefore, we need to apply the PiSSA adapter before smashing and save the adapter weights to a temporary file.
        This file will be loaded during smashing and the adapter weights will be applied to the base weights.

        Parameters
        ----------
        model : torch.nn.Module
            The model to prepare.
        smash_config : SmashConfigPrefixWrapper
            The configuration to use for defining LoRA layers.
        seed : int | None
            The seed used to reproducibly initialize the LoRA layers.
        """
        if smash_config["variant"] == "pissa":
            pruna_logger.info("Performing pre-smash setup for PiSSA adapter.")
            target_modules = smash_config["target_modules"]
            if target_modules is None:
                target_modules = cls.get_default_target_modules(model)
            lora_config = LoraConfig(
                r=smash_config["r"],
                lora_alpha=int(smash_config["alpha_r_ratio"] * smash_config["r"]),
                lora_dropout=float(smash_config["dropout"]),
                target_modules=target_modules,
                bias="none",
                init_lora_weights="pissa_niter_4",  # type: ignore[arg-type]
            )
            model = _get_peft_model_with_seed(model, lora_config, seed=seed)
            # reset LoRA initialization to default to avoid computing PiSSA weights a second time when loading
            model.peft_config["default"].init_lora_weights = True  # type: ignore[attr-defined]
            pruna_logger.info(f"Saving PiSSA weights to {smash_config.cache_dir / 'pissa_weights'}")
            model.save_pretrained(smash_config.cache_dir / "pissa_weights")
            model.unload()

    @staticmethod
    def get_default_target_modules(model: Any) -> list[str] | None:
        """
        Return default target modules based on huggingface's finetuning scripts.

        Parameters
        ----------
        model : Any
            The model to get the default target modules for.

        Returns
        -------
        list[str] | None
            The default target modules.
        """
        if is_causal_lm(model):
            return None
        elif isinstance(model, UNet2DConditionModel):  # SD and SDXL
            return ["to_k", "to_q", "to_v", "to_out.0"]
        elif isinstance(model, SanaTransformer2DModel):  # Sana
            return ["to_k", "to_q", "to_v"]
        elif isinstance(model, FluxTransformer2DModel):  # Flux
            return [
                "attn.to_k",
                "attn.to_q",
                "attn.to_v",
                "attn.to_out.0",
                "attn.add_k_proj",
                "attn.add_q_proj",
                "attn.add_v_proj",
                "attn.to_add_out",
                "ff.net.0.proj",
                "ff.net.2",
                "ff_context.net.0.proj",
                "ff_context.net.2",
            ]
        else:
            pruna_logger.warning(
                "Could not infer the target modules in the pipeline, "
                "falling back to peft-defined defaults if peft recognizes the model architecture."
            )
            # let the peft package handle the default target modules (see is_causal_lm case)
            return None


def _get_peft_model_with_seed(*args: Any, seed: int | None = None, **kwargs: Any) -> PeftModel | PeftMixedModel:
    """
    Call get_peft_model with a seed for reproducible initialization.

    Parameters
    ----------
    *args
        The arguments to pass to get_peft_model.
    seed : int | None
        The seed to use for the model.
    **kwargs
        The keyword arguments to pass to get_peft_model.

    Returns
    -------
    PeftModel | PeftMixedModel
        The peft model.
    """
    if seed is None:
        return get_peft_model(*args, **kwargs)

    with isolate_rng():
        seed_everything(seed, verbose=False)
        model = get_peft_model(*args, **kwargs)  # type: ignore[arg-type]

    return model

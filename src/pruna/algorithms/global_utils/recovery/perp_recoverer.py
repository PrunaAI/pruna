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

from typing import Any, Dict

import torch
from ConfigSpace import Constant

from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag
from pruna.algorithms.global_utils.recovery.adapters.utils import freeze_parameters
from pruna.algorithms.global_utils.recovery.finetuners import PrunaFinetuner
from pruna.algorithms.global_utils.recovery.finetuners.diffusers.utils import get_denoiser_attr
from pruna.algorithms.global_utils.recovery.utils import get_trainable_parameters
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.engine.model_checks import (
    is_causal_lm,
    is_flux_pipeline,
    is_sana_pipeline,
    is_sd_pipeline,
    is_sdxl_pipeline,
)
from pruna.engine.save import SAVE_FUNCTIONS
from pruna.logging.logger import pruna_logger


class PERPRecoverer(PrunaAlgorithmBase):
    """
    General purpose PERP recoverer using norm, head and bias finetuning and optionally HuggingFace's LoRA.

    Parameters
    ----------
    task_name : str
        The name of the task to recover.
    use_lora : bool
        Whether to use LoRA adapters, which will not be merged and therefore slow down inference.
    use_in_place : bool
        Whether to use norm, bias and head finetuning which will modify the model in place.
    is_distillation : bool
        Whether to use distillation which requires a distillation datamodule, otherwise finetuning is used.
    """

    group_tags: list[AlgorithmTag] = [AlgorithmTag.RECOVERER]  # type: ignore[attr-defined]
    save_fn = SAVE_FUNCTIONS.pickled
    references: dict[str, str] = {
        "GitHub": "https://github.com/huggingface/peft",
        "Paper": "https://arxiv.org/pdf/2312.15230",
    }
    processor_required: bool = False
    dataset_required: bool = True
    runs_on: list[str] = ["cpu", "cuda"]

    def __init__(self, task_name: str, use_lora: bool, use_in_place: bool, is_distillation: bool) -> None:
        self.task_name = task_name
        self.tokenizer_required = task_name == "text_to_text"  # type: ignore[misc]

        if not use_lora and not use_in_place:
            raise ValueError("Arguments use_lora and use_in_place cannot both be False, please use one of the two.")
        self.use_lora = use_lora
        self.use_in_place = use_in_place
        self.is_distillation = is_distillation
        # define all used types of adapters
        self.adapters = []
        if self.use_in_place:
            self.adapters.append("NormAdapter")
            self.adapters.append("BiasAdapter")
            if self.task_name == "text_to_text":
                self.adapters.append("HeadAdapter")
        if self.use_lora:
            self.adapters.append("LoraAdapter")

        # The recoverer receives a single seed to create a seed generator to seed any adapter initialization and the
        # actual distillation. We don't know at which point in the application of the algorithmth, adapters are created
        # (during apply or in the pre-smash-hook) so we store a single generator here, which gets initliazed in apply or
        # in the pre-smash hook (whatever is called first) and use this generator for seeding at any point during the
        # application of this algorithm.
        self.seed_generator: torch.Generator | None = None

        super().__init__()  # self.adapters need to be set before calling get_hyperparameters

    def get_hyperparameters(self) -> list:
        """
        Configure all algorithm-specific hyperparameters with ConfigSpace.

        Returns
        -------
        list
            The hyperparameters.
        """
        imported_modules = self.import_algorithm_packages()
        adapters = [imported_modules[adapter] for adapter in self.adapters]

        hyperparameters = []

        # collect adapters hyperparameters and add the adapter's prefix
        for adapter in adapters:
            adapter_hyperparams = adapter.get_hyperparameters(self.task_name)
            for param in adapter_hyperparams:
                param.name = f"{adapter.adapter_prefix}_{param.name}"
            hyperparameters.extend(adapter_hyperparams)

        # collect finetuner hyperparameters
        finetuner_hyperparams = imported_modules["Finetuner"].get_hyperparameters()
        hyperparameters.extend(finetuner_hyperparams)

        seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
        hyperparameters.append(
            Constant(  # set to constant, waiting for user-defined non-optimized hyperparameters
                "seed",
                seed,
                meta=dict(desc="Random seed used for reproducibility."),
            )
        )

        return hyperparameters

    def model_check_fn(self, model: Any) -> bool:
        """
        Check if the model is compatible with PERP.

        Parameters
        ----------
        model : Any
            The model to check.

        Returns
        -------
        bool
            True if the model is a Stable Diffusion or Stable Diffusion XL pipeline, False otherwise.
        """
        if self.task_name == "text_to_image":
            return is_sd_pipeline(model) or is_sdxl_pipeline(model) or is_sana_pipeline(model) or is_flux_pipeline(model)
        elif self.task_name == "text_to_text":
            return is_causal_lm(model)
        else:
            raise ValueError(f"Task name {self.task_name} is not supported for PERP recovery.")

    def _pre_smash_hook(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> None:
        """
        Perform any necessary setup steps before the smashing process begins.

        Parameters
        ----------
        model : Any
            The model to prepare for smashing.
        smash_config : SmashConfig
            Configuration object containing algorithm settings.
        """
        # Identify which components in the pipeline might need to be setup before smashing
        if self.task_name == "text_to_image":
            model_recovery, denoiser_attr_name = get_denoiser_attr(model)
            if model_recovery is None:
                pruna_logger.error("Could not infer the denoiser attribute in the pipeline, skipping recovery.")
                return
        else:  # text_to_text
            model_recovery = model

        # initialize the seed generator if it is not already done, see comment in __init__
        if self.seed_generator is None:
            self.seed_generator = torch.Generator().manual_seed(smash_config["seed"])
        # prepare individual seeds for adapters
        adapter_seeds = [
            int(torch.randint(0, 2**32 - 1, (1,), generator=self.seed_generator).item()) for _ in self.adapters
        ]

        imported_modules = self.import_algorithm_packages()
        adapters = [imported_modules[adapter] for adapter in self.adapters]

        for adapter, adapter_seed in zip(adapters, adapter_seeds):
            adapter_smash_config = SmashConfigPrefixWrapper(smash_config, adapter.adapter_prefix + "_")
            adapter.pre_smash_hook(model_recovery, adapter_smash_config, seed=adapter_seed)

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Recover performances from a given model with a given config.

        Parameters
        ----------
        model : Any
            The model to recover from.
        smash_config : SmashConfig
            The configuration for the recovery.

        Returns
        -------
        Any
            The quantized model.
        """
        # Identify which components in the pipeline will have adapters
        if self.task_name == "text_to_image":
            model_recovery, denoiser_attr_name = get_denoiser_attr(model)
            if model_recovery is None:
                pruna_logger.error("Could not infer the denoiser attribute in the pipeline, skipping recovery.")
                return model
        else:  # text_to_text
            model_recovery = model

        # freeze all parameters so we can add trainable parameters / select a subset for finetuning
        freeze_parameters(model)
        model_recovery.train()  # activate running batch norm / dropout / etc.. as preparation for finetuning

        # store device and dtype and warn user against finetuning on CPU if necessary
        device = smash_config.device
        if device == "cpu" or (hasattr(device, "type") and device.type == "cpu"):
            warning_if_cpu = "Model is on CPU, this is not recommended for recovery as it may take a long time."
            pruna_logger.warning(warning_if_cpu)

        # initialize the seed generator if it is not already done, see comment in __init__
        if self.seed_generator is None:
            self.seed_generator = torch.Generator().manual_seed(smash_config["seed"])
        # # prepare individual seeds for adapters and distllation (e.g. seeds the random initialization of LoRA)
        distillation_seed = int(torch.randint(0, 2**32 - 1, (1,), generator=self.seed_generator).item())
        adapter_seeds = [
            int(torch.randint(0, 2**32 - 1, (1,), generator=self.seed_generator).item()) for _ in self.adapters
        ]

        # activate adapters
        imported_modules = self.import_algorithm_packages()
        adapters = [imported_modules[adapter] for adapter in self.adapters]

        prefixes_used = []
        for adapter, adapter_seed in zip(adapters, adapter_seeds):
            adapter_smash_config = SmashConfigPrefixWrapper(smash_config, adapter.adapter_prefix + "_")
            model_recovery, num_activ_param, num_skip_param = adapter.activate(
                model_recovery, adapter_smash_config, seed=adapter_seed
            )

            # log skipped parameters and record which adapters were actually used
            if num_skip_param > 0:
                pruna_logger.warning(
                    f"Skipped {num_skip_param:.2e} {adapter.adapter_prefix} parameters "
                    "that were not trainable due to quantization."
                )
            elif num_activ_param == 0:  # num_skip_param = 0 too so there is no such parameter
                pruna_logger.info(f"No trainable {adapter.adapter_prefix} parameters found: skipping adapter.")
            else:
                prefixes_used.append(adapter.adapter_prefix)
        model_recovery.to(device=device)

        # check if any parameters were activated
        num_trainable_params = sum(p.numel() for p in get_trainable_parameters(model))
        if num_trainable_params == 0:
            pruna_logger.error("No trainable parameters were activated, skipping recovery.")
            return model
        else:
            pruna_logger.info(
                f"Recovering with PERP: {' + '.join(prefixes_used)}, totaling {num_trainable_params:.2e} parameters."
            )

        # replace the component in the pipeline
        if self.task_name == "text_to_image":
            setattr(model, denoiser_attr_name, model_recovery)
        else:
            model = model_recovery

        # finetune the model
        model = imported_modules["Finetuner"].finetune(model, smash_config, distillation_seed, self.algorithm_name)

        # switch back to eval mode
        model_recovery.eval()  # disable dropout, set batch norm to eval mode, etc..
        freeze_parameters(model_recovery)  # freeze all finetuned parameters for inference

        # remove peft wrapper to recover a model with the same type as the recoverer's input
        if self.use_lora and self.task_name == "text_to_image":
            base_denoiser = getattr(model, denoiser_attr_name).base_model.model
            setattr(model, denoiser_attr_name, base_denoiser)
        elif self.use_lora:
            model = model.base_model.model

        return model

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Provide a algorithm packages for the algorithm.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        from pruna.algorithms.global_utils.recovery.adapters.bias import BiasAdapter
        from pruna.algorithms.global_utils.recovery.adapters.head import HeadAdapter
        from pruna.algorithms.global_utils.recovery.adapters.lora import LoraAdapter
        from pruna.algorithms.global_utils.recovery.adapters.norm import NormAdapter

        Finetuner: type[PrunaFinetuner] | None = None  # noqa: N806
        if self.task_name == "text_to_image" and self.is_distillation:
            from pruna.algorithms.global_utils.recovery.finetuners.text_to_image_distiller import (
                TextToImageDistiller as Finetuner,
            )
        elif self.task_name == "text_to_image":
            from pruna.algorithms.global_utils.recovery.finetuners.text_to_image_finetuner import (
                TextToImageFinetuner as Finetuner,
            )
        elif self.task_name == "text_to_text" and self.is_distillation:
            raise NotImplementedError("Distillation for text-to-text models is not implemented yet.")
        elif self.task_name == "text_to_text":
            from pruna.algorithms.global_utils.recovery.finetuners.text_to_text_finetuner import (
                TextToTextFinetuner as Finetuner,
            )
        else:
            raise ValueError(f"Task name {self.task_name} is not supported for PERP recovery.")

        return dict(
            BiasAdapter=BiasAdapter,
            HeadAdapter=HeadAdapter,
            LoraAdapter=LoraAdapter,
            NormAdapter=NormAdapter,
            Finetuner=Finetuner,
        )

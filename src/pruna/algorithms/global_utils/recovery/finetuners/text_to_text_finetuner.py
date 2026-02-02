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

import tempfile
from typing import Iterator, Optional

import torch
from ConfigSpace import (
    CategoricalHyperparameter,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from pruna.algorithms.global_utils.recovery.finetuners import PrunaFinetuner
from pruna.algorithms.global_utils.recovery.utils import get_dtype, split_defaults
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.logging.logger import pruna_logger


class TextToTextFinetuner(PrunaFinetuner):
    """Finetuner for text-to-text models."""

    @classmethod
    def get_hyperparameters(cls, **override_defaults) -> list:
        """
        Configure all method-specific hyperparameters with ConfigSpace.

        Parameters
        ----------
        **override_defaults : dict
            The hyperparameters to override.

        Returns
        -------
        list
            The hyperparameters.
        """
        defaults: dict[str, int | float | str] = {
            "training_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "num_epochs": 1.0,
            "learning_rate": 2e-4,
            "dataset_text_field": "text",
            "report_to": "none",
            "optimizer": "AdamW8bit",
        }
        defaults.update(override_defaults)
        string_defaults, numeric_defaults = split_defaults(defaults)
        return [
            UniformIntegerHyperparameter(
                "training_batch_size",
                lower=1,
                upper=4096,
                default_value=numeric_defaults["training_batch_size"],
                meta=dict(desc="Batch size for finetuning."),
            ),
            UniformIntegerHyperparameter(
                "gradient_accumulation_steps",
                lower=1,
                upper=1024,
                default_value=numeric_defaults["gradient_accumulation_steps"],
                meta=dict(desc="Number of gradient accumulation steps for finetuning."),
            ),
            UniformFloatHyperparameter(
                "num_epochs",
                lower=0.0,
                upper=4096.0,
                default_value=numeric_defaults["num_epochs"],
                meta=dict(desc="Number of epochs for finetuning."),
            ),
            UniformFloatHyperparameter(
                "learning_rate",
                lower=0.0,
                upper=1.0,
                default_value=numeric_defaults["learning_rate"],
                meta=dict(desc="Learning rate for finetuning."),
            ),
            Constant("dataset_text_field", string_defaults["dataset_text_field"]),
            CategoricalHyperparameter(
                "report_to",
                choices=["none", "wandb", "tensorboard"],
                default_value=string_defaults["report_to"],
                meta=dict(desc="Where to report the finetuning results."),
            ),
            CategoricalHyperparameter(
                "optimizer",
                choices=["AdamW", "AdamW8bit", "PagedAdamW8bit"],
                default_value=string_defaults["optimizer"],
                meta=dict(desc="Which optimizer to use for finetuning."),
            ),
        ]

    @classmethod
    def finetune(
        cls,
        model: torch.nn.Module,
        smash_config: SmashConfigPrefixWrapper,
        seed: int,
        recoverer: str,
        report_every_n_samples: int | None = None,
    ) -> torch.nn.Module:
        """
        Finetune the model's previously activated parameters.

        Parameters
        ----------
        model : torch.nn.Module
            The model to apply the finetuner to.
        smash_config : SmashConfigPrefixWrapper
            The configuration for the finetuner.
        seed : int
            The seed to use for reproducibility.
        recoverer : str
            The recoverer used, i.e. the selection of parameters to finetune. This is only used for logging purposes.
        report_every_n_samples : int | None, optional
            The number of samples between reports to the logger.
            If None, the number of samples between reports is set to 1/8 of the dataset size.

        Returns
        -------
        torch.nn.Module
            The finetuned model.
        """
        dtype = get_dtype(model)

        # format dataset
        dataset, dataset_text_field = cls._format_dataset_for_causal_lm(
            # dataloader can't be None because of the dataset_required flag
            smash_config.train_dataloader().dataset,  # type: ignore[union-attr]
            smash_config["dataset_text_field"],
        )

        # setup optimizer
        if smash_config["optimizer"] == "AdamW8bit":
            optim = "adamw_bnb_8bit"
        elif smash_config["optimizer"] == "PagedAdamW8bit":
            optim = "paged_adamw_8bit"
        else:
            optim = "adamw_torch"

        # setup training
        model.train()
        if report_every_n_samples is None:
            report_every_n_samples = max(1, len(dataset) // 8)
        with tempfile.TemporaryDirectory(prefix=str(smash_config["cache_dir"])) as temp_dir:
            training_args = SFTConfig(
                # task
                dataset_text_field=dataset_text_field,
                optim_target_modules=["lora"],
                # batch size
                per_device_train_batch_size=smash_config["training_batch_size"],
                gradient_accumulation_steps=smash_config["gradient_accumulation_steps"],
                # optimization
                warmup_steps=100,
                num_train_epochs=smash_config["num_epochs"],
                learning_rate=smash_config["learning_rate"],
                lr_scheduler_type="cosine",
                weight_decay=0.01,
                fp16=(dtype == torch.float16),
                bf16=(dtype == torch.bfloat16),
                optim=optim,
                # logging
                logging_strategy="steps",
                logging_steps=report_every_n_samples,
                disable_tqdm=False,
                report_to=smash_config["report_to"],
                # saving
                run_name=f"Recovery-{recoverer}",
                output_dir=temp_dir,
                seed=seed,
            )
            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                processing_class=smash_config.tokenizer,
                args=training_args,
            )
            trainer.train()

        # Get the unwrapped model
        model = trainer.accelerator.unwrap_model(trainer.model)
        model.eval()
        model = model.to(dtype=dtype)

        return model

    @staticmethod
    def _format_dataset_for_causal_lm(
        dataset: Dataset | torch.utils.data.Dataset, text_field: str
    ) -> tuple[Dataset, Optional[str]]:
        """
        Format a dataset for SFTTrainer.

        Parameters
        ----------
        dataset : Dataset | torch.utils.data.Dataset
            The dataset to format.
        text_field : str
            The text field to use.

        Returns
        -------
        tuple[Dataset, Optional[str]]
            The formatted dataset and dataset text field.
            A text field is only provided if the dataset is a huggingface dataset not yet tokenized.
        """
        # Handle HuggingFace Dataset - only access column_names for this type
        if isinstance(dataset, Dataset):
            column_names = dataset.column_names
            if "input_ids" in column_names:
                # processed dataset with no need for tokenization
                # remove all other columns, otherwise HF's Trainer tends to infer the wrong task from those columns
                removed_columns = [col for col in column_names if col not in [text_field, "input_ids"]]
                return dataset.remove_columns(removed_columns), None
            elif text_field in column_names:
                # raw dataset with text field
                # remove all other columns, otherwise HF's Trainer tends to infer the wrong task from those columns
                removed_columns = [col for col in column_names if col != text_field]
                return dataset.remove_columns(removed_columns), text_field
            else:
                pruna_logger.error(
                    "The dataset provided for recovery is not compatible. Accepted format include:\n"
                    " - huggingface datasets with a text field,\n"
                    " - huggingface datasets with an input_ids field,\n"
                    " - (input, label) format for next token prediction,\n"
                    " - (extended_inputs,) format for next token prediction."
                )
                raise ValueError(f"Expected a HuggingFace dataset with text or input_ids fields for LoRA recovery but got: {dataset}")

        # Handle PyTorch Dataset
        if isinstance(dataset, torch.utils.data.Dataset):
            first_sample = dataset[0]
            
            # Check if already in (extended_inputs,) format - single tensor or tuple/list with one tensor
            if isinstance(first_sample, torch.Tensor):
                # Single tensor format
                def data_generator() -> Iterator[dict[str, torch.Tensor]]:
                    for idx in range(len(dataset)):  # type: ignore[arg-type]
                        input_ids = dataset[idx]
                        if not isinstance(input_ids, torch.Tensor):
                            input_ids = torch.tensor(input_ids)
                        attention_mask = torch.ones_like(input_ids)
                        yield {"input_ids": input_ids, "attention_mask": attention_mask}
                return Dataset.from_generator(data_generator), None
            elif isinstance(first_sample, (tuple, list)) and len(first_sample) == 1:
                # Tuple/list with single element - already in extended format
                if isinstance(first_sample[0], torch.Tensor):
                    def data_generator() -> Iterator[dict[str, torch.Tensor]]:
                        for idx in range(len(dataset)):  # type: ignore[arg-type]
                            sample = dataset[idx]
                            input_ids = sample[0] if isinstance(sample, (tuple, list)) else sample
                            if not isinstance(input_ids, torch.Tensor):
                                input_ids = torch.tensor(input_ids)
                            attention_mask = torch.ones_like(input_ids)
                            yield {"input_ids": input_ids, "attention_mask": attention_mask}
                    return Dataset.from_generator(data_generator), None
            
            # Check if in (input, label) format for next token prediction
            if isinstance(first_sample, (tuple, list)) and len(first_sample) == 2:
                data_input, label = first_sample
                if (
                    isinstance(data_input, torch.Tensor)
                    and isinstance(label, torch.Tensor)
                    and len(data_input) > 0
                    and len(label) > 0
                    and torch.all(data_input[1:] == label[:-1])
                ):
                    # (input, label) format with single token shift
                    def data_generator() -> Iterator[dict[str, torch.Tensor]]:
                        for idx in range(len(dataset)):  # type: ignore[arg-type]
                            data_input, label = dataset[idx]
                            # append last token of label to input for next token prediction
                            input_ids = torch.cat((data_input, label[..., -1:]))  # this conversion slows finetuning a little
                            attention_mask = torch.ones_like(input_ids)
                            yield {"input_ids": input_ids, "attention_mask": attention_mask}
                    return Dataset.from_generator(data_generator), None
            
            # If we get here, the torch dataset format is not recognized
            pruna_logger.error(
                "The dataset provided for recovery is not compatible. Accepted format include:\n"
                " - huggingface datasets with a text field,\n"
                " - huggingface datasets with an input_ids field,\n"
                " - (input, label) format for next token prediction,\n"
                " - (extended_inputs,) format for next token prediction."
            )
            raise ValueError(f"Expected a torch dataset in (input, label) or (extended_inputs,) format for LoRA recovery but got: {dataset}")

        # Unknown dataset type
        pruna_logger.error(
            "The dataset provided for recovery is not compatible. Accepted format include:\n"
            " - huggingface datasets with a text field,\n"
            " - huggingface datasets with an input_ids field,\n"
            " - (input, label) format for next token prediction,\n"
            " - (extended_inputs,) format for next token prediction."
        )
        raise ValueError(f"Expected a Dataset or torch.utils.data.Dataset for LoRA recovery but got: {type(dataset)}")

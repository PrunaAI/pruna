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

import functools
import random
from typing import Any, List, Literal

import pytorch_lightning as pl
import torch
from diffusers.optimization import get_scheduler
from diffusers.utils import BaseOutput
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.seed import isolate_rng

try:
    from bitsandbytes.optim import AdamW8bit  # type: ignore[import-untyped]
except ImportError:
    AdamW8bit = None

import pathlib

from ConfigSpace import (
    CategoricalHyperparameter,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from pruna.algorithms.global_utils.recovery.finetuners import PrunaFinetuner
from pruna.algorithms.global_utils.recovery.finetuners.diffusers import utils
from pruna.algorithms.global_utils.recovery.finetuners.diffusers.distillation_arg_utils import (
    get_latent_replacement_fn,
)
from pruna.algorithms.global_utils.recovery.utils import (
    filter_kwargs,
    get_dtype,
    get_trainable_parameters,
    split_defaults,
)
from pruna.config.hyperparameters import Boolean
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.data.diffuser_distillation_data_module import DiffusionDistillationDataModule
from pruna.engine.utils import get_device, get_device_type
from pruna.logging.logger import pruna_logger


class TextToImageDistiller(PrunaFinetuner):
    """Distiller for text-to-image models."""

    @classmethod
    def get_hyperparameters(cls, **override_defaults) -> List:
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
        defaults = {
            "training_batch_size": 0,  # 0: all steps of the diffusion process are used
            "gradient_accumulation_steps": 1,
            "num_epochs": 1.0,
            "validate_every_n_epoch": 1.0,
            "learning_rate": 1e-4,
            "weight_decay": 1e-2,
            "report_to": "none",
            "optimizer": "AdamW8bit" if torch.cuda.is_available() else "AdamW",  # AdamW8bit from BnB assumes CUDA
            "lr_decay": 0.5,
            "warmup_steps": 0,
        }
        defaults.update(override_defaults)
        string_defaults, numeric_defaults = split_defaults(defaults)

        return [
            # not a true batch size, but suggests the increase of VRAM coming with higher batch_size
            UniformIntegerHyperparameter(
                "training_batch_size",
                lower=0,
                upper=4096,
                default_value=numeric_defaults["training_batch_size"],
                meta=dict(desc="Number of steps from each diffusion process to use for distillation."),
            ),
            UniformIntegerHyperparameter(
                "gradient_accumulation_steps",
                lower=1,
                upper=1024,
                default_value=numeric_defaults["gradient_accumulation_steps"],
                meta=dict(desc="Number of captions processed to estimate each gradient step."),
            ),
            UniformIntegerHyperparameter(
                "num_epochs",
                lower=0,
                upper=4096,
                default_value=numeric_defaults["num_epochs"],
                meta=dict(desc="Number of epochs for distillation."),
            ),
            UniformFloatHyperparameter(
                "validate_every_n_epoch",
                lower=0.0,
                upper=4096.0,
                default_value=numeric_defaults["validate_every_n_epoch"],
                meta=dict(
                    desc="Number of epochs between each round of validation and model checkpointing. "
                    "If the value is between 0 and 1, validation will be performed multiple times per epoch, "
                    "e.g. 1/8 will result in 8 validations per epoch."
                ),
            ),
            UniformFloatHyperparameter(
                "learning_rate",
                lower=0.0,
                upper=1.0,
                default_value=numeric_defaults["learning_rate"],
                meta=dict(desc="Learning rate for distillation."),
            ),
            Constant("weight_decay", numeric_defaults["weight_decay"]),
            # report_to: for consistency with text-to-text-lora but wandb and tensorboard are not supported yet
            Constant("report_to", string_defaults["report_to"]),
            Boolean(
                "use_cpu_offloading",
                default=False,
                meta=dict(desc="Whether to use CPU offloading for distillation."),
            ),
            CategoricalHyperparameter(
                "optimizer",
                choices=["AdamW8bit", "AdamW", "Adam"],
                default_value=string_defaults["optimizer"],
                meta=dict(desc="Which optimizer to use for distillation."),
            ),
            UniformFloatHyperparameter(
                "lr_decay",
                lower=0.0,
                upper=1.0,
                default_value=numeric_defaults["lr_decay"],
                meta=dict(desc="Learning rate decay, applied at each epoch."),
            ),
            UniformIntegerHyperparameter(
                "warmup_steps",
                lower=0,
                upper=2**14,
                default_value=numeric_defaults["warmup_steps"],
                meta=dict(desc="Number of warmup steps for the learning rate scheduler."),
            ),
        ]

    @classmethod
    def finetune(cls, pipeline: Any, smash_config: SmashConfigPrefixWrapper, seed: int, recoverer: str) -> Any:
        """
        Train the model previously activated parameters on distillation data extracted from the original model.

        Parameters
        ----------
        pipeline : Any
            The pipeline containing components to finetune.
        smash_config : SmashConfigPrefixWrapper
            The configuration for the finetuner.
        seed : int
            The seed to use for reproducibility.
        recoverer : str
            The recoverer used, i.e. the selection of parameters to finetune. This is only used for logging purposes.

        Returns
        -------
        Any
            The finetuned pipeline.
        """
        if not isinstance(smash_config.data, DiffusionDistillationDataModule):
            raise ValueError(
                f"DiffusionDistillation data module is required for distillation, but got {smash_config.data}."
            )

        dtype = get_dtype(pipeline)
        device = get_device(pipeline)
        try:
            lora_r = smash_config["lora_r"]
        except KeyError:
            lora_r = 0

        # split seed into two rng: generator for the dataloader and a seed for the training part
        generator = torch.Generator().manual_seed(seed)
        fit_seed = int(torch.randint(0, 2**32 - 1, (1,), generator=generator).item())

        # Dataloaders (batch size is used to decide how many diffusion steps to backprop)
        train_dataloader = smash_config.train_dataloader(generator=generator)
        val_dataloader = smash_config.val_dataloader()

        # Finetune the model
        trainable_distiller = DistillerTL(
            pipeline,
            smash_config["training_batch_size"],
            smash_config["gradient_accumulation_steps"],
            smash_config["optimizer"],
            smash_config["learning_rate"],
            smash_config["lr_decay"],
            smash_config["warmup_steps"],
            smash_config["weight_decay"],
            lora_r,
            recoverer,
            smash_config["use_cpu_offloading"],
            pipeline_kwargs=smash_config.data.pipeline_kwargs,
        )
        # make directory for logs and checkpoints
        model_path = pathlib.Path(smash_config.cache_dir) / "recovery"
        model_path.mkdir(parents=True)

        early_stopping = EarlyStopping(monitor="validation_loss", patience=3, mode="min", check_finite=True)
        checkpoint_callback = ModelCheckpoint(
            monitor="validation_loss",
            save_top_k=1,
            mode="min",
            every_n_epochs=1,
            dirpath=model_path,
            filename="model-{epoch:02d}-{validation_loss:.4f}",
        )
        callbacks = [early_stopping, checkpoint_callback]

        if smash_config["validate_every_n_epoch"] >= 1.0:
            check_val_every_n_epoch = int(smash_config["validate_every_n_epoch"])
            val_check_interval = None
        else:
            check_val_every_n_epoch = 1
            val_check_interval = smash_config["validate_every_n_epoch"]

        precision: Literal["16-true", "bf16-true", "32"]
        if dtype == torch.float16:
            precision = "16-true"
        elif dtype == torch.bfloat16:
            precision = "bf16-true"
        else:
            precision = "32"

        accelerator = get_device_type(pipeline)
        if accelerator == "accelerator":
            accelerator = "auto"
        trainer = pl.Trainer(
            callbacks=callbacks,
            max_epochs=smash_config["num_epochs"],
            inference_mode=False,
            precision=precision,
            log_every_n_steps=smash_config["gradient_accumulation_steps"],
            check_val_every_n_epoch=check_val_every_n_epoch,
            val_check_interval=val_check_interval,
            logger=False,
            num_sanity_val_steps=0,  # the train.validate already acts as a sanity check
            accelerator=accelerator,
        )

        with isolate_rng():
            pl.seed_everything(fit_seed)
            trainer.validate(trainable_distiller, dataloaders=val_dataloader, verbose=False)
            trainer.fit(trainable_distiller, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        # Loading the best checkpoint is slow and currently creates conflicts with some quantization algorithms,
        # e.g. diffusers_int8. Skipping calling DenoiserTL.load_from_checkpoint for now.
        return pipeline.to(device)


class DistillerTL(pl.LightningModule):
    """
    Pipeline in LightningModule format for distilling the denoiser.

    Parameters
    ----------
    pipeline : Any
        The pipeline to finetune.
    batch_size : int
        The batch size to use for distillation, used to extract a subset of steps from each diffusion process.
    gradient_accumulation_steps : int
        The number of prompts processed to estimate each gradient step.
    optimizer_name : str
        The name of the optimizer to use, options are "AdamW8bit", or optimizers from torch.optim.
    learning_rate : float
        The learning rate to use for finetuning.
    lr_decay : float
        The learning rate decay to use for finetuning.
    warmup_steps : int
        The number of warmup steps to use for finetuning.
    weight_decay : float
        The weight decay to use for finetuning.
    lora_r : int
        The rank of the LoRA matrices.
    recoverer : str
        The recoverer used, i.e. the selection of parameters to finetune. This is only used for logging purposes.
    use_cpu_offloading : bool, optional
        Whether to use CPU offloading for finetuning.
    pipeline_kwargs : dict[str, Any], optional
        Additional keyword arguments to pass to the pipeline, such as `guidance_scale` or `num_inference_steps`.
    """

    def __init__(
        self,
        pipeline: Any,
        batch_size: int,
        gradient_accumulation_steps: int,
        optimizer_name: str,
        learning_rate: float,
        lr_decay: float,
        warmup_steps: int,
        weight_decay: float,
        lora_r: int,
        recoverer: str,
        use_cpu_offloading: bool = False,
        pipeline_kwargs: dict[str, Any] = {},
    ):
        super().__init__()

        self.pipeline = pipeline
        self.latent_replacement_fn = get_latent_replacement_fn(pipeline)

        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.lora_r = lora_r
        self.recoverer = recoverer
        self.use_cpu_offloading = use_cpu_offloading
        self.pipeline_kwargs = pipeline_kwargs
        self.save_hyperparameters(ignore=["pipeline"])
        # register the denoiser explicitly to work with self.parameters() and other LightningModule methods
        self.denoiser, _ = utils.get_denoiser_attr(pipeline)
        self.num_previous_steps = 0
        if self.denoiser is None:
            raise ValueError("Could not find the denoiser in the pipeline.")
        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.log("num_trainable_params", num_trainable_params)

        # basic logging
        self.val_losses: List[torch.Tensor] = []
        self.train_losses: List[torch.Tensor] = []
        self.automatic_optimization = False

    def forward(
        self,
        caption: str,
        latent_inputs: torch.Tensor,
        latent_targets: torch.Tensor,
        seed: int,
        active_steps: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the denoised latents from the input latents recorded at all timesteps.

        Parameters
        ----------
        caption : str
            The caption to use for the denoiser.
        latent_inputs : torch.Tensor
            The inputs of the pipeline with shape (number_of_steps, *latent_shape) recorded in the distillation dataset.
        latent_targets : torch.Tensor
            The outputs of the pipeline with shape (number_of_steps, *latent_shape) recorded in the distillation dataset.
        seed : int
            The seed to used when recording the distillation dataset.
        active_steps : list[int] | None, optional
            The steps to use for the distillation. If None (default), all steps are used.

        Returns
        -------
        torch.Tensor
            The denoised latents at each active step.
        torch.Tensor
            The loss for each active diffusion step.
        """
        # some variables accessible within the denoiser's monkey patched forward
        self.num_previous_steps = 0
        latent_outputs = []
        diffusion_step_losses = []
        original_forward = self.denoiser.forward  # type: ignore[union-attr]

        is_training = active_steps is not None
        is_first_training_step = is_training and self.num_previous_steps == 0

        @functools.wraps(original_forward)
        def distillation_forward(*args, **kwargs):
            if self.use_cpu_offloading and is_first_training_step:
                utils.move_secondary_components(self.pipeline, "cpu")

            # current denoising_step is self.num_previous_steps
            recorded_input = latent_inputs[self.num_previous_steps]

            # select which steps to record: all during validation, only trained steps during training
            if active_steps is None or self.num_previous_steps in active_steps:
                with torch.set_grad_enabled(is_training):  # diffusers disable gradients, re-enable them for training
                    args, kwargs = self.latent_replacement_fn(recorded_input, args, kwargs)
                    output = original_forward(*args, **kwargs)
                    latent_output = (
                        output["sample"] if ("return_dict" in kwargs and kwargs["return_dict"]) else output[0]
                    )
                    loss = self.loss(latent_output, latent_targets[self.num_previous_steps])
                    if is_training:
                        accumulation_normalized_loss = loss / (len(active_steps) * self.gradient_accumulation_steps)
                        self.manual_backward(accumulation_normalized_loss)
                diffusion_step_losses.append(loss)
                latent_outputs.append(latent_output)

            # recreate the expected output format
            if "return_dict" in kwargs and kwargs["return_dict"]:
                recorded_output = self._get_denoiser_output_object(latent_targets[self.num_previous_steps])
            else:
                recorded_output = (latent_targets[self.num_previous_steps],)

            self.num_previous_steps += 1
            return recorded_output

        self.denoiser.forward = distillation_forward  # type: ignore[union-attr]

        # Run the pipeline on the recorded latents and collect the outputs
        _ = self.pipeline(caption, generator=torch.Generator().manual_seed(seed), **self.pipeline_kwargs)
        stacked_latent_outputs = torch.stack(latent_outputs, dim=0)
        stacked_diffusion_step_losses = torch.stack(diffusion_step_losses, dim=0)

        # Restore the original forward, reversing cpu_offloading will be done only after the gradient backward
        self.denoiser.forward = original_forward  # type: ignore[union-attr]

        return stacked_latent_outputs, stacked_diffusion_step_losses

    def training_step(self, batch: tuple[list[str], torch.Tensor, torch.Tensor, list[int]], batch_idx: int):
        """
        Compute a single-step loss from the denoiser on training data.

        Parameters
        ----------
        batch : tuple[list[str], torch.Tensor, torch.Tensor, int]
            The batch of (captions, latent_inputs, latent_targets, seed).
        batch_idx : int
            The index of the batch.

        Returns
        -------
        dict[str, torch.Tensor]
            The single-step training loss.
        """
        opt = self.optimizers()
        opt.zero_grad()  # type: ignore[union-attr]

        captions, latent_inputs, latent_targets, seeds = batch
        assert len(captions) == 1  # only a batch size of 1 corresponding to a full diffusion process is supported
        caption, latent_inputs, latent_targets, seed = captions[0], latent_inputs[0], latent_targets[0], seeds[0]

        self.pipeline.set_progress_bar_config(disable=True)
        if self.use_cpu_offloading:
            # avoids a bug when using gradient_accumulation_steps > 1 because the on_after_backward hasn't run yet
            utils.move_secondary_components(self.pipeline, self.device)

        diffusion_steps = latent_inputs.shape[0]
        trained_steps = (
            random.sample(range(diffusion_steps), min(self.batch_size, diffusion_steps))
            if self.batch_size > 0
            else list(range(diffusion_steps))
        )

        latent_outputs, diffusion_step_losses = self.forward(
            caption, latent_inputs, latent_targets, seed, active_steps=trained_steps
        )
        loss = diffusion_step_losses.mean()
        if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or self.trainer.is_last_batch:
            self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")  # type: ignore[arg-type]
            opt.step()  # type: ignore[union-attr]

        if self.trainer.is_last_batch:
            lr_schedulers = self.lr_schedulers()
            if lr_schedulers:
                if isinstance(lr_schedulers, list):
                    for scheduler in lr_schedulers:
                        scheduler.step()  # type: ignore[call-arg]
                else:
                    lr_schedulers.step()

        self.log("train_loss", loss)
        self.train_losses.append(loss)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """
        Compute a single-step loss from the denoiser on validation data.

        Parameters
        ----------
        batch : tuple[list[str], torch.Tensor]
            The batch of (captions, images).
        batch_idx : int
            The index of the batch.

        Returns
        -------
        dict[str, torch.Tensor]
            The single-step validation loss.
        """
        caption, latent_inputs, latent_targets, seed = batch
        assert len(caption) == 1  # only a batch size of 1 corresponding to a full diffusion process is supported
        caption, latent_inputs, latent_targets, seed = caption[0], latent_inputs[0], latent_targets[0], seed[0]

        self.pipeline.set_progress_bar_config(disable=True)

        latent_outputs, diffusion_step_losses = self.forward(caption, latent_inputs, latent_targets, seed)
        loss = diffusion_step_losses.mean()
        self.log("validation_loss", loss)
        self.val_losses.append(loss)

        # no need to do CPU offloading in evaluation since gradients are not computed
        return {"loss": loss}

    def on_train_epoch_end(self):
        """Log the train loss."""
        loss = torch.stack(self.train_losses).mean()
        pruna_logger.info(f"{self.recoverer} - epoch {self.current_epoch} - train loss: {loss:.3e}")
        self.train_losses.clear()

    def on_validation_epoch_end(self):
        """Log the validation loss."""
        if self.trainer.sanity_checking:
            return
        loss = torch.stack(self.val_losses).mean()
        epoch_descr = "before distillation" if self.trainer.global_step == 0 else f"epoch {self.current_epoch}"
        pruna_logger.info(f"{self.recoverer} - {epoch_descr} - validation loss: {loss:.3e}")
        self.val_losses.clear()

    def on_after_backward(self):
        """Move the secondary components to the device after backward is done."""
        if self.use_cpu_offloading:
            # ensure the cpu_offloading is reversed, so the validation doesn't have to protect against it
            utils.move_secondary_components(self.pipeline, self.device)

    def loss(self, model_pred, target):
        """
        Compute the denoising loss.

        Parameters
        ----------
        model_pred : torch.Tensor
            The predicted latents.
        target : torch.Tensor
            The target latents.

        Returns
        -------
        torch.Tensor
            The MSE loss between the predicted and target latents.
        """
        return torch.nn.functional.mse_loss(model_pred, target)

    def configure_optimizers(self):
        """
        Configure the optimizer.

        Returns
        -------
        torch.optim.Optimizer
            The optimizer.
        """
        kwargs = {"eps": 1e-7} if self.trainer.precision in [16, "16", "16-true"] else {}
        kwargs["lr"] = self.learning_rate
        kwargs["weight_decay"] = self.weight_decay

        optimizer_cls: type[torch.optim.Optimizer]
        if self.optimizer_name == "AdamW8bit":
            if self.device == "cpu" or isinstance(self.device, torch.device) and self.device.type == "cpu":
                pruna_logger.warning("AdamW8bit is not supported on CPU, continuing with AdamW from torch.")
                optimizer_cls = torch.optim.AdamW
            elif AdamW8bit is None:
                pruna_logger.warning(
                    "Recovery with AdamW8bit requires bitsandbytes to be installed, continuing with AdamW from torch."
                )
                optimizer_cls = torch.optim.AdamW
            else:
                optimizer_cls = AdamW8bit
        else:
            queried_optimizer_cls = getattr(torch.optim, self.optimizer_name)
            if issubclass(queried_optimizer_cls, torch.optim.Optimizer):
                optimizer_cls = queried_optimizer_cls
            else:
                raise ValueError(f"Invalid optimizer: {self.optimizer_name}")

        finetune_params = get_trainable_parameters(self.pipeline)
        used_kwargs, unused_kwargs = filter_kwargs(optimizer_cls.__init__, kwargs)
        if unused_kwargs:
            pruna_logger.warning(f"Unused optimizer arguments: {list(unused_kwargs.keys())}")
        optimizer = optimizer_cls(finetune_params, **used_kwargs)

        lr_scheduler: torch.optim.lr_scheduler.LRScheduler
        if self.warmup_steps > 0 and self.lr_decay < 1.0:
            raise ValueError("Warmup steps and lr_decay cannot both be set for now.")
        elif self.warmup_steps > 0:
            lr_scheduler = get_scheduler(
                name="constant_with_warmup",
                optimizer=optimizer,
                num_warmup_steps=self.warmup_steps,
            )
            return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]
        elif self.lr_decay < 1.0:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay)
            return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"}]
        else:
            return [optimizer], []

    def _get_denoiser_output_object(self, output_tensor: torch.Tensor) -> BaseOutput:
        """
        Wrap the output tensor in the BaseOutput class expected by the pipeline.

        Parameters
        ----------
        output_tensor : torch.Tensor
            The output tensor from the denoiser.

        Returns
        -------
        BaseOutput
            The wrapped output tensor.
        """
        if not hasattr(self, "_denoiser_output_class"):  # lazy initialization
            self._denoiser_output_class = utils.get_denoiser_output_class(self.denoiser)
        return self._denoiser_output_class(sample=output_tensor)

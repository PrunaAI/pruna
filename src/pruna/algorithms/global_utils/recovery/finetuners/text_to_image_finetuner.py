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

from pathlib import Path
from typing import Any, List, Literal, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.seed import isolate_rng

try:
    from bitsandbytes.optim import AdamW8bit  # type: ignore[import-untyped]
except ImportError:
    AdamW8bit = None

from ConfigSpace import (
    CategoricalHyperparameter,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from pruna.algorithms.global_utils.recovery.finetuners import PrunaFinetuner
from pruna.algorithms.global_utils.recovery.finetuners.diffusers import (
    pack_and_predict,
    scheduler_interface,
    utils,
)
from pruna.algorithms.global_utils.recovery.utils import (
    get_dtype,
    get_trainable_parameters,
    split_defaults,
)
from pruna.config.hyperparameters import Boolean
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.logging.logger import pruna_logger


class TextToImageFinetuner(PrunaFinetuner):
    """Finetuner for text-to-image models."""

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
            "training_batch_size": 0,  # 0: the default smash_config.train_dataloader's batch size is used
            "gradient_accumulation_steps": 1,
            "num_epochs": 1.0,
            "validate_every_n_epoch": 1.0,
            "learning_rate": 1e-4,
            "weight_decay": 1e-2,
            "report_to": "none",
            "optimizer": "AdamW8bit" if torch.cuda.is_available() else "AdamW",  # AdamW8bit from BnB assumes CUDA
        }
        defaults.update(override_defaults)
        string_defaults, numeric_defaults = split_defaults(defaults)

        return [
            UniformIntegerHyperparameter(
                "training_batch_size",
                lower=0,
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
            UniformIntegerHyperparameter(
                "num_epochs",
                lower=0,
                upper=4096,
                default_value=numeric_defaults["num_epochs"],
                meta=dict(desc="Number of epochs for finetuning."),
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
                meta=dict(desc="Learning rate for finetuning."),
            ),
            Constant("weight_decay", numeric_defaults["weight_decay"]),
            # report_to: for consistency with text-to-text-lora but wandb and tensorboard are not supported yet
            Constant("report_to", string_defaults["report_to"]),
            Boolean(
                "use_cpu_offloading",
                default=False,
                meta=dict(desc="Whether to use CPU offloading for finetuning."),
            ),  # necessary for Flux in float16 on L40S GPU (48gb VRAM)
            CategoricalHyperparameter(
                "optimizer",
                choices=["AdamW8bit", "AdamW", "Adam"],
                default_value=string_defaults["optimizer"],
                meta=dict(desc="Which optimizer to use for finetuning."),
            ),
        ]

    @classmethod
    def finetune(cls, pipeline: Any, smash_config: SmashConfigPrefixWrapper, seed: int, recoverer: str) -> Any:
        """
        Finetune the model's previously activated parameters on data.

        This function is adapted from the HuggingFace implementation of the finetuning process at
        https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py,
        https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_sana.py,
        and https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_flux.py

        Parameters
        ----------
        pipeline : Any
            The pipeline containing components to finetune.
        smash_config : SmashConfigPrefixWrapper
            The configuration for the finetuner.
        seed : int
            The seed to use for reproducibility.
        recoverer : str
            The name of the algorithm used for finetuning, used for logging.

        Returns
        -------
        Any
            The finetuned pipeline.
        """
        dtype = get_dtype(pipeline)
        device = smash_config.device if isinstance(smash_config.device, str) else smash_config.device.type

        # split seed into two rng: generator for the dataloader and a seed for the training part
        generator = torch.Generator().manual_seed(seed)
        fit_seed = int(torch.randint(0, 2**32 - 1, (1,), generator=generator).item())

        # Dataloaders
        # override batch size if user specified one for finetuning specifically
        batch_size: int
        if smash_config["training_batch_size"] > 0 and smash_config.is_batch_size_locked():
            pruna_logger.warning(
                "Batch size is locked by a previous smashing algorithm, "
                "ignoring user-specified batch size for finetuning."
            )
            batch_size = smash_config.batch_size
        elif smash_config["training_batch_size"] > 0:
            batch_size = smash_config["training_batch_size"]
        else:
            batch_size = smash_config.batch_size
        # train dataloader has a generator for reproducibility, val is not shuffled so it doesn't need one
        train_dataloader = smash_config.data.train_dataloader(
            batch_size=batch_size,
            output_format="normalized",
            generator=generator,
        )
        val_dataloader = smash_config.data.val_dataloader(
            batch_size=batch_size,
            output_format="normalized",
        )

        optimizer_name = smash_config["optimizer"]
        if optimizer_name == "AdamW8bit" and device != "cuda":
            pruna_logger.warning(
                "Optimizer AdamW8bit from bitsandbytes requires CUDA, continuing with AdamW from torch."
            )
            optimizer_name = "AdamW"

        # Check resolution mismatch
        utils.check_resolution_mismatch(pipeline, train_dataloader)

        # Finetune the model
        trainable_denoiser = DenoiserTL(
            pipeline,
            optimizer_name,
            smash_config["learning_rate"],
            smash_config["weight_decay"],
            recoverer,
            smash_config["use_cpu_offloading"],
        )
        # make directory for checkpoints
        model_path = Path(smash_config.cache_dir) / "recovery"
        model_path.mkdir(exist_ok=True, parents=True)

        early_stopping = EarlyStopping(monitor="validation_loss", patience=3, mode="min", check_finite=True)
        checkpoint_callback = ModelCheckpoint(
            monitor="validation_loss",
            save_top_k=1,
            mode="min",
            every_n_epochs=1,
            dirpath=model_path,
            filename="{recoverer}-{epoch:02d}-{validation_loss:.4f}",
        )
        callbacks = [early_stopping, checkpoint_callback]

        if smash_config["validate_every_n_epoch"] >= 1.0:
            # set TL trainer to perform validation every n epochs, for small datasets
            check_val_every_n_epoch = int(smash_config["validate_every_n_epoch"])
            val_check_interval = None
        else:
            # set TL trainer to perform validation multiple times per epoch, for large datasets
            check_val_every_n_epoch = 1
            val_check_interval = smash_config["validate_every_n_epoch"]

        precision: Literal["16-true", "bf16-true", "32"]
        if dtype == torch.float16:
            precision = "16-true"
        elif dtype == torch.bfloat16:
            precision = "bf16-true"
        else:
            precision = "32"

        trainer = pl.Trainer(
            accelerator=device,
            callbacks=callbacks,
            max_epochs=smash_config["num_epochs"],
            gradient_clip_val=1.0,
            precision=precision,
            accumulate_grad_batches=smash_config["gradient_accumulation_steps"],
            check_val_every_n_epoch=check_val_every_n_epoch,
            val_check_interval=val_check_interval,
            logger=False,
        )
        with isolate_rng():
            pl.seed_everything(fit_seed, workers=True, verbose=False)
            trainer.validate(trainable_denoiser, dataloaders=val_dataloader, verbose=False)
            trainer.fit(trainable_denoiser, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        # Loading the best checkpoint is slow and currently creates conflicts with some quantization algorithms,
        # e.g. diffusers_int8. Skipping calling DenoiserTL.load_from_checkpoint for now.
        return pipeline.to(device)


class DenoiserTL(pl.LightningModule):
    """
    Pipeline in LightningModule format for finetuning the denoiser.

    Parameters
    ----------
    pipeline : Any
        The pipeline to finetune.
    optimizer_name : str
        The name of the optimizer to use, options are "AdamW8bit", or optimizers from torch.optim.
    learning_rate : float
        The learning rate to use for finetuning.
    weight_decay : float
        The weight decay to use for finetuning.
    recoverer : str
        The name of the algorithm used for finetuning, used for logging.
    use_cpu_offloading : bool, optional
        Whether to use CPU offloading for finetuning.
    validation_seed : int, optional
        The seed to use for validation, used to reproducibly generate the random noise and timesteps for the
        validation set.
    """

    def __init__(
        self,
        pipeline: Any,
        optimizer_name: str,
        learning_rate: float,
        weight_decay: float,
        recoverer: str,
        use_cpu_offloading: bool = False,
        validation_seed: int = 42,
    ):
        super().__init__()
        self.pipeline = pipeline
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_cpu_offloading = use_cpu_offloading
        self.validation_generator = torch.Generator().manual_seed(validation_seed)
        self.validation_seeds: List[int] = []
        # register the denoiser explicitly to work with self.parameters() and other LightningModule methods
        denoiser, _ = utils.get_denoiser_attr(pipeline)
        if denoiser is None or not isinstance(denoiser, torch.nn.Module):
            raise ValueError("Could not find the denoiser in the pipeline.")
        self.denoiser = denoiser

        self.pack_and_predict = pack_and_predict.get_pack_and_predict_fn(pipeline)
        self.uses_prompt_2 = utils.uses_prompt_2(pipeline)
        self.encode_arguments = utils.get_encode_arguments(pipeline)
        self.training_scheduler = scheduler_interface.get_training_scheduler(pipeline.scheduler)

        # basic logging
        self.recoverer = recoverer
        self.val_losses: List[torch.Tensor] = []
        self.train_losses: List[torch.Tensor] = []

    def forward(
        self, noisy_latents: torch.Tensor, encoder_hidden_states: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the denoiser.

        Parameters
        ----------
        noisy_latents : torch.Tensor
            The noisy latents to denoise.
        encoder_hidden_states : torch.Tensor
            The encoder hidden states.
        timesteps : torch.Tensor
            The timesteps used for positional encoding.

        Returns
        -------
        torch.Tensor
            The denoised latents.
        """
        return self.pack_and_predict(self.pipeline, noisy_latents, encoder_hidden_states, timesteps)

    def prepare_latents_targets(
        self, images: torch.Tensor, timesteps: torch.Tensor, generator: torch.Generator | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare the latents and targets for the denoiser.

        Parameters
        ----------
        images : torch.Tensor
            The images to prepare.
        timesteps : torch.Tensor
            The timesteps to use for the denoiser.
        generator : torch.Generator | None, optional
            The generator to use for drawing random noise.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The noisy latents and the targets.
        """
        dtype = self.pipeline.vae.dtype

        # Convert images to latent space
        latents = self.pipeline.vae.encode(images.to(dtype))
        # applying vae is fine with Sana in 32 bit and 16 bit (model in 16bit w/o, both with no info for the trainer and
        # with 16-true), but produces NaNs in 16-mixed
        latents = latents.latent_dist.sample() if hasattr(latents, "latent_dist") else latents.latent
        # Apply shift and scaling factors
        if hasattr(self.pipeline.vae.config, "shift_factor") and self.pipeline.vae.config.shift_factor is not None:
            latents = latents - self.pipeline.vae.config.shift_factor
        latents = latents * self.pipeline.vae.config.scaling_factor

        # Add noise to the latents according to the noise magnitude at each timestep
        # torch.randn_like does not support generators
        noise = torch.randn(latents.shape, dtype=latents.dtype, device=latents.device, generator=generator)
        noisy_latents = scheduler_interface.add_noise(self.training_scheduler, latents, noise, timesteps)

        # define the corresponding target for training
        target = scheduler_interface.get_target(self.training_scheduler, latents, noise, timesteps)

        return noisy_latents, target

    def encode_prompt(self, captions: List[str]) -> torch.Tensor:
        """
        Encode the prompts.

        Parameters
        ----------
        captions : list[str]
            The captions to encode.

        Returns
        -------
        torch.Tensor
            The encoded prompts.
        """
        prompt_args = [captions]
        if self.uses_prompt_2:
            prompt_args = prompt_args * 2
        encoder_hidden_states = self.pipeline.encode_prompt(
            *prompt_args,
            device=self.device,
            num_images_per_prompt=1,
            **self.encode_arguments,
        )
        return encoder_hidden_states

    def training_step(self, batch: Tuple[List[str], torch.Tensor], batch_idx: int):
        """
        Training step of the denoiser.

        Parameters
        ----------
        batch : tuple[list[str], torch.Tensor]
            The batch of (captions, images).
        batch_idx : int
            The index of the batch.

        Returns
        -------
        torch.Tensor
            The MSE loss between the predicted and target latents.
        """
        captions, images = batch
        batch_size = int(images.shape[0])

        if self.use_cpu_offloading:
            # make sure secondary components are on the same device as the model
            utils.move_secondary_components(self.pipeline, self.device)

        # uniform timesteps for simplicity, replace with compute_density_for_timestep_sampling in future
        timesteps = scheduler_interface.sample_timesteps(self.training_scheduler, batch_size, self.device)
        noisy_latents, target = self.prepare_latents_targets(images, timesteps)
        encoder_hidden_states = self.encode_prompt(captions)

        if self.use_cpu_offloading:
            # clear memory: required in testing to fit flux bfloat16 + lora finetuning with bs=1 onto 48gb VRAM
            utils.move_secondary_components(self.pipeline, "cpu")

        model_pred = self.forward(noisy_latents, encoder_hidden_states, timesteps)
        loss = self.loss(model_pred, target)

        self.log("train_loss", loss)
        self.train_losses.append(loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """
        Validation step of the denoiser.

        Parameters
        ----------
        batch : tuple[list[str], torch.Tensor]
            The batch of (captions, images).
        batch_idx : int
            The index of the batch.

        Returns
        -------
        torch.Tensor
            The MSE loss between the predicted and target latents.
        """
        captions, images = batch
        batch_size = int(images.shape[0])

        if self.use_cpu_offloading:
            # make sure secondary components are on the same device as the model
            utils.move_secondary_components(self.pipeline, self.device)

        # get seeds for reproducibility
        # - make sure the timesteps and random noise are the same across validations
        # - make sure the seeds are different across batches
        if len(self.validation_seeds) <= batch_idx:
            seed = int(torch.randint(0, 2**32 - 1, (1,), generator=self.validation_generator).item())
            self.validation_seeds.append(seed)
        else:
            seed = self.validation_seeds[batch_idx]

        with isolate_rng():
            pl.seed_everything(seed, verbose=False)

            timesteps = scheduler_interface.sample_timesteps(self.training_scheduler, batch_size, self.device)
            noisy_latents, target = self.prepare_latents_targets(images, timesteps)
            encoder_hidden_states = self.encode_prompt(captions)

            # no need to do CPU offloading in evaluation since gradients are not computed
            model_pred = self.forward(noisy_latents, encoder_hidden_states, timesteps)
        loss = self.loss(model_pred, target)

        self.log("validation_loss", loss)
        self.val_losses.append(loss)
        return {"loss": loss}

    def loss(self, model_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
        pruna_logger.info(f"{self.recoverer} - epoch {self.current_epoch} - validation loss: {loss:.3e}")
        self.val_losses.clear()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer.

        Returns
        -------
        torch.optim.Optimizer
            The optimizer.
        """
        lr = self.learning_rate
        wd = self.weight_decay
        kwargs = {"eps": 1e-7} if self.trainer.precision in [16, "16", "16-true"] else {}

        if self.optimizer_name == "AdamW8bit":
            optimizer_cls = AdamW8bit
            if optimizer_cls is None:
                pruna_logger.warning(
                    "Recovery with AdamW8bit requires bitsandbytes to be installed, continuing with AdamW from torch."
                )
                optimizer_cls = torch.optim.AdamW
        else:
            optimizer_cls = getattr(torch.optim, self.optimizer_name)
        finetune_params = get_trainable_parameters(self.pipeline)

        return optimizer_cls(finetune_params, lr=lr, weight_decay=wd, **kwargs)

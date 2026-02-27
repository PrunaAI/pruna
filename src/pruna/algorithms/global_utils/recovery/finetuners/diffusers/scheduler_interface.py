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
from diffusers import DDPMScheduler, FlowMatchEulerDiscreteScheduler

from pruna.logging.logger import pruna_logger


def get_training_scheduler(scheduler: Any) -> Any:
    """
    Initialize a scheduler specifically for training to isolate finetuning from inference.

    Parameters
    ----------
    scheduler : Any
        The scheduler native to the pipeline.

    Returns
    -------
    Any
        A scheduler using during training, initialized from the pipeline's configuration.
        If no training scheduler could be inferred, returns the native scheduler.
    """
    prediction_type = get_prediction_type(scheduler)
    if prediction_type == "flow_prediction":
        # Sana and Flux schedulers
        return FlowMatchEulerDiscreteScheduler.from_config(scheduler.config)
    elif prediction_type in ["epsilon", "v_prediction"]:
        # DDPM and DDPMSolverMultistepScheduler
        return DDPMScheduler.from_config(scheduler.config)
    else:
        pruna_logger.warning(
            f"Could not infer a scheduler for finetuning {scheduler.__class__.__name__}, "
            "defaulting to the native scheduler, which may cause numerical issues."
        )
        return scheduler


def sample_timesteps(training_scheduler: Any, batch_size: int, device: str | torch.device) -> torch.Tensor:
    """
    Sample timesteps for the scheduler.

    Parameters
    ----------
    training_scheduler : Any
        The scheduler used during training.
    batch_size : int
        The batch size.
    device : str | torch.device
        The device to sample the timesteps on.

    Returns
    -------
    torch.Tensor
        The sampled timesteps, with shape (batch_size,).
    """
    # uniform timesteps for simplicity, replaced with compute_density_for_timestep_sampling in future
    indices = torch.randint(0, training_scheduler.config.num_train_timesteps, (batch_size,)).long()
    if hasattr(training_scheduler, "timesteps") and training_scheduler.timesteps is not None:
        timesteps = training_scheduler.timesteps[indices]
    else:
        # same distribution as indices but count backwards for consistency with what simple timesteps[indices] does
        timesteps = training_scheduler.config.num_train_timesteps - 1 - indices
    return timesteps.to(device=device)


def add_noise(
    training_scheduler: Any, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
) -> torch.Tensor:
    """
    Add noise to the latents using the training scheduler.

    Parameters
    ----------
    training_scheduler : Any
        The scheduler used during training.
    latents : torch.Tensor
        The latents to add noise to.
    noise : torch.Tensor
        The noise to add, with the same shape as `latents`.
    timesteps : torch.Tensor
        The timesteps used to compute how much noise to add.

    Returns
    -------
    torch.Tensor
        The noisy latents.
    """
    if hasattr(training_scheduler, "add_noise"):
        return training_scheduler.add_noise(latents, noise, timesteps)
    elif hasattr(training_scheduler, "scale_noise"):
        return training_scheduler.scale_noise(latents, timesteps, noise)
    else:
        raise ValueError("Unknown method for adding noise to latents")


def get_target(
    training_scheduler: Any, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
) -> torch.Tensor:
    """
    Define the target used to finetune the denoiser.

    Parameters
    ----------
    training_scheduler : Any
        The scheduler used during training.
    latents : torch.Tensor
        The latents to add noise to.
    noise : torch.Tensor
        The noise to add, with the same shape as `latents`.
    timesteps : torch.Tensor
        The used for the training step and corresponding to the noise levels.

    Returns
    -------
    torch.Tensor
        The target used to finetune the denoising process.
    """
    prediction_type = get_prediction_type(training_scheduler)
    if prediction_type == "epsilon":
        return noise
    elif prediction_type == "v_prediction":
        return training_scheduler.get_velocity(latents, noise, timesteps)
    elif prediction_type == "flow_prediction":
        # Sana and Flux schedulers
        return noise - latents
    else:
        raise ValueError(f"Unknown prediction type or scheduler {prediction_type}")


def get_prediction_type(scheduler: Any) -> str:
    """
    Get the prediction type from the scheduler.

    Parameters
    ----------
    scheduler : Any
        The scheduler to get the prediction type from.

    Returns
    -------
    str
        The prediction type, in ['epsilon', 'v_prediction', 'flow_prediction'].
    """
    prediction_type = getattr(scheduler.config, "prediction_type", scheduler.config._class_name)
    if prediction_type == "flow_prediction" or "flowmatch" in prediction_type.lower():
        return "flow_prediction"  # e.g. Sana and Flux schedulers
    else:
        return prediction_type

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

from typing import Any, Callable

import torch

from pruna.engine.model_checks import (
    is_flux_pipeline,
    is_sana_pipeline,
    is_sd_pipeline,
    is_sdxl_pipeline,
)


def get_pack_and_predict_fn(pipeline: Any) -> Callable:
    """
    Get a function to call the denoiser with consistent arguments.

    Parameters
    ----------
    pipeline : Any
        The pipeline to get the predict noise function from.

    Returns
    -------
    Callable
        The predict function, taking as arguments the denoiser, the noisy latents, the encoder hidden states,
        and the timesteps.
    """
    if is_sd_pipeline(pipeline):
        return _pack_and_predict_stable_diffusion
    elif is_sdxl_pipeline(pipeline):
        return _pack_and_predict_stable_diffusion_xl
    elif is_sana_pipeline(pipeline):
        return _pack_and_predict_sana
    elif is_flux_pipeline(pipeline):
        return _pack_and_predict_flux
    else:
        raise ValueError(f"Unknown pipeline: {pipeline.__class__.__name__}")


def _pack_and_predict_stable_diffusion(
    pipeline: Any,
    noisy_latents: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Format inputs for Stable Diffusion and apply unet."""
    prompt_embeds, _ = encoder_hidden_states
    model_pred = pipeline.unet(noisy_latents, encoder_hidden_states=prompt_embeds, timestep=timesteps)

    return model_pred[0]


def _pack_and_predict_stable_diffusion_xl(
    pipeline: Any, noisy_latents: torch.Tensor, encoder_hidden_states: torch.Tensor, timesteps: torch.Tensor
) -> torch.Tensor:
    """Format inputs for Stable Diffusion XL and apply unet."""
    prompt_embeds, _, pooled_prompt_embeds, _ = encoder_hidden_states

    # Get resolution from the latents:
    # Multiply by vae_scale_factor since latents are downsampled
    height = noisy_latents.shape[2] * pipeline.vae_scale_factor
    width = noisy_latents.shape[3] * pipeline.vae_scale_factor

    # Get text encoder projection dim from the pipeline
    text_encoder_projection_dim = pipeline.text_encoder_2.config.projection_dim

    add_time_ids = pipeline._get_add_time_ids(
        original_size=(height, width),
        crops_coords_top_left=(0, 0),
        target_size=(height, width),
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )
    add_time_ids = add_time_ids.to(device=noisy_latents.device)
    add_time_ids = add_time_ids.repeat(noisy_latents.shape[0], 1)

    added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}

    model_pred = pipeline.unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=prompt_embeds,
        added_cond_kwargs=added_cond_kwargs,
        return_dict=False,
    )

    return model_pred[0]


def _pack_and_predict_sana(
    pipeline: Any, noisy_latents: torch.Tensor, encoder_hidden_states: torch.Tensor, timesteps: torch.Tensor
) -> torch.Tensor:
    """Format inputs for Sana and apply transformer."""
    prompt_embeds, prompt_attention_mask, _, _ = encoder_hidden_states
    return pipeline.transformer(
        noisy_latents,
        encoder_hidden_states=prompt_embeds,
        encoder_attention_mask=prompt_attention_mask,
        timestep=timesteps,
    )[0]


def _pack_and_predict_flux(
    pipeline: Any, noisy_latents: torch.Tensor, encoder_hidden_states: torch.Tensor, timesteps: torch.Tensor
) -> torch.Tensor:
    """Format inputs for Flux and apply transformer."""
    prompt_embeds, pooled_prompt_embeds, text_ids = encoder_hidden_states
    latent_image_ids = pipeline._prepare_latent_image_ids(
        noisy_latents.shape[0],
        noisy_latents.shape[2] // 2,
        noisy_latents.shape[3] // 2,
        noisy_latents.device,
        noisy_latents.dtype,
    )
    packed_noisy_model_input = pipeline._pack_latents(
        noisy_latents,
        batch_size=noisy_latents.shape[0],
        num_channels_latents=noisy_latents.shape[1],
        height=noisy_latents.shape[2],
        width=noisy_latents.shape[3],
    )
    # guidance with 3.5 following default value in
    # https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_flux.py
    if pipeline.transformer.config.guidance_embeds:
        guidance = torch.tensor([3.5], device=noisy_latents.device)
        guidance = guidance.expand(noisy_latents.shape[0])
    else:
        guidance = None

    # predict
    model_pred = pipeline.transformer(
        hidden_states=packed_noisy_model_input,
        timestep=timesteps / 1000,
        guidance=guidance,
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=latent_image_ids,
        return_dict=False,
    )[0]
    # unpack for loss computation
    vae_scale_factor = 2 ** (len(pipeline.vae.config.block_out_channels) - 1)
    model_pred = pipeline._unpack_latents(
        model_pred,
        height=noisy_latents.shape[2] * vae_scale_factor,
        width=noisy_latents.shape[3] * vae_scale_factor,
        vae_scale_factor=vae_scale_factor,
    )

    return model_pred

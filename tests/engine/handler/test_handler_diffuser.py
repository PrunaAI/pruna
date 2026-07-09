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

"""Tests for the DiffuserHandler inference handler."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from datasets import Dataset
from PIL import Image

from pruna.config.smash_config import SmashConfig
from pruna.data.collate import (
    image_generation_collate,
    prompt_collate,
    prompt_with_auxiliaries_collate,
)
from pruna.data.pruna_datamodule import PrunaDataModule
from pruna.engine.handler.handler_diffuser import DiffuserHandler
from pruna.engine.pruna_model import PrunaModel
from pruna.engine.utils import move_to_device


def _img(size: int = 32) -> Image.Image:
    """Return a random RGB PIL image."""
    return Image.fromarray(np.random.randint(0, 255, (size, size, 3), dtype=np.uint8))


def _fake_output(n: int, size: int = 16) -> SimpleNamespace:
    """Return a fake diffusers output object exposing an ``images`` list of PIL images."""
    return SimpleNamespace(images=[_img(size) for _ in range(n)])


class PromptPipe:
    """Dummy pipeline whose call signature exposes a ``prompt`` parameter."""

    def __call__(self, prompt, negative_prompt=None, num_inference_steps=1, generator=None, **kwargs):  # noqa: D102
        return _fake_output(len(prompt) if isinstance(prompt, list) else 1)


class ArgsPipe:
    """Dummy pipeline whose call signature exposes only variadic ``args``."""

    def __call__(self, *args, generator=None):  # noqa: D102
        return _fake_output(1)


class UncondPipe:
    """Dummy pipeline whose call signature exposes neither ``prompt`` nor ``args``."""

    def __call__(self, num_inference_steps=1, generator=None):  # noqa: D102
        return _fake_output(1)


def _handler_for(pipe: Any) -> DiffuserHandler:
    """Build a DiffuserHandler from a pipeline's call signature, mirroring production wiring."""
    return DiffuserHandler(call_signature=inspect.signature(pipe.__call__))


# -- prepare_inputs: prompt branch fed by real pruna collates --
# Tests are available only for the collates appropriate for a diffuser pipeline.


@pytest.mark.cpu
def test_prepare_inputs_prompt_collate():
    """prompt_collate output is wrapped as a prompt dict for the model call."""
    handler = _handler_for(PromptPipe())
    data = [{"text": "a cat"}, {"text": "a dog"}]
    batch = prompt_collate(data)
    prepared = handler.prepare_inputs(batch)
    assert prepared == {"prompt": ["a cat", "a dog"]}


@pytest.mark.cpu
def test_prepare_inputs_prompt_with_auxiliaries_collate():
    """prompt_with_auxiliaries_collate: prompts are wrapped and auxiliaries are dropped."""
    handler = _handler_for(PromptPipe())
    data = [
        {"text": "a cat", "category": "animal"},
        {"text": "a dog", "category": "animal"},
    ]
    batch = prompt_with_auxiliaries_collate(data)
    prepared = handler.prepare_inputs(batch)
    assert prepared == {"prompt": ["a cat", "a dog"]}


@pytest.mark.cpu
def test_prepare_inputs_image_generation_collate():
    """image_generation_collate texts become the prompt for the model call."""
    handler = _handler_for(PromptPipe())
    data = [{"image": _img(), "text": "a cat"}, {"image": _img(), "text": "a dog"}]
    batch = image_generation_collate(data, img_size=32)
    prepared = handler.prepare_inputs(batch)
    assert prepared == {"prompt": ["a cat", "a dog"]}


# -- prepare_inputs: custom negative-prompt path --

def custom_negative_prompt_collate(data: Any) -> tuple[dict[str, list[str]], None]:
    """Collate a dataset into a dict of prompt/negative_prompt lists as the model input."""
    return {"prompt": [d["prompt"] for d in data], "negative_prompt": [d["negative_prompt"] for d in data]}, None


@pytest.mark.cpu
def test_prepare_inputs_negative_prompt_via_datamodule():
    """A custom collate emitting a dict input flows through PrunaDataModule and is preserved."""
    handler = _handler_for(PromptPipe())
    ds = Dataset.from_dict(
        {"prompt": ["a cat", "a dog"], "negative_prompt": ["blurry", "low quality"]}
    )
    dm = PrunaDataModule(ds, ds, ds, custom_negative_prompt_collate, dataloader_args={})
    batch = next(iter(dm.train_dataloader(batch_size=2, shuffle=False)))
    prepared = handler.prepare_inputs(batch)
    print(prepared)
    assert prepared == {"prompt": ["a cat", "a dog"], "negative_prompt": ["blurry", "low quality"]}


@pytest.mark.cpu
def test_prepare_inputs_dict_batch_preserved():
    """A dict passed directly as the first batch element is forwarded unchanged."""
    handler = _handler_for(PromptPipe())
    payload = {"prompt": ["a cat"], "negative_prompt": ["blurry"]}
    prepared = handler.prepare_inputs((payload, {}))
    assert prepared == payload


# ── prepare_inputs: other signature branches ─────────────────────


@pytest.mark.cpu
def test_prepare_inputs_args_branch():
    """When the signature only exposes *args, the first element is returned as-is."""
    handler = _handler_for(ArgsPipe())
    prepared = handler.prepare_inputs((["a cat", "a dog"], {}))
    assert prepared == ["a cat", "a dog"]


@pytest.mark.cpu
def test_prepare_inputs_unconditional_branch():
    """Unconditional pipelines (no prompt/args) receive no prepared inputs."""
    handler = _handler_for(UncondPipe())
    assert handler.prepare_inputs((["a cat"], {})) is None


# ── process_output and __init__ ──────────────────────────────────


@pytest.mark.cpu
def test_process_output_stacks_images():
    """process_output stacks PIL images into a uint8 (N, 3, H, W) tensor."""
    handler = _handler_for(PromptPipe())
    output = handler.process_output(_fake_output(2, size=16))
    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 3, 16, 16)
    assert output.dtype == torch.uint8


@pytest.mark.cpu
def test_init_sets_generator_and_merges_model_args():
    """__init__ provides a fixed-seed generator and merges extra model args without dropping it."""
    handler = DiffuserHandler(
        call_signature=inspect.signature(PromptPipe().__call__),
        model_args={"num_inference_steps": 2},
    )
    assert isinstance(handler.model_args["generator"], torch.Generator)
    assert handler.model_args["num_inference_steps"] == 2


# ── real end-to-end generation on a tiny CPU model ───────────────


@pytest.mark.cpu
@pytest.mark.parametrize("model_fixture", ["sd_tiny_random", "flux_tiny_random"], indirect=True)
def test_generation_prompt_batch(model_fixture: tuple[Any, SmashConfig]):
    """Real generation from a prompt-list batch returns a stacked image tensor."""
    model, smash_config = model_fixture
    smash_config.device = "cpu"
    pruna_model = PrunaModel(model, smash_config=smash_config)
    move_to_device(pruna_model, "cpu")
    pruna_model.inference_handler.model_args["num_inference_steps"] = 2

    batch = (["a red apple", "a blue car"], None)
    outputs = pruna_model.run_inference(batch)

    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape[0] == 2
    assert outputs.shape[1] == 3


@pytest.mark.cpu
@pytest.mark.parametrize("model_fixture", ["sd_tiny_random", "flux_tiny_random"], indirect=True)
def test_generation_negative_prompt_batch(model_fixture: tuple[Any, SmashConfig]):
    """Real generation with a negative_prompt dict batch reaches the pipeline and returns an image tensor."""
    model, smash_config = model_fixture
    smash_config.device = "cpu"
    pruna_model = PrunaModel(model, smash_config=smash_config)
    move_to_device(pruna_model, "cpu")
    pruna_model.inference_handler.model_args["num_inference_steps"] = 2

    batch = ({"prompt": ["a red apple"], "negative_prompt": ["blurry, low quality"]}, None)
    outputs = pruna_model.run_inference(batch)

    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape[0] == 1
    assert outputs.shape[1] == 3


@pytest.mark.cpu
@pytest.mark.parametrize("model_fixture", ["flux2_tiny_random"], indirect=True)
def test_generation_flux2_image_prompt_batch(model_fixture: tuple[Any, SmashConfig]):
    """Flux2 accepts an image+prompt dict batch end-to-end via the dict escape hatch."""
    model, smash_config = model_fixture
    smash_config.device = "cpu"
    pruna_model = PrunaModel(model, smash_config=smash_config)
    move_to_device(pruna_model, "cpu")
    pruna_model.inference_handler.model_args["num_inference_steps"] = 2
    pruna_model.inference_handler.model_args["text_encoder_out_layers"] = (1,)

    batch = ({"prompt": ["a red apple"], "image": [_img(64)]}, None)
    outputs = pruna_model.run_inference(batch)

    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape[0] == 1
    assert outputs.shape[1] == 3

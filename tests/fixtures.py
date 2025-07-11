import weakref
from functools import partial
from typing import Any, Callable

import pytest
import torch
from huggingface_hub import snapshot_download
from torchvision.models import get_model as torchvision_get_model
from transformers import AutoModelForCausalLM, AutoTokenizer, JanusForConditionalGeneration, pipeline

from pruna import SmashConfig
from pruna.data.pruna_datamodule import PrunaDataModule
from pruna.engine.load import load_diffusers_model
from pruna.engine.utils import safe_memory_cleanup

HIGH_RESOURCE_FIXTURES = ["sana"]
HIGH_RESOURCE_FIXTURES_CPU = HIGH_RESOURCE_FIXTURES + [
    "llama_3_1_8b",
    "llama_3_2_1b",
    "stable_diffusion_3_medium_diffusers",
]


@pytest.fixture(scope="function")
def model_fixture(request: pytest.FixtureRequest) -> Any:
    """Model fixture for testing."""
    no_weakref = request.param.startswith("noref_")
    if no_weakref:
        request.param = request.param.removeprefix("noref_")

    model, smash_config = MODEL_FACTORY[request.param]()

    if no_weakref:
        yield model, smash_config
    else:
        yield weakref.proxy(model), weakref.proxy(smash_config)

    del model
    del smash_config

    safe_memory_cleanup()


@pytest.fixture(scope="function")
def dataloader_fixture(request: pytest.FixtureRequest) -> Any:
    """Model fixture for testing."""
    if request.param in ["LAION256", "ImageNet"]:
        dm = PrunaDataModule.from_string(request.param)
    elif request.param in ["WikiText"]:
        bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        dm = PrunaDataModule.from_string(request.param, collate_fn_args=dict(tokenizer=bert_tokenizer, max_seq_len=512))
    else:
        raise ValueError(f"Invalid dataset: {request.param}")

    return dm.val_dataloader()


def whisper_tiny_random_model() -> tuple[Any, SmashConfig]:
    """Whisper tiny random model for speech recognition."""
    source_model_id = "openai/whisper-tiny"
    model_id = "yujiepan/whisper-v3-tiny-random"
    model = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        chunk_length_s=30,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    smash_config = SmashConfig()
    smash_config.add_tokenizer(source_model_id)
    smash_config.add_processor(source_model_id)
    return model, smash_config


def dummy_model() -> tuple[Any, SmashConfig]:
    """Dummy function for testing."""
    dummy_model = lambda x: x  # noqa: E731
    smash_config = SmashConfig()
    return dummy_model, smash_config


def get_diffusers_model(model_id: str, **kwargs: dict[str, Any]) -> tuple[Any, SmashConfig]:
    """Get a diffusers model for image generation."""
    # snapshot download of the model
    model_path = snapshot_download(model_id)
    model = load_diffusers_model(model_path, smash_config=SmashConfig(device="cpu"), **kwargs)
    smash_config = SmashConfig()
    smash_config.add_data("LAION256")
    return model, smash_config


def get_automodel_transformers(model_id: str, **kwargs: dict[str, Any]) -> tuple[Any, SmashConfig]:
    """Get an AutoModelForCausalLM model for text generation."""
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    smash_config = SmashConfig()
    try:
        smash_config.add_tokenizer(model_id)
    except Exception:
        smash_config.add_tokenizer("bert-base-uncased")

    if hasattr(smash_config.tokenizer, "pad_token"):
        smash_config.tokenizer.pad_token = smash_config.tokenizer.eos_token
        dataset = PrunaDataModule.from_string(
            "WikiText", collate_fn_args=dict(tokenizer=smash_config.tokenizer, max_seq_len=64)
        )
        dataset.limit_datasets(16)
        smash_config.add_data(dataset)

    return model, smash_config


def get_torchvision_model(name: str) -> tuple[Any, SmashConfig]:
    """Get a torchvision model for image classification."""
    model = torchvision_get_model(name=name)
    smash_config = SmashConfig()
    smash_config.add_data("ImageNet")
    return model, smash_config


def get_janus_model(model_id: str) -> tuple[Any, SmashConfig]:
    """Get a Janus model for image generation."""
    model = JanusForConditionalGeneration.from_pretrained(model_id)
    smash_config = SmashConfig()
    return model, smash_config


MODEL_FACTORY: dict[str, Callable] = {
    # whisper models
    "whisper_tiny_random": whisper_tiny_random_model,
    # vision models
    "shufflenet": partial(get_torchvision_model, "shufflenet_v2_x0_5"),
    "mobilenet_v2": partial(get_torchvision_model, "mobilenet_v2"),
    "resnet_18": partial(get_torchvision_model, "resnet18"),
    "vit_b_16": partial(get_torchvision_model, "vit_b_16"),
    # image generation models
    "stable_diffusion_v1_4": partial(get_diffusers_model, "CompVis/stable-diffusion-v1-4"),
    "stable_diffusion_3_medium_diffusers": partial(
        get_diffusers_model, "stabilityai/stable-diffusion-3-medium-diffusers"
    ),
    "ddpm-cifar10": partial(get_diffusers_model, "google/ddpm-cifar10-32"),
    "sd_tiny_random": partial(get_diffusers_model, "dg845/tiny-random-stable-diffusion"),
    "sana": partial(
        get_diffusers_model,
        "Efficient-Large-Model/Sana_600M_512px_diffusers",
        variant="fp16",
        torch_dtype=torch.float16,
    ),
    "sana_tiny_random": partial(get_diffusers_model, "katuni4ka/tiny-random-sana"),
    "flux_tiny_random": partial(get_diffusers_model, "katuni4ka/tiny-random-flux"),
    # text generation models
    "opt_125m": partial(get_automodel_transformers, "facebook/opt-125m"),
    "opt_tiny_random": partial(get_automodel_transformers, "yujiepan/opt-tiny-random"),
    "smollm_135m": partial(get_automodel_transformers, "HuggingFaceTB/SmolLM2-135M"),
    "llama_3_2_1b": partial(get_automodel_transformers, "NousResearch/Llama-3.2-1B"),
    "llama_3_1_8b": partial(get_automodel_transformers, "NousResearch/Hermes-3-Llama-3.1-8B"),
    "llama_3_tiny_random": partial(get_automodel_transformers, "llamafactory/tiny-random-Llama-3"),
    "dummy_lambda": dummy_model,

    # image generation AR models
    "tiny_janus_pro": partial(get_janus_model, "loulou2/tiny_janus"),
}

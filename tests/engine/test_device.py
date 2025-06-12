import pytest
import torch
from pruna.config.smash_config import SmashConfig
from pruna.engine.utils import move_to_device, get_device
from diffusers import StableDiffusionPipeline, FluxTransformer2DModel, FluxPipeline
from typing import Any
from ..common import construct_device_map_manually
from transformers import Pipeline

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

@pytest.mark.cuda
def test_device_default() -> None:
    """Test that the default device is 'cuda' when CUDA is available."""
    smash_config = SmashConfig()
    assert smash_config.device == "mps" or smash_config.device == "cuda"


@pytest.mark.cpu
@pytest.mark.parametrize("device", ["cpu", torch.device("cpu")])
def test_device_cpu(device: str | torch.device) -> None:
    """Test that setting device to 'cpu' works."""
    smash_config = SmashConfig(device=device)
    assert smash_config.device == "cpu"


@pytest.mark.cpu
def test_device_none() -> None:
    """Test that setting device to None defaults to best available device."""
    smash_config = SmashConfig(device=None)
    assert smash_config.device == DEVICE


@pytest.mark.cuda
@pytest.mark.parametrize(
    "device,expected",
    [
        ("mps", "cuda") if torch.cuda.is_available() else ("cuda", "mps"),
        (torch.device("mps"), "cuda") if torch.cuda.is_available() else (torch.device("cuda"), "mps"),
    ],
)
def test_device_available(device: str | torch.device, expected: str) -> None:
    """Test that setting device to an unavailable device falls back to CPU."""
    smash_config = SmashConfig(device=device)
    assert smash_config.device == expected

@pytest.mark.cuda
@pytest.mark.parametrize(
    "device,expected",
    [
        ("mps:0", "mps:0") if torch.backends.mps.is_available() else ("cuda:0", "cuda:0"),
        ("cuda:0", "cuda:0") if torch.cuda.is_available() else ("mps:0", "mps:0"),
    ],
)
def test_device_available_with_index(device: str | torch.device, expected: str) -> None:
    """Test that setting device to an unavailable device falls back to CPU."""
    smash_config = SmashConfig(device=device)
    assert smash_config.device == expected

@pytest.mark.cuda
@pytest.mark.parametrize(
    "input_device,target_device",
    [
        ("cpu", "cuda"),
        ("cuda", "cpu"),
    ],
)
@pytest.mark.parametrize(
    "model_fixture", ["flux_tiny_random", "whisper_tiny_random", "opt_tiny_random"], indirect=True
)
def test_device_casting(input_device: str | torch.device, target_device: str | torch.device, model_fixture: Any) -> None:
    """Test that the device can be cast to the target device."""
    model, _ = model_fixture
    move_to_device(model, input_device)
    assert get_device(model) == input_device
    move_to_device(model, target_device)
    assert get_device(model) == target_device


@pytest.mark.distributed
@pytest.mark.parametrize("target_device", ["cuda", "cpu"])
@pytest.mark.parametrize("model_fixture", ["sd_tiny_random"], indirect=True)
def test_accelerate_diffusers_casting(target_device: str | torch.device, model_fixture: Any) -> None:
    """Test that a diffusers pipeline can be cast to the target device."""
    model, _ = model_fixture
    device_map = construct_device_map_manually(model)
    move_to_device(model, "accelerate", device_map=device_map)

    move_and_verify(model, target_device, device_map)

    # verify functionality of forward pass
    model("an elf on a shelf", num_inference_steps=2, width=16, height=16)

    # cast back to distributed state
    move_and_verify(model, "accelerate", device_map)
    model("an elf on a shelf", num_inference_steps=2, width=16, height=16)


@pytest.mark.distributed
@pytest.mark.parametrize("target_device", ["cuda", "cpu"])
@pytest.mark.parametrize("model_fixture", ["opt_tiny_random"], indirect=True)
def test_accelerate_autocausallm_casting(target_device: str | torch.device, model_fixture: Any) -> None:
    """Test that a transformer AutoModel can be cast to the target device."""
    model, config = model_fixture
    device_map = construct_device_map_manually(model)
    move_to_device(model, "accelerate", device_map=device_map)

    move_and_verify(model, target_device, device_map)
    dummy = config.tokenizer([""] * 10, max_length=100, padding="max_length", return_tensors="pt")
    dummy = dummy.to(target_device)
    model(**dummy)

    move_and_verify(model, "accelerate", device_map)
    model(**dummy)


@pytest.mark.distributed
@pytest.mark.parametrize("target_device", ["cuda", "cpu"])
def test_accelerate_diffusers_model_casting(target_device: str | torch.device) -> None:
    """Test that a diffusers model can be cast to the target device."""
    model = FluxTransformer2DModel.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="transformer", device_map="balanced")
    full_pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", device_map="balanced", transformer=None)

    device_map_model = model.hf_device_map.copy()
    device_map_full_pipe = full_pipe.hf_device_map.copy()

    move_and_verify(full_pipe, target_device, device_map_full_pipe)
    move_and_verify(model, target_device, device_map_model)

    full_pipe.transformer = model
    full_pipe("an elf on a shelf", num_inference_steps=2, width=16, height=16)
    full_pipe.transformer = None

    move_and_verify(model, "accelerate", device_map_model)
    move_and_verify(full_pipe, "accelerate", device_map_full_pipe)

    full_pipe.transformer = model
    full_pipe("an elf on a shelf", num_inference_steps=2, width=16, height=16)


@pytest.mark.distributed
@pytest.mark.parametrize("target_device", ["cuda", "cpu"])
@pytest.mark.parametrize("model_fixture", ["whisper_tiny_random"], indirect=True)
def test_accelerate_transformer_pipeline_casting(target_device: str | torch.device, model_fixture: Any) -> None:
    """Test that a diffusers model can be cast to the target device."""
    model, _ = model_fixture
    device_map = construct_device_map_manually(model)
    move_to_device(model, target_device, device_map=device_map)
    move_and_verify(model, target_device, device_map)
    move_and_verify(model, "accelerate", device_map)



def move_and_verify(model: Any, target_device: str | torch.device, device_map: dict[str, Any]) -> None:
    """Move the model to the target device and verify that the casting was successful."""
    unique = target_device != "accelerate"
    move_to_device(model, target_device, device_map=device_map)
    # verify that get_device works as intended
    assert get_device(model) == target_device
    # verify that all tensors were actually cast correctly
    if unique:
        assert len(find_unique_devices(model)) == 1
    else:
        assert len(find_unique_devices(model)) > 1


def find_unique_devices(model: Any) -> list[torch.device]:
    """Find all unique devices in the model."""
    if isinstance(model, Pipeline):
        return find_unique_devices(model.model)
    devices = []
    targets = [model] if hasattr(model, "parameters") else [getattr(model, component) for component in model.components]
    for target in targets:
        if not hasattr(target, "parameters"):
            continue
        for param in target.parameters():
            devices.append(param.device)
    return list(set(devices))

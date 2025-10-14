import types
import pytest
import torch
from pruna.engine.handler.handler_inference import (
    set_seed,
    validate_seed_strategy,
)
from pruna.engine.handler.handler_diffuser import DiffuserHandler
from pruna.engine.pruna_model import PrunaModel
from pruna.engine.utils import move_to_device
#  Default handler tests, mainly for checking seeding.

def test_validate_seed_strategy_valid():
    '''Test to see validate_seed_strategy is valid for valid strategies'''
    validate_seed_strategy("per_sample", 42)
    validate_seed_strategy("no_seed", None)


@pytest.mark.parametrize("strategy,seed", [
    ("per_sample", None),
    ("no_seed", 42),
])
def test_validate_seed_strategy_invalid(strategy, seed):
    '''Test to see validate_seed_strategy raises an error for invalid strategies'''
    with pytest.raises(ValueError):
        validate_seed_strategy(strategy, seed)


def test_set_seed_reproducibility():
    '''Test to see set_seed is reproducible'''
    set_seed(42)
    a = torch.randn(3)
    set_seed(42)
    b = torch.randn(3)
    assert torch.equal(a, b)


# Diffuser handler tests, checking output processing and seeding.
@pytest.mark.parametrize("model_fixture",
    [
        pytest.param("flux_tiny_random", marks=pytest.mark.cuda),
    ],
    indirect=["model_fixture"],
    )
def test_assignment_of_diffuser_handler(model_fixture):
    """Check if a diffusion model is assigned to the DiffuserHandler"""
    model, smash_config = model_fixture

    pruna_model = PrunaModel(model, smash_config=smash_config)
    assert isinstance(pruna_model.inference_handler, DiffuserHandler)

@pytest.mark.parametrize("model_fixture, seed, output_attr, return_dict, device",
    [
        pytest.param("flux_tiny_random", 42, "images", True, "cpu", marks=pytest.mark.cpu),
        pytest.param("wan_tiny_random", 42, "frames" ,True, "cuda", marks=pytest.mark.cuda),
        pytest.param("flux_tiny_random", 42, "none", False, "cpu", marks=pytest.mark.cpu),
    ],
    indirect=["model_fixture"],)
def test_process_output_images(model_fixture, seed, output_attr, return_dict, device):
    """Check if the output of the model is processed correctly"""
    input_text = "a photo of a cute prune"

    # Get the output from PrunaModel
    model, smash_config = model_fixture
    smash_config.device = device
    move_to_device(model, device)
    pruna_model = PrunaModel(model, smash_config=smash_config)
    pruna_model.inference_handler.configure_seed("per_sample", global_seed=seed)
    result = pruna_model.run_inference(input_text)

    # Get the output from the pipeline.
    pipe_output = model(input_text, output_type="pt", generator=torch.Generator("cpu").manual_seed(seed), return_dict=return_dict)
    if output_attr != "none":
        pipe_output = getattr(pipe_output, output_attr)
        pipe_output = pipe_output[0]

    assert (result == pipe_output).all().item()


@pytest.mark.parametrize("model_fixture",
    [
        pytest.param("flux_tiny_random", marks=pytest.mark.cpu),
    ],
    indirect=["model_fixture"],)
def test_per_sample_seed_is_applied(model_fixture):
    """Check if samples change per inference run when per_sample seed is applied"""
    model, smash_config = model_fixture
    smash_config.device = "cpu"
    move_to_device(model, "cpu")
    input_text = "a photo of a cute prune"
    pruna_model = PrunaModel(model, smash_config=smash_config)
    pruna_model.inference_handler.configure_seed("per_sample", global_seed=42)
    first_result = pruna_model.run_inference(input_text)
    second_result = pruna_model.run_inference(input_text)
    # If seeding is successfull, each sample should create a different output.
    assert not torch.equal(first_result, second_result)

@pytest.mark.parametrize("model_fixture",
    [
        pytest.param("flux_tiny_random", marks=pytest.mark.cpu),
    ],
    indirect=["model_fixture"],)
def test_seed_is_removed(model_fixture):
    """ Check if seed is removed when no_seed seed is applied"""
    model, smash_config = model_fixture
    smash_config.device = "cpu"
    move_to_device(model, "cpu")
    pruna_model = PrunaModel(model, smash_config=smash_config)
    pruna_model.inference_handler.configure_seed("per_sample", global_seed=42)
    # First check if the seed is set.
    assert pruna_model.inference_handler.model_args["generator"] is not None
    pruna_model.inference_handler.configure_seed("no_seed", None)
    # Then check if the seed generator is removed.
    assert pruna_model.inference_handler.model_args["generator"] is None

import os
import pytest
import torch
from pathlib import Path
from unittest.mock import patch
from transformers import AutoModelForCausalLM
from pruna.config.smash_config import SmashConfig
from pruna import smash
from pruna.engine.save import save_pruna_model
from pruna.engine.save import save_pruna_model_to_hub
from pruna.engine.save import SAVE_FUNCTIONS
from pruna.engine.load import load_pruna_model
from pruna.config.smash_config import SmashConfig
from diffusers import DiffusionPipeline



@pytest.mark.skipif("HF_TOKEN" not in os.environ, reason="HF_TOKEN environment variable is not set, skipping tests.")
@pytest.mark.slow
@pytest.mark.cpu
def test_save_llm_to_hub() -> None:
    """Test saving an LLM model to the Hugging Face Hub."""
    download_repo_id = "hf-internal-testing/tiny-random-llama4"
    upload_repo_id = "PrunaAI/test-save-tiny-random-llama4-smashed"
    model = AutoModelForCausalLM.from_pretrained(download_repo_id)
    smash_config = SmashConfig(device="cpu")
    smash(
        model=model,
        smash_config=smash_config,
    ).save_to_hub(upload_repo_id, private=False)

@pytest.mark.skipif("HF_TOKEN" not in os.environ, reason="HF_TOKEN environment variable is not set, skipping tests.")
@pytest.mark.slow
@pytest.mark.cpu
def test_save_diffusers_to_hub() -> None:
    """Test saving a diffusers model to the Hugging Face Hub."""
    download_repo_id = "hf-internal-testing/tiny-stable-diffusion-pipe"
    upload_repo_id = "PrunaAI/test-save-tiny-stable-diffusion-pipe-smashed"

    model = DiffusionPipeline.from_pretrained(download_repo_id)
    smash_config = SmashConfig(device="cpu")
    smash(
        model=model,
        smash_config=smash_config,
    ).save_to_hub(upload_repo_id, private=False)


@pytest.mark.parametrize(
    "path_type",
    ["string", "pathlib"],
)
@pytest.mark.cpu
def test_save_pruna_model_path_types(tmp_path, path_type: str) -> None:
    """Test saving PrunaModel with different path types (str vs Path)."""
    dummy_model = torch.nn.Linear(10, 5)
    config = SmashConfig()
    config.save_fns = []  
    model_path = tmp_path / "test_model"
    if path_type == "string":
        test_path = str(model_path)
    else:
        test_path = Path(model_path)
    
    save_pruna_model(dummy_model, test_path, config)
    
    assert os.path.exists(test_path)
    assert os.path.exists(os.path.join(test_path, "smash_config.json"))


@pytest.mark.parametrize(
    "path_type",
    ["string", "pathlib"],
)
@pytest.mark.cpu
def test_save_functions_path_types(tmp_path, path_type: str) -> None:
    """Test individual save functions with different path types."""
    
    dummy_model = torch.nn.Linear(5, 3)
    config = SmashConfig()
    
    model_path = tmp_path / "pickled_test"
    if path_type == "string":
        test_path = str(model_path)
    else: 
        test_path = Path(model_path)
    os.makedirs(test_path, exist_ok=True)
    SAVE_FUNCTIONS.pickled(dummy_model, test_path, config)
    expected_file = os.path.join(test_path, "optimized_model.pt")
    assert os.path.exists(expected_file)


@pytest.mark.parametrize(
    "save_path_type,load_path_type",
    [
        ("string", "string"),
        ("string", "pathlib"),
        ("pathlib", "string"),
        ("pathlib", "pathlib"),
    ],
)
@pytest.mark.cpu
def test_save_load_integration_path_types(tmp_path, save_path_type: str, load_path_type: str) -> None:
    """Test integration between save and load with different path types."""
    original_model = torch.nn.Linear(8, 4)
    original_model.weight.data.fill_(1.0) 
    original_model.bias.data.fill_(0.5)
    original_model = original_model.cpu()
    
    config = SmashConfig()
    config.save_fns = []  
    
    model_path = tmp_path / "integration_test"
    if save_path_type == "string":
        save_path = str(model_path)
    else:  
        save_path = Path(model_path)
    
    if load_path_type == "string":
        load_path = str(model_path)
    else:
        load_path = Path(model_path)
    save_pruna_model(original_model, save_path, config)
    loaded_model, loaded_config = load_pruna_model(load_path)
    loaded_model = loaded_model.cpu()
    assert isinstance(loaded_model, torch.nn.Linear)
    assert loaded_model.in_features == 8
    assert loaded_model.out_features == 4
    assert torch.allclose(loaded_model.weight.cpu(), original_model.weight.cpu())
    assert torch.allclose(loaded_model.bias.cpu(), original_model.bias.cpu())


@pytest.mark.cpu
def test_save_to_hub_path_types(tmp_path) -> None:
    """Test save_to_hub with different path types for local model_path."""

    dummy_model = torch.nn.Linear(3, 2)
    config = SmashConfig()
    string_path = str(tmp_path / "string_test")
    pathlib_path = Path(tmp_path / "pathlib_test")
    
    with patch('pruna.engine.save.upload_large_folder') as mock_upload:
        save_pruna_model_to_hub(
            model=dummy_model,
            smash_config=config,
            repo_id="test/repo",
            model_path=string_path,
            private=True
        )
        assert mock_upload.called
        
        mock_upload.reset_mock()
        
        save_pruna_model_to_hub(
            model=dummy_model,
            smash_config=config,
            repo_id="test/repo2",
            model_path=pathlib_path,
            private=True
        )
        assert mock_upload.called
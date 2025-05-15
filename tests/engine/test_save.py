import os
import pytest
from pruna.engine.pruna_model import PrunaModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from pruna.config.smash_config import SmashConfig
from pruna import smash
from diffusers import DiffusionPipeline



@pytest.mark.skipif("HF_TOKEN" not in os.environ, reason="HF_TOKEN environment variable is not set, skipping tests.")
@pytest.mark.slow
@pytest.mark.cpu
def test_save_llm_to_hub() -> None:
    """Test saving an LLM model to the Hugging Face Hub."""
    download_repo_id = "hf-internal-testing/tiny-random-llama4"
    upload_repo_id = "PrunaAI/test-tiny-random-llama4-smashed"
    model = AutoModelForCausalLM.from_pretrained(download_repo_id)
    smash_config = SmashConfig()
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
    upload_repo_id = "PrunaAI/test-tiny-stable-diffusion-pipe-smashed"

    model = DiffusionPipeline.from_pretrained(download_repo_id)
    smash_config = SmashConfig()
    smash(
        model=model,
        smash_config=smash_config,
    ).save_to_hub(upload_repo_id, private=False)
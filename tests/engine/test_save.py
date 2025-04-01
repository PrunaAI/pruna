import os
import shutil

import pytest

from pruna.engine.pruna_model import PrunaModel
from transformers import AutoModelForCausalLM

import os

@pytest.mark.skipif("HF_TOKEN" not in os.environ, reason="HF_TOKEN environment variable is not set, skipping tests.")
@pytest.mark.slow
def test_save_to_hub() -> None:
    """Test PrunaModel.save_to_hub."""
    repo_id = "PrunaAI/opt-125m-smashed"
    save_directory = "saved_model"

    model = PrunaModel.from_hub(repo_id)
    model.save_to_hub(save_directory, )
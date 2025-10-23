import pytest
import torch
from pathlib import Path
import tempfile
from pruna.evaluation.evaluation_agent import EvaluationAgent
from pruna.engine.pruna_model import PrunaModel


@pytest.mark.cuda
@pytest.mark.parametrize("model_fixture, export_format",
[pytest.param("wan_tiny_random", "mp4", marks=pytest.mark.cuda),
pytest.param("wan_tiny_random", "gif", marks=pytest.mark.cuda)], indirect=["model_fixture"])
def test_agent_saves_artifacts(model_fixture, export_format):
    """ Test that the agent runs inference and saves the inference output artifacts correctly."""
    model, smash_config = model_fixture
    # Artifact path
    temp_path = tempfile.mkdtemp()

    # Let's limit the number of data points
    dm  = smash_config.data
    data_points = 2
    dm.limit_datasets(data_points)


    agent = EvaluationAgent(
        request=['background_consistency'],
        datamodule=dm,
        device="cuda",
        save_artifacts=True,
        root_dir=temp_path,
        saving_kwargs={"fps":4},
        artifact_saver_export_format=export_format,
    )

    pruna_model = PrunaModel(model, smash_config)
    pruna_model.inference_handler.model_args["num_inference_steps"] = 1

    agent.evaluate(model=pruna_model)
    files = list(Path(temp_path).rglob(f"*.{export_format}"))

    # Check that we saved the correct number of files
    assert len(files) == data_points

import pytest
import torch
from pathlib import Path
import tempfile
from pruna.evaluation.evaluation_agent import EvaluationAgent
from pruna.engine.pruna_model import PrunaModel


@pytest.mark.cuda
@pytest.mark.parametrize("model_fixture",
[pytest.param("wan_tiny_random", marks=pytest.mark.cuda)], indirect=["model_fixture"])
def test_agent_saves_artifacts(model_fixture):
    model, smash_config = model_fixture
    # Metrics don't work with bfloat16
    model.to(dtype=torch.float16, device="cuda")
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
        artifact_saver_export_format="mp4",
        saving_kwargs={"fps":4}
    )

    pruna_model = PrunaModel(model, smash_config)
    pruna_model.inference_handler.model_args["num_inference_steps"] = 2

    agent.evaluate(model=pruna_model)
    mp4_files = list(Path(temp_path).rglob("*.mp4"))

    # Check that we saved the correct number of files
    assert len(mp4_files) == data_points

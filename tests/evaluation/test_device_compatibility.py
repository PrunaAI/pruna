import pytest
import torch

from ..common import construct_device_map_manually
from pruna.evaluation.task import Task
from pruna.evaluation.evaluation_agent import EvaluationAgent
from pruna.data.pruna_datamodule import PrunaDataModule
from pruna.engine.utils import move_to_device, split_device, device_to_string
from pruna.evaluation.metrics.metric_base import BaseMetric
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.metric_elapsed_time import LatencyMetric
from pruna.evaluation.metrics.metric_pairwise_clip import PairwiseClipScore
from pruna.evaluation.metrics.metric_cmmd import CMMD
from typing import Any, List

@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs ≥2 GPUs to build a split model")
@pytest.mark.parametrize(
    "datamodule_fixture, model_fixture, evaluation_request",
    [
        ("LAION256",  "sd_tiny_random", list(('latency', 'cmmd', 'psnr', 'pairwise_clip_score'))),
        ("LAION256", "sd_tiny_random", list(('latency', 'cmmd', 'psnr', 'pairwise_clip_score'))),
    ],
    indirect=["datamodule_fixture", "model_fixture"],
)
def test_auto_device_task_adapts_to_accelerate_model(datamodule_fixture: PrunaDataModule, model_fixture: Any, evaluation_request: List[str]):

    task = Task(request=evaluation_request, datamodule=datamodule_fixture, device=None)
    agent = EvaluationAgent(task)

    model, _ = model_fixture
    device_map = construct_device_map_manually(model)
    move_to_device(model, "accelerate", device_map=device_map)

    model = agent.prepare_model(model)

    assert split_device(device_to_string(model.get_device())) == split_device(device_to_string(agent.device))
    assert split_device(device_to_string(agent.device)) == split_device(device_to_string(task.device))
    assert split_device(device_to_string(task.stateful_metric_device)) == split_device(device_to_string("cpu"))
    for metric in task.metrics:
        if isinstance(metric, BaseMetric):
            assert split_device(device_to_string(metric.device)) == split_device(device_to_string(task.device))
        elif isinstance(metric, StatefulMetric): # Stateful metrics should not be moved
            if hasattr(metric, "device"):
                assert split_device(device_to_string(metric.device)) == split_device(device_to_string(task.stateful_metric_device))
            elif hasattr(metric, "metric") and hasattr(metric.metric, "device"): # Wrapper metric
                assert split_device(device_to_string(metric.metric.device)) == split_device(device_to_string(task.stateful_metric_device))
            else:
                raise ValueError("Could not find device for metric.")

@pytest.mark.cuda
@pytest.mark.parametrize(
    "datamodule_fixture,model_device, model_fixture, evaluation_request",
    [
        ("LAION256", "cpu", "sd_tiny_random", list(('latency', 'cmmd', 'psnr', 'pairwise_clip_score'))),
        ("LAION256", "cuda", "sd_tiny_random", list(('latency', 'cmmd', 'psnr', 'pairwise_clip_score'))),
    ],
    indirect=["datamodule_fixture", "model_fixture"],
)
def test_auto_device_task_adapts_to_model(datamodule_fixture: PrunaDataModule, model_device: str, model_fixture: Any, evaluation_request: List[str]):
    task = Task(request=evaluation_request, datamodule=datamodule_fixture)
    agent = EvaluationAgent(task)

    model, _ = model_fixture
    move_to_device(model, model_device)
    model = agent.prepare_model(model)

    assert split_device(device_to_string(model.get_device())) == split_device(device_to_string(task.device))
    assert split_device(device_to_string(task.device)) == split_device(device_to_string(agent.device))
    assert split_device(device_to_string(task.stateful_metric_device)) == split_device(device_to_string(task.device))
    for metric in task.metrics:
        if isinstance(metric, BaseMetric):
            assert split_device(device_to_string(metric.device)) == split_device(device_to_string(task.device))
        elif isinstance(metric, StatefulMetric):
            if hasattr(metric, "device"):
                assert split_device(device_to_string(metric.device)) == split_device(device_to_string(task.device))
            elif hasattr(metric, "metric") and hasattr(metric.metric, "device"):
                assert split_device(device_to_string(metric.metric.device)) == split_device(device_to_string(task.device))
            else:
                raise ValueError("Could not find device for metric.")


@pytest.mark.cuda
@pytest.mark.parametrize(
    "datamodule_fixture,model_fixture,task_device,model_device",
    [
        ("LAION256", "sd_tiny_random", "cuda", "cpu"),
        ("LAION256", "sd_tiny_random", "cpu", "cuda"),
    ],
    indirect=["datamodule_fixture", "model_fixture"],
)
def test_ensure_task_model_device_mismatch_raises(datamodule_fixture: PrunaDataModule, model_fixture: Any, task_device: str, model_device: str):
    task = Task(request=["latency", "cmmd", "pairwise_clip_score"], datamodule=datamodule_fixture, device=task_device)
    agent = EvaluationAgent(task)

    model, _ = model_fixture
    move_to_device(model, model_device)
    with pytest.raises(ValueError):
        model = agent.prepare_model(model)

@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs ≥2 GPUs to build a split model")
@pytest.mark.parametrize(
    "datamodule_fixture,model_fixture,task_device",
    [
        ("LAION256", "sd_tiny_random", "cuda"),
        ("LAION256", "sd_tiny_random", "cpu"),
    ],
    indirect=["datamodule_fixture", "model_fixture"],
)
def test_ensure_task_model_accelerate_device_mismatch_raises(datamodule_fixture: PrunaDataModule, model_fixture: Any, task_device:str):
    task = Task(request=["latency", "cmmd", "pairwise_clip_score"], datamodule=datamodule_fixture, device=task_device)
    agent = EvaluationAgent(task)

    model, _ = model_fixture
    device_map = construct_device_map_manually(model)
    move_to_device(model, "accelerate", device_map=device_map)

    with pytest.raises(ValueError):
        agent.prepare_model(model)


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs ≥2 GPUs to build a split model")
@pytest.mark.parametrize(
    "datamodule_fixture,model_fixture",
    [
        ("LAION256", "sd_tiny_random"),
    ],
    indirect=["datamodule_fixture", "model_fixture"],
)
def test_mismatched_metric_instances_adapts_to_model(datamodule_fixture: PrunaDataModule, model_fixture: Any):
    latency = LatencyMetric(device="cpu")
    cmmd = CMMD(device="cpu")
    pairwise_clip_score = PairwiseClipScore(device="cpu")

    task = Task(request=[latency, cmmd, pairwise_clip_score], datamodule=datamodule_fixture)
    task_device , task_idx = split_device(device_to_string(task.device))

    # Right now auto device casting will set the task device to cuda.
    # We need to make sure to write additional tests for when it can return a different device.
    assert task_device == "cuda"
    agent = EvaluationAgent(task)

    model, _ = model_fixture
    device_map = construct_device_map_manually(model)
    move_to_device(model, "accelerate", device_map=device_map)

    model = agent.prepare_model(model)
    assert split_device(device_to_string(model.get_device())) == split_device(device_to_string(task.device))
    assert split_device(device_to_string(task.device)) == split_device(device_to_string(agent.device))
    assert split_device(device_to_string(task.stateful_metric_device)) == split_device(device_to_string("cpu"))
    for metric in task.metrics:
        if isinstance(metric, BaseMetric):
            assert split_device(device_to_string(metric.device)) == split_device(device_to_string(task.device))
        elif isinstance(metric, StatefulMetric):
            if hasattr(metric, "device"):
                assert split_device(device_to_string(metric.device)) == split_device(device_to_string(task.stateful_metric_device))
            elif hasattr(metric, "metric") and hasattr(metric.metric, "device"):
                assert split_device(device_to_string(metric.metric.device)) == split_device(device_to_string(task.stateful_metric_device))
            else:
                raise ValueError("Could not find device for metric.")

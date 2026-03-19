from pathlib import Path
from unittest.mock import MagicMock, patch

import PIL.Image
import pytest
import torch
from datasets import Dataset

from pruna.data.pruna_datamodule import PrunaDataModule
from pruna.evaluation.metrics.metric_rapiddata import METRIC_RAPIDATA, RapidataMetric
from pruna.evaluation.metrics.result import CompositeMetricResult


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def metric(mock_client):
    return RapidataMetric(client=mock_client)


@pytest.fixture
def metric_with_benchmark(metric):
    benchmark = MagicMock()
    benchmark.id = "bench-123"
    benchmark.leaderboards = []
    metric.benchmark = benchmark
    return metric


@pytest.fixture
def metric_ready(metric_with_benchmark):
    metric_with_benchmark.set_current_context("test-model")
    return metric_with_benchmark


# Initialization with / without a client
def test_default_client_created_when_none_provided():
    """Test that a RapidataClient is created when none is provided."""
    with patch("pruna.evaluation.metrics.metric_rapiddata.RapidataClient") as mock_cls:
        mock_cls.return_value = MagicMock()
        m = RapidataMetric()
        mock_cls.assert_called_once()


def test_custom_client_used(mock_client):
    """Test that a custom client is used when provided."""
    m = RapidataMetric(client=mock_client)
    assert m.client is mock_client


# Creation from existing benchmark
def test_from_benchmark():
    """Test creating a metric from an existing benchmark."""
    benchmark = MagicMock()
    with patch("pruna.evaluation.metrics.metric_rapiddata.RapidataClient"):
        m = RapidataMetric.from_benchmark(benchmark)
    assert m.benchmark is benchmark


# Creation from benchmark ID
def test_from_benchmark_id():
    """Test creating a metric from a benchmark ID."""
    with patch("pruna.evaluation.metrics.metric_rapiddata.RapidataClient") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        mock_instance.mri.get_benchmark_by_id.return_value = MagicMock(id="abc")
        m = RapidataMetric.from_benchmark_id("abc")
        mock_instance.mri.get_benchmark_by_id.assert_called_once_with("abc")
        assert m.benchmark is not None


def test_create_benchmark_with_prompt_list(metric, mock_client):
    """Test creating a benchmark with a list of prompts."""
    prompts = ["a cat", "a dog"]
    metric.create_benchmark("my-bench", data=prompts)
    mock_client.mri.create_new_benchmark.assert_called_once_with("my-bench", prompts=prompts)
    assert metric.benchmark is not None


def test_create_benchmark_from_datamodule(metric, mock_client):
    """Test creating a benchmark from a PrunaDataModule."""
    ds = Dataset.from_dict({"text": ["prompt1", "prompt2"]})
    dm = PrunaDataModule(train_ds=ds, val_ds=ds, test_ds=ds, collate_fn=lambda x: x, dataloader_args={})

    metric.create_benchmark("my-bench", data=dm, split="test")
    mock_client.mri.create_new_benchmark.assert_called_once_with("my-bench", prompts=["prompt1", "prompt2"])


def test_create_benchmark_raises_if_already_exists(metric_with_benchmark):
    """Test that creating a benchmark twice raises."""
    with pytest.raises(ValueError, match="Benchmark already created"):
        metric_with_benchmark.create_benchmark("dup", data=["x"])

def test_create_request_raises_without_benchmark(metric):
    """Test that create_request raises without a benchmark."""
    with pytest.raises(ValueError, match="No benchmark configured"):
        metric.create_request("quality", "Rate image quality")


def test_create_request_delegates_to_leaderboard(metric_with_benchmark):
    """Test that create_request delegates to the benchmark."""
    metric_with_benchmark.create_request("quality", "Rate image quality")
    metric_with_benchmark.benchmark.create_leaderboard.assert_called_once_with(
        "quality", "Rate image quality", False
    )


def test_set_current_context_resets_caches(metric_ready):
    """Test that set_current_context resets the caches."""
    metric_ready.prompt_cache.append("leftover")
    metric_ready.media_cache.append("leftover")
    metric_ready.set_current_context("model-b")
    assert metric_ready.prompt_cache == []
    assert metric_ready.media_cache == []


def test_update_accumulates_prompts_and_media(metric_ready):
    """Test that update accumulates prompts and media."""
    x = ["a cat on a sofa", "a dog in rain"]
    gt = [None, None]
    outputs = [torch.rand(3, 64, 64), torch.rand(3, 64, 64)]
    metric_ready.update(x, gt, outputs)

    assert metric_ready.prompt_cache == x
    assert len(metric_ready.media_cache) == 2


def test_update_raises_without_benchmark(metric):
    """Test that update raises without a benchmark."""
    metric.current_benchmarked_model = "m"
    with pytest.raises(ValueError, match="No benchmark configured"):
        metric.update(["p"], [None], [torch.rand(3, 32, 32)])


def test_update_raises_without_model(metric_with_benchmark):
    """Test that update raises without a model context."""
    with pytest.raises(ValueError, match="No model set"):
        metric_with_benchmark.update(["p"], [None], [torch.rand(3, 32, 32)])


def test_prepare_media_string_passthrough(metric_ready):
    """Test that string URLs/paths are passed through as-is."""
    metric_ready.media_cache = ["https://example.com/img.png", "/tmp/local.png"]
    paths = metric_ready._prepare_media_for_upload()
    assert paths == ["https://example.com/img.png", "/tmp/local.png"]
    metric_ready._cleanup_temp_media()


def test_prepare_media_pil_image(metric_ready):
    """Test that PIL images are saved to temp files."""
    img = PIL.Image.new("RGB", (64, 64), color="red")
    metric_ready.media_cache = [img]
    paths = metric_ready._prepare_media_for_upload()
    assert len(paths) == 1
    assert Path(paths[0]).exists()
    metric_ready._cleanup_temp_media()


def test_prepare_media_tensor(metric_ready):
    """Test that tensors are saved to temp files."""
    tensor = torch.rand(3, 64, 64)
    metric_ready.media_cache = [tensor]
    paths = metric_ready._prepare_media_for_upload()
    assert len(paths) == 1
    assert Path(paths[0]).exists()
    metric_ready._cleanup_temp_media()


def test_prepare_media_tensor_uint8_range(metric_ready):
    """Test that tensors in 0-255 range are normalised before saving."""
    tensor = torch.randint(0, 256, (3, 32, 32)).float()
    assert tensor.max() > 1.0
    metric_ready.media_cache = [tensor]
    paths = metric_ready._prepare_media_for_upload()
    assert len(paths) == 1
    assert Path(paths[0]).exists()
    metric_ready._cleanup_temp_media()


def test_prepare_media_unsupported_type_raises(metric_ready):
    """Test that unsupported media types raise."""
    metric_ready.media_cache = [12345]
    with pytest.raises(TypeError, match="Unsupported media type"):
        metric_ready._prepare_media_for_upload()


def test_compute_submits_to_rapidata(metric_ready):
    """Test that compute submits the accumulated data."""
    img = PIL.Image.new("RGB", (32, 32))
    metric_ready.media_cache = [img]
    metric_ready.prompt_cache = ["a cat"]
    metric_ready.compute()
    metric_ready.benchmark.evaluate_model.assert_called_once()
    call_kwargs = metric_ready.benchmark.evaluate_model.call_args
    assert call_kwargs[0][0] == "test-model"


def test_compute_raises_when_cache_empty(metric_ready):
    """Test that compute raises when no data has been accumulated."""
    with pytest.raises(ValueError, match="No data accumulated"):
        metric_ready.compute()


def test_compute_raises_without_model_context(metric_with_benchmark):
    """Test that compute raises without a model context."""
    with pytest.raises(ValueError, match="No model set"):
        metric_with_benchmark.compute()


def test_compute_cleans_up_temp_dir(metric_ready):
    """Test that compute removes the temp directory after submission."""
    metric_ready.media_cache = [torch.rand(3, 32, 32)]
    metric_ready.prompt_cache = ["test"]
    metric_ready.compute()
    assert not hasattr(metric_ready, "_temp_dir") or not metric_ready._temp_dir.exists()


def test_retrieve_results_returns_composite_result(metric_with_benchmark):
    """Test that retrieve_results returns a CompositeMetricResult."""
    metric_with_benchmark.benchmark.get_overall_standings.return_value = {
        "name": ["model-a", "model-b"],
        "score": [0.85, 0.72],
    }
    result = metric_with_benchmark.retrieve_results()
    assert isinstance(result, CompositeMetricResult)
    assert result.name == METRIC_RAPIDATA
    assert result.result == {"model-a": 0.85, "model-b": 0.72}
    assert result.higher_is_better is True


def test_retrieve_results_raises_without_benchmark(metric):
    """Test that retrieve_results raises without a benchmark."""
    with pytest.raises(ValueError, match="No benchmark configured"):
        metric.retrieve_results()


def test_retrieve_results_reraises_non_validation_error(metric_with_benchmark):
    """Test that non-validation errors are re-raised."""
    metric_with_benchmark.benchmark.get_overall_standings.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        metric_with_benchmark.retrieve_results()


def test_retrieve_granular_results_per_leaderboard(metric_with_benchmark):
    """Test that granular results returns one result per leaderboard."""
    lb = MagicMock()
    lb.name = "quality"
    lb.instruction = "Rate quality"
    lb.get_standings.return_value = {
        "name": ["model-a"],
        "score": [0.9],
    }
    metric_with_benchmark.benchmark.leaderboards = [lb]
    results = metric_with_benchmark.retrieve_granular_results()
    assert len(results) == 1
    assert results[0].name == "quality"
    assert results[0].params == {"instruction": "Rate quality"}
    assert results[0].result == {"model-a": 0.9}


def test_retrieve_granular_results_raises_without_benchmark(metric):
    """Test that granular results raises without a benchmark."""
    with pytest.raises(ValueError, match="No benchmark configured"):
        metric.retrieve_granular_results()
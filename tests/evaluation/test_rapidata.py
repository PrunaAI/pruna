import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import PIL.Image
import pytest
import torch
from datasets import Dataset

from pruna.data.pruna_datamodule import PrunaDataModule
from pruna.evaluation.metrics.metric_rapiddata import RapidataMetric
from rapidata.rapidata_client.benchmark.rapidata_benchmark import RapidataBenchmark
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
    metric.higher_is_better = True
    return metric


@pytest.fixture
def metric_ready(metric_with_benchmark):
    metric_with_benchmark.current_context = "test-model"
    return metric_with_benchmark

@pytest.fixture
def metric_ready_with_cleanup(metric_ready):
    """metric_ready that auto-cleans temp media after the test."""
    yield metric_ready
    metric_ready._cleanup_temp_media()


# Initialization with / without a client
def test_default_client_created_when_none_provided():
    """Test that a RapidataClient is created when none is provided."""
    with patch("pruna.evaluation.metrics.metric_rapiddata.RapidataClient") as mock_cls:
        mock_cls.return_value = MagicMock()
        _ = RapidataMetric()
        mock_cls.assert_called_once()


def test_custom_client_used(mock_client):
    """Test that a custom client is used when provided."""
    m = RapidataMetric(client=mock_client)
    assert m.client is mock_client


# Creation from existing benchmark
def test_from_benchmark():
    """Test creating a metric from an existing benchmark."""
    benchmark = MagicMock(spec=RapidataBenchmark)
    with patch("pruna.evaluation.metrics.metric_rapiddata.RapidataClient"):
        m = RapidataMetric.from_rapidata_benchmark(benchmark)
    assert m.benchmark is benchmark


# Creation from benchmark ID
def test_from_benchmark_id():
    """Test creating a metric from a benchmark ID."""
    with patch("pruna.evaluation.metrics.metric_rapiddata.RapidataClient") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        mock_instance.mri.get_benchmark_by_id.return_value = MagicMock(id="abc")
        m = RapidataMetric.from_rapidata_benchmark("abc")
        mock_instance.mri.get_benchmark_by_id.assert_called_once_with("abc")
        assert m.benchmark is not None


def test_create_benchmark_with_prompt_list(metric, mock_client):
    """Test creating a benchmark with a list of prompts."""
    prompts = ["a cat", "a dog"]
    metric.create_benchmark("my-bench", data=prompts)
    mock_client.mri.create_new_benchmark.assert_called_once_with("my-bench", prompts=prompts, prompt_assets=None)
    assert metric.benchmark is not None


def test_create_benchmark_from_datamodule(metric, mock_client):
    """Test creating a benchmark from a PrunaDataModule."""
    ds = Dataset.from_dict({"text": ["prompt1", "prompt2"]})
    dm = PrunaDataModule(train_ds=ds, val_ds=ds, test_ds=ds, collate_fn=lambda x: x, dataloader_args={})

    metric.create_benchmark("my-bench", data=dm, split="test")
    mock_client.mri.create_new_benchmark.assert_called_once_with("my-bench", prompts=["prompt1", "prompt2"], prompt_assets=None)


def test_create_benchmark_raises_if_already_exists(metric_with_benchmark):
    """Test that creating a benchmark twice raises."""
    with pytest.raises(ValueError, match="Benchmark already created"):
        metric_with_benchmark.create_benchmark("dup", data=["x"])

def test_create_async_request_raises_without_benchmark(metric):
    """Test that create_async_request raises without a benchmark."""
    with pytest.raises(ValueError, match="No benchmark configured"):
        metric.create_async_request("quality", "Rate image quality")


def test_create_async_request_delegates_to_leaderboard(metric_with_benchmark):
    """Test that create_async_request delegates to the benchmark."""
    metric_with_benchmark.create_async_request("quality", "Rate image quality")
    metric_with_benchmark.benchmark.create_leaderboard.assert_called_once_with(
        "quality", "Rate image quality", False, False
    )


def test_set_current_context_resets_caches(metric_ready):
    """Test that set_current_context resets the caches."""
    metric_ready.prompt_cache.append("leftover")
    metric_ready.media_cache.append("leftover")
    metric_ready.current_context = "model-b"
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
    metric.current_context = "m"
    with pytest.raises(ValueError, match="No benchmark configured"):
        metric.update(["p"], [None], [torch.rand(3, 32, 32)])


def test_update_raises_without_context(metric_with_benchmark):
    """Test that update raises without a model context."""
    with pytest.raises(ValueError, match="No context set. Set current_context first."):
        metric_with_benchmark.update(["p"], [None], [torch.rand(3, 32, 32)])

def test_prepare_media_string_passthrough(metric_ready_with_cleanup):
    """Test that string URLs/paths are passed through as-is."""
    local_path = os.path.join(tempfile.gettempdir(), "local.png")
    metric_ready_with_cleanup.media_cache = ["https://example.com/img.png", local_path]
    paths = metric_ready_with_cleanup._prepare_media_for_upload()
    assert paths == ["https://example.com/img.png", local_path]


def test_prepare_media_pil_image(metric_ready_with_cleanup):
    """Test that PIL images are saved to temp files."""
    img = PIL.Image.new("RGB", (64, 64), color="red")
    metric_ready_with_cleanup.media_cache = [img]
    paths = metric_ready_with_cleanup._prepare_media_for_upload()
    assert len(paths) == 1
    assert Path(paths[0]).exists()


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
    with pytest.raises(ValueError, match="No context set. Set current_context first."):
        metric_with_benchmark.compute()


def test_compute_cleans_up_temp_dir(metric_ready):
    """Test that compute removes the temp directory after submission."""
    metric_ready.media_cache = [torch.rand(3, 32, 32)]
    metric_ready.prompt_cache = ["test"]
    metric_ready.compute()
    assert not hasattr(metric_ready, "_temp_dir") or not metric_ready._temp_dir.exists()


class _FakeValidationError(Exception):
    pass


def test_is_not_ready_error_recognises_validation_error():
    assert RapidataMetric._is_not_ready_error(_FakeValidationError()) is True
    assert RapidataMetric._is_not_ready_error(RuntimeError()) is False


def test_retrieve_non_blocking_returns_result_when_ready(metric_with_benchmark):
    metric_with_benchmark.benchmark.get_overall_standings.return_value = {
        "name": ["model-a", "model-b"], "score": [0.85, 0.72],
    }
    result = metric_with_benchmark.retrieve_async_results()
    assert isinstance(result, CompositeMetricResult)
    assert result.result == {"model-a": 0.85, "model-b": 0.72}


def test_retrieve_non_blocking_returns_none_when_not_ready(metric_with_benchmark):
    metric_with_benchmark.benchmark.get_overall_standings.side_effect = _FakeValidationError()
    assert metric_with_benchmark.retrieve_async_results() is None


def test_retrieve_non_blocking_granular_returns_partial(metric_with_benchmark):
    lb_ready = MagicMock(name="quality", instruction="Rate quality", inverse_ranking=False)
    lb_ready.get_standings.return_value = {"name": ["m-a"], "score": [0.9]}
    lb_pending = MagicMock(name="alignment")
    lb_pending.get_standings.side_effect = _FakeValidationError()
    metric_with_benchmark.benchmark.leaderboards = [lb_ready, lb_pending]

    results = metric_with_benchmark.retrieve_async_results(is_granular=True)
    assert len(results) == 1
    assert results[0].result == {"m-a": 0.9}


def test_retrieve_reraises_non_validation_error(metric_with_benchmark):
    metric_with_benchmark.benchmark.get_overall_standings.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        metric_with_benchmark.retrieve_async_results()


@patch("pruna.evaluation.metrics.metric_rapiddata.time")
def test_retrieve_blocking_polls_until_ready(mock_time, metric_with_benchmark):
    _clock = iter(range(0, 1000, 10))
    mock_time.monotonic.side_effect = lambda: next(_clock)

    standings = {"name": ["m-a"], "score": [0.9]}
    metric_with_benchmark.benchmark.get_overall_standings.side_effect = [
        _FakeValidationError(), _FakeValidationError(), standings,
    ]
    result = metric_with_benchmark.retrieve_async_results(is_blocking=True, timeout=60, poll_interval=5)
    assert isinstance(result, CompositeMetricResult)
    assert result.result == {"m-a": 0.9}
    assert mock_time.sleep.call_count == 2


@patch("pruna.evaluation.metrics.metric_rapiddata.time")
def test_retrieve_blocking_raises_timeout(mock_time, metric_with_benchmark):
    _clock = iter(range(0, 1000, 30))
    mock_time.monotonic.side_effect = lambda: next(_clock)
    metric_with_benchmark.benchmark.get_overall_standings.side_effect = _FakeValidationError()

    with pytest.raises(TimeoutError, match="not ready after 60s"):
        metric_with_benchmark.retrieve_async_results(is_blocking=True, timeout=60, poll_interval=5)


def test_create_benchmark_forwards_explicit_data_assets(metric, mock_client):
    """Explicit data_assets are forwarded as prompt_assets."""
    prompts = ["edit this", "fix that"]
    assets = ["/imgs/a.png", "/imgs/b.png"]
    metric.create_benchmark("bench", data=prompts, data_assets=assets)
    mock_client.mri.create_new_benchmark.assert_called_once_with(
        "bench", prompts=prompts, prompt_assets=assets,
    )


def test_create_benchmark_datamodule_extracts_images(metric, mock_client):
    """PrunaDataModule with an 'image' column extracts and converts images to prompt_assets."""
    from datasets import Features, Image as HFImage, Value
    img1 = PIL.Image.new("RGB", (32, 32), "red")
    img2 = PIL.Image.new("RGB", (32, 32), "blue")
    ds = Dataset.from_dict(
        {"text": ["prompt1", "prompt2"], "image": [img1, img2]},
        features=Features({"text": Value("string"), "image": HFImage()}),
    )
    dm = PrunaDataModule(train_ds=ds, val_ds=ds, test_ds=ds, collate_fn=lambda x: x, dataloader_args={})
    fake_paths = [os.path.join(tempfile.gettempdir(), f"{i}.png") for i in range(2)]
    with patch.object(metric, "_prepare_media_for_upload", return_value=fake_paths) as mock_prep:
        metric.create_benchmark("my-bench", data=dm, split="test")
        mock_prep.assert_called_once()
        images_arg = mock_prep.call_args[0][0]
        assert len(images_arg) == 2
        assert all(isinstance(img, PIL.Image.Image) for img in images_arg)
    mock_client.mri.create_new_benchmark.assert_called_once_with(
        "my-bench", prompts=["prompt1", "prompt2"], prompt_assets=fake_paths,
    )


def test_create_benchmark_datamodule_ignores_explicit_data_assets(metric, mock_client):
    """When using a PrunaDataModule, explicit data_assets are overridden."""
    ds = Dataset.from_dict({"text": ["p1"]})
    dm = PrunaDataModule(train_ds=ds, val_ds=ds, test_ds=ds, collate_fn=lambda x: x, dataloader_args={})
    metric.create_benchmark("bench", data=dm, data_assets=["/should/be/ignored.png"])
    mock_client.mri.create_new_benchmark.assert_called_once_with(
        "bench", prompts=["p1"], prompt_assets=None,
    )


def test_create_async_request_forwards_show_prompt_assets_true(metric_with_benchmark):
    """show_prompt_assets=True is forwarded to create_leaderboard."""
    metric_with_benchmark.create_async_request("quality", "Rate quality", show_prompt_assets=True)
    metric_with_benchmark.benchmark.create_leaderboard.assert_called_once_with(
        "quality", "Rate quality", False, True,
    )


def test_prepare_media_uses_explicit_list_over_cache(metric_ready):
    """Passing an explicit media list uses it instead of media_cache."""
    metric_ready.media_cache = [torch.rand(3, 32, 32)]  # should be ignored
    explicit = [PIL.Image.new("RGB", (16, 16))]

    paths = metric_ready._prepare_media_for_upload(explicit)
    assert len(paths) == 1
    assert Path(paths[0]).exists()
    loaded = PIL.Image.open(paths[0])
    assert loaded.size == (16, 16)
    metric_ready._cleanup_temp_media()
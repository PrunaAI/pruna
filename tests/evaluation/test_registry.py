import uuid

import pytest

from pruna.evaluation.metrics.metric_base import BaseMetric
from pruna.evaluation.metrics.registry import MetricRegistry


class MockMetric(BaseMetric):
    """A simple mock metric for testing the registry."""

    def __init__(self, name: str = "mock_metric", **kwargs):
        super().__init__()
        self.name = name
        self.kwargs = kwargs

    def compute(self, *args, **kwargs):
        return {"mock_result": 42}


@pytest.fixture
def registry():
    """Provide a clean registry for each test."""
    return MetricRegistry()


@pytest.fixture
def registered_metric(registry):
    """Register a test metric and return the registry, metric class, and name."""
    metric_name = f"test_metric_{uuid.uuid4().hex[:8]}"

    @registry.register(metric_name)
    class TestMetric(MockMetric):
        pass

    return registry, TestMetric, metric_name


@pytest.fixture
def wrapper_registry():
    """Setup wrapper metrics with an isolated registry."""
    test_registry = MetricRegistry()
    wrapper_metrics = [f"wrapped_a_{uuid.uuid4().hex[:8]}", f"wrapped_b_{uuid.uuid4().hex[:8]}"]

    @test_registry.register_wrapper(available_metrics=wrapper_metrics)
    class WrapperMetric(MockMetric):
        def __init__(self, metric_name: str, **kwargs):
            super().__init__(name=metric_name, **kwargs)
            self.metric_name = metric_name

    return test_registry, WrapperMetric, wrapper_metrics


def test_get_single_metric(registered_metric):
    """Test retrieving a single metric from the registry."""
    registry, TestMetric, metric_name = registered_metric
    custom_param = "value"

    metric = registry.get_metric(metric_name, custom_param=custom_param)

    assert isinstance(metric, TestMetric)
    assert metric.kwargs["custom_param"] == custom_param


def test_get_multiple_metrics(registered_metric):
    """Test retrieving multiple metrics at once."""
    registry, TestMetric, metric_name = registered_metric
    custom_param = "multi_value"

    # Test with a single metric
    metrics_single = registry.get_metrics([metric_name], custom_param=custom_param)
    assert len(metrics_single) == 1
    assert isinstance(metrics_single[0], TestMetric)
    assert metrics_single[0].kwargs["custom_param"] == custom_param

    # Test with multiple instances of the same metric
    metrics_multiple = registry.get_metrics([metric_name, metric_name], custom_param=custom_param)
    assert len(metrics_multiple) == 2
    for metric in metrics_multiple:
        assert isinstance(metric, TestMetric)
        assert metric.kwargs["custom_param"] == custom_param


def test_register_duplicate(registered_metric):
    """Test that registering a duplicate name logs an error but doesn't raise."""
    registry, TestMetric, metric_name = registered_metric

    # This should not raise an exception, but will log an error
    @registry.register(metric_name)
    class DuplicateMetric(MockMetric):
        pass

    # The original registration should still be in place
    metric = registry.get_metric(metric_name)
    assert isinstance(metric, TestMetric)
    assert not isinstance(metric, DuplicateMetric)


def test_get_nonexistent_metric(registry):
    """Test that getting a non-existent metric raises a ValueError."""
    nonexistent_name = f"nonexistent_{uuid.uuid4().hex[:8]}"

    with pytest.raises(ValueError, match=f"Metric '{nonexistent_name}' is not registered"):
        registry.get_metric(nonexistent_name)


def test_wrapper_metrics(wrapper_registry):
    """Test metrics registered through a wrapper."""
    registry, WrapperMetric, wrapper_metrics = wrapper_registry

    # Test first wrapper metric
    metric_a = registry.get_metric(wrapper_metrics[0], custom_param="a_value")
    assert isinstance(metric_a, WrapperMetric)
    assert metric_a.metric_name == wrapper_metrics[0]
    assert metric_a.kwargs["custom_param"] == "a_value"

    # Test second wrapper metric
    metric_b = registry.get_metric(wrapper_metrics[1], custom_param="b_value")
    assert isinstance(metric_b, WrapperMetric)
    assert metric_b.metric_name == wrapper_metrics[1]
    assert metric_b.kwargs["custom_param"] == "b_value"

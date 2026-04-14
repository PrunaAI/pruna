# flake8: noqa
"""Tests for the metrics module.

Tests cover:
1. Decorator usage (with and without parameters)
2. Direct counter usage
3. Error handling
4. Metric labels/attributes
5. Enable/disable functionality
6. Config persistence
"""

import asyncio
import functools
import logging
import os
import uuid
from unittest.mock import MagicMock, mock_open, patch

import pytest

from pruna.telemetry import increment_counter, set_telemetry_metrics, track_usage
from pruna.telemetry.metrics import (
    _save_metrics_config,
    is_metrics_enabled,
    set_opentelemetry_log_level,
)

@pytest.fixture
def mock_config():
    return {
        "metrics_enabled": False,
        "otlp_endpoint": "http://localhost:4318/v1/metrics",
    }


@pytest.fixture(autouse=True)
def reset_metrics_state():
    """Reset metrics state before each test."""
    # Clear environment variable
    if "PRUNA_METRICS_ENABLED" in os.environ:
        del os.environ["PRUNA_METRICS_ENABLED"]
    yield


@pytest.fixture(autouse=True)
def mock_file_operations(mock_config):
    """Mock configuration file operations."""
    with patch("builtins.open", mock_open()), patch("yaml.safe_load", return_value=mock_config), patch(
        "yaml.safe_dump"
    ), patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.mkdir"):
        yield


@pytest.fixture(autouse=True)
def setup_metrics():
    """Set up metrics testing environment."""
    mock_counter = MagicMock()
    with patch("pruna.telemetry.metrics.function_counter", mock_counter):
        yield mock_counter


def assert_metric_labels(
    mock_counter, expected_function: str, expected_status: str = "success", expected_config: str = ""
):
    """Helper to verify metric labels including session_id.

    Args:
        mock_counter: The mocked counter
        expected_function: Expected function name
        expected_status: Expected status (success/error)
        expected_config: Expected smash config
    """
    labels = mock_counter.add.call_args[0][1]

    # Verify UUID format but don't compare exact value
    session_id = labels.pop("session_id")
    try:
        uuid.UUID(session_id)
    except ValueError:
        pytest.fail(f"Invalid UUID format for session: {session_id}")

    # Compare remaining labels
    assert labels == {
        "function": expected_function,
        "status": expected_status,
        "smash_config": expected_config,
    }


def test_basic_decorator(setup_metrics):
    """Test basic decorator without parameters."""
    set_telemetry_metrics(True)

    @track_usage
    def test_func():
        return "success"

    result = test_func()
    assert result == "success"
    setup_metrics.add.assert_called_once()
    assert_metric_labels(setup_metrics, "test_metrics.test_basic_decorator.<locals>.test_func")


def test_named_decorator(setup_metrics):
    """Test decorator with custom name."""
    set_telemetry_metrics(True)

    @track_usage("custom_operation")
    def test_func():
        return "success"

    result = test_func()
    assert result == "success"
    setup_metrics.add.assert_called_once()
    assert_metric_labels(setup_metrics, "custom_operation")


def test_error_tracking(setup_metrics):
    """Test error tracking in decorator."""
    set_telemetry_metrics(True)

    @track_usage
    def failing_func():
        raise ValueError("test error")

    with pytest.raises(ValueError):
        failing_func()

    setup_metrics.add.assert_called_once()
    assert_metric_labels(
        setup_metrics, "test_metrics.test_error_tracking.<locals>.failing_func", expected_status="error"
    )


def test_decorator_preserves_function_metadata():
    """Test that decorator preserves function metadata."""
    from pruna.telemetry.metrics import track_usage

    @track_usage
    def test_func():
        """Test docstring."""
        pass

    assert test_func.__name__ == "test_func"
    assert test_func.__doc__ == "Test docstring."


def test_nested_decorators(setup_metrics):
    """Test decorator works with other decorators."""
    set_telemetry_metrics(True)

    def other_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    @track_usage
    @other_decorator
    def test_func():
        return "success"

    result = test_func()
    assert result == "success"
    setup_metrics.add.assert_called_once()
    assert_metric_labels(setup_metrics, "test_metrics.test_nested_decorators.<locals>.test_func")


def test_async_function(setup_metrics):
    """Test decorator works with async functions."""
    set_telemetry_metrics(True)

    @track_usage
    async def async_func():
        return "success"

    result = asyncio.run(async_func())
    assert result == "success"
    setup_metrics.add.assert_called_once()
    assert_metric_labels(setup_metrics, "test_metrics.test_async_function.<locals>.async_func")


def test_metrics_default_state():
    """Test default metrics state is disabled."""

    assert not is_metrics_enabled()


def test_set_metrics_state(setup_metrics):
    """Test enabling and disabling metrics."""
    # Test default state
    assert not is_metrics_enabled()
    increment_counter("test")
    setup_metrics.add.assert_not_called()

    # Test enabled state
    set_telemetry_metrics(True)
    assert is_metrics_enabled()
    increment_counter("test")
    setup_metrics.add.assert_called_once()
    assert_metric_labels(setup_metrics, "test")

    # Test disabled state
    set_telemetry_metrics(False)
    assert not is_metrics_enabled()
    setup_metrics.reset_mock()
    increment_counter("test")
    setup_metrics.add.assert_not_called()


def test_direct_counter_usage(setup_metrics):
    """Test using the counter directly for manual operations."""
    set_telemetry_metrics(True)

    # Track successful operation
    increment_counter("manual_operation")
    setup_metrics.add.assert_called_once()
    assert_metric_labels(setup_metrics, "manual_operation")

    # Track failed operation
    setup_metrics.reset_mock()
    increment_counter("manual_operation", success=False)
    setup_metrics.add.assert_called_once()
    assert_metric_labels(setup_metrics, "manual_operation", expected_status="error")

    # Verify total calls
    assert setup_metrics.add.call_count == 1  # Since we reset_mock


def test_metrics_state_precedence(mock_file_operations):
    """Test that env var takes precedence over config file."""
    # When only config file exists
    assert is_metrics_enabled() is False  # From mock_file_operations fixture

    # When env var exists, it takes precedence
    os.environ["PRUNA_METRICS_ENABLED"] = "true"
    assert is_metrics_enabled() is True


def test_load_metrics_state(reset_metrics_state):
    """Test loading metrics state."""
    # Test env var states
    os.environ["PRUNA_METRICS_ENABLED"] = "true"
    assert is_metrics_enabled() is True

    os.environ["PRUNA_METRICS_ENABLED"] = "false"
    assert is_metrics_enabled() is False

    # Test default when no env var
    del os.environ["PRUNA_METRICS_ENABLED"]
    assert is_metrics_enabled() is False


def test_metrics_config_persistence(mock_file_operations, mock_config):
    """Test that metrics state is persisted to config only when requested."""
    with patch("yaml.safe_dump") as mock_dump:
        # Should not save to config by default
        set_telemetry_metrics(True)
        mock_dump.assert_not_called()

        # Should save to config when set_as_default=True
        set_telemetry_metrics(True, set_as_default=True)
        mock_dump.assert_called_once()
        config = mock_dump.call_args.args[0]
        assert config == mock_config


def test_save_metrics_config_new_file(mock_file_operations, mock_config):
    """Test saving metrics config to a new file."""
    with patch("pathlib.Path.exists", return_value=False), patch("yaml.safe_dump") as mock_dump:
        _save_metrics_config(True)
        mock_dump.assert_called_once()
        config = mock_dump.call_args.args[0]
        assert config == mock_config


def test_save_metrics_config_existing_file(mock_file_operations):
    """Test saving metrics config to existing file with other settings."""
    existing_config = {"other_setting": "value", "metrics_enabled": False}
    with patch("pathlib.Path.exists", return_value=True), patch("yaml.safe_load", return_value=existing_config), patch(
        "yaml.safe_dump"
    ) as mock_dump:
        _save_metrics_config(True)
        mock_dump.assert_called_once()
        config = mock_dump.call_args.args[0]
        assert config == {"other_setting": "value", "metrics_enabled": True}


def test_class_method_tracking(setup_metrics):
    """Test tracking metrics for a class method"""
    set_telemetry_metrics(True)

    class DummyClass:
        number = 10

        def __init__(self):
            self.instance_number = 1

        @track_usage
        def some_method(self, arg_1, arg_2):
            return self.instance_number + 1 + arg_1 + arg_2

        @classmethod
        @track_usage
        def class_method(cls, x):
            return cls.number + x

        @staticmethod
        @track_usage
        def static_method(x, y):
            return x + y

        @property
        @track_usage
        def computed_value(self):
            return self.instance_number * 2

        @track_usage
        def no_args(self):
            return 42

    # Create instance and test instance method
    dummy = DummyClass()
    result = dummy.some_method(1, arg_2=3)
    assert result == 6
    assert_metric_labels(
        setup_metrics, "test_metrics.test_class_method_tracking.<locals>.DummyClass.some_method", "success"
    )

    # Test class method
    result = DummyClass.class_method(5)
    assert result == 15
    assert_metric_labels(
        setup_metrics, "test_metrics.test_class_method_tracking.<locals>.DummyClass.class_method", "success"
    )

    # Test static method
    result = dummy.static_method(3, 4)
    assert result == 7
    assert_metric_labels(
        setup_metrics, "test_metrics.test_class_method_tracking.<locals>.DummyClass.static_method", "success"
    )

    # Test property
    result = dummy.computed_value
    assert result == 2
    assert_metric_labels(
        setup_metrics, "test_metrics.test_class_method_tracking.<locals>.DummyClass.computed_value", "success"
    )

    # Test no-args method
    result = dummy.no_args()
    assert result == 42
    assert_metric_labels(setup_metrics, "test_metrics.test_class_method_tracking.<locals>.DummyClass.no_args", "success")

    # Test error case
    with pytest.raises(TypeError):
        dummy.some_method("invalid", arg_2=3)
    assert_metric_labels(
        setup_metrics, "test_metrics.test_class_method_tracking.<locals>.DummyClass.some_method", "error"
    )


def test_set_opentelemetry_log_level():
    """Test setting OpenTelemetry log level."""
    # Test valid levels
    for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        set_opentelemetry_log_level(level)

        # Verify log level was set correctly
        for logger_name in logging.root.manager.loggerDict:
            if logger_name.startswith("opentelemetry"):
                assert logging.getLogger(logger_name).level == getattr(logging, level)

    # Test invalid level
    with pytest.raises(ValueError):
        set_opentelemetry_log_level("INVALID_LEVEL")

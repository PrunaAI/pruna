# Copyright 2025 - Pruna AI GmbH. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenTelemetry metrics for tracking function usage in Pruna.

This module provides:
1. A decorator for tracking function calls and errors
2. Direct counter access through OpenTelemetry meters

Usage:
    Using the decorator:
    >>> @track_usage
    ... def my_function():
    ...     pass

    >>> @track_usage("custom_name")
    ... def another_function():
    ...     pass

    Using direct counters:
    >>> from pruna.telemetry.metrics import increment_counter
    >>> increment_counter("my_operation")
    >>> increment_counter("my_operation", success=False)

Configuration:
    Metrics can be enabled/disabled for the current python kernel:
    >>> from pruna.telemetry.metrics import set_telemetry_metrics
    >>> set_telemetry_metrics(True)  # Enable
    >>> set_telemetry_metrics(False)  # Disable

    to activate / deactivate globally:
    >>> set_telemetry_metrics(True, set_as_default=True)
"""

from __future__ import annotations

import functools
import inspect
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

import yaml
from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

CONFIG_FILE = Path(__file__).parent / "config.yaml"

# Load initial configuration
with open(CONFIG_FILE) as config_file_stream:
    CONFIG = yaml.safe_load(config_file_stream)

OTLP_ENDPOINT = CONFIG["otlp_endpoint"]
OTLP_HEADERS = CONFIG.get("otlp_headers", {})
SESSION_ID = str(uuid4())
DEFAULT_LOG_LEVEL = CONFIG["telemetry_log_level"]

# Initialize metrics with basic setup
exporter = OTLPMetricExporter(endpoint=OTLP_ENDPOINT, headers=OTLP_HEADERS)
reader = PeriodicExportingMetricReader(exporter, export_interval_millis=60000)

provider = MeterProvider(metric_readers=[reader])
metrics.set_meter_provider(provider)
meter = metrics.get_meter(__name__)

# Create function usage counter
function_counter = meter.create_counter(
    name="pruna.function.calls",
    description="Number of function calls",
    unit="1",
)

# Set initial metrics enabled state in env var if not already set
if "PRUNA_METRICS_ENABLED" not in os.environ:
    os.environ["PRUNA_METRICS_ENABLED"] = str(CONFIG["metrics_enabled"]).lower()


def is_metrics_enabled() -> bool:
    """
    Check if metrics are enabled.

    Returns
    -------
    bool
        True if metrics are enabled, False otherwise.
    """
    return os.environ.get("PRUNA_METRICS_ENABLED", "false").lower() == "true"


def _save_metrics_config(enabled: bool) -> None:
    """
    Save metrics state to the configuration file.

    Parameters
    ----------
    enabled : bool
        Whether metrics should be enabled or disabled.
    """
    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f)

    config["metrics_enabled"] = enabled

    with open(CONFIG_FILE, "w") as f:
        yaml.safe_dump(config, f)


def set_telemetry_metrics(enabled: bool, set_as_default: bool = False) -> None:
    """
    Enable or disable metrics globally.

    Parameters
    ----------
    enabled : bool
        Whether to enable or disable the metrics.
    set_as_default : bool, optional
        If True, saves the state to the configuration file as the default value.
    """
    enabled = bool(enabled)
    os.environ["PRUNA_METRICS_ENABLED"] = str(enabled).lower()
    if set_as_default:
        _save_metrics_config(enabled)


def increment_counter(function_name: str, success: bool = True, smash_config: Optional[str] = "") -> None:
    """
    Increment the counter for a given function or operation.

    Can be used directly to track operations that aren't functions
    or when the decorator pattern is not suitable.

    Parameters
    ----------
    function_name : str
        Name of the function or operation being tracked.
    success : bool, optional
        Whether the operation was successful (default is True).
    smash_config : str, optional
        A string representation of the smash config used for the operation (default is an empty string).

    Examples
    --------
    Manual operation tracking:
        >>> try:
        ...     increment_counter("manual_operation")
        ... except Exception:
        ...     increment_counter("manual_operation", success=False)
        ...     raise

    Tracking a block of code:
        >>> with contextlib.suppress(Exception):
        ...     try:
        ...         increment_counter("complex_operation")
        ...     except Exception:
        ...         increment_counter("complex_operation", success=False)
        ...         raise
    """
    if is_metrics_enabled():
        function_counter.add(
            1,
            {
                "function": function_name,
                "status": "success" if success else "error",
                "smash_config": smash_config,  # type: ignore
                "session_id": SESSION_ID,
            },
        )


def track_usage(name_or_func: Optional[str | Callable] = None) -> Callable:
    """
    Decorator to track function usage.

    Parameters
    ----------
    name_or_func : str or Callable, optional
        Either a string name for the operation or the function to decorate.
        If None, uses the decorated function's name.

    Returns
    -------
    Callable
        The decorated function that tracks its usage when metrics are enabled.
    """

    def get_full_path(function_or_method_of_class: Callable) -> str:
        module = inspect.getmodule(function_or_method_of_class)
        module_path = module.__name__ if module else ""
        if hasattr(
            function_or_method_of_class, "__qualname__"
        ):  # Handles the path within the module, e.g. part of a class
            return f"{module_path}.{function_or_method_of_class.__qualname__}"
        return f"{module_path}.{function_or_method_of_class.__name__}"

    def decorator(func: Callable) -> Callable:
        function_name = name_or_func if isinstance(name_or_func, str) else get_full_path(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            smash_config = kwargs.get("smash_config")
            smash_config = repr(smash_config) if smash_config is not None else ""
            try:
                result = func(*args, **kwargs)
                increment_counter(function_name, success=True, smash_config=smash_config)
                return result
            except Exception:
                increment_counter(function_name, success=False, smash_config=smash_config)
                raise

        return wrapper

    if callable(name_or_func):
        return decorator(name_or_func)
    return decorator


def set_opentelemetry_log_level(level: str) -> None:
    """
    Set the log level for OpenTelemetry loggers to control error visibility.

    This can be used to suppress error messages when telemetry fails.

    Parameters
    ----------
    level : str
        The log level to set. Must be one of: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.

        - 'DEBUG': Show all messages including detailed debugging information
        - 'INFO': Show informational messages, warnings and errors
        - 'WARNING': Show only warnings and errors (default)
        - 'ERROR': Show only errors
        - 'CRITICAL': Show only critical errors

    Raises
    ------
    ValueError
        If the provided level is not a valid logging level.

    Examples
    --------
    To suppress most error messages:

    >>> set_opentelemetry_log_level('ERROR')

    To show all messages:

    >>> set_opentelemetry_log_level('DEBUG')
    """
    valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    level = level.upper()

    if level not in valid_levels:
        raise ValueError(f"Log level must be one of: {', '.join(valid_levels)}")

    # Configure OpenTelemetry loggers
    logging_level = getattr(logging, level)

    # Set log level for all OpenTelemetry loggers
    for logger_name in logging.root.manager.loggerDict:
        if logger_name.startswith("opentelemetry"):
            logging.getLogger(logger_name).setLevel(logging_level)


set_opentelemetry_log_level(DEFAULT_LOG_LEVEL)

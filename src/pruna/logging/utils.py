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

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Union


@contextmanager
def temporary_log_level(logger_name: str, level: Union[int, str] = "WARNING"):
    """
    Context manager to temporarily change the logging level of a specific logger.

    This is useful for suppressing verbose output from third-party libraries during
    specific operations, such as quantization or model loading.

    Parameters
    ----------
    logger_name : str
        The name of the logger to modify. Can be a full module path like
        "torchao.quantization.quant_api" or a simpler name like "transformers".
    level : Union[int, str]
        The temporary logging level to set. Can be an integer (e.g., logging.WARNING)
        or a string (e.g., "WARNING", "ERROR", "INFO", "DEBUG", "CRITICAL").

    Examples
    --------
    >>> # Suppress INFO messages from torchao during quantization
    >>> with temporary_log_level("torchao.quantization.quant_api", "WARNING"):
    ...     model = quantize_model(model)

    >>> # Suppress DEBUG messages from transformers during model loading
    >>> with temporary_log_level("transformers", logging.INFO):
    ...     model = AutoModel.from_pretrained(model_name)

    >>> # Suppress all messages except critical errors
    >>> with temporary_log_level("some.noisy.library", "CRITICAL"):
    ...     result = noisy_function()
    """
    logger = logging.getLogger() if logger_name is None else logging.getLogger(logger_name)
    original_level = logger.getEffectiveLevel()

    # Convert string level to integer if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(original_level)


@contextmanager
def suppress_logging(logger_name: str, level: Union[int, str] = "WARNING"):
    """
    Context manager to suppress logging messages below a certain level.

    This is a convenience wrapper around temporary_log_level specifically for
    suppressing verbose output.

    Parameters
    ----------
    logger_name : str
        The name of the logger to suppress. Can be a full module path like
        "torchao.quantization.quant_api" or a simpler name like "transformers".
    level : Union[int, str], default="WARNING"
        The minimum logging level to allow. Messages below this level will be
        suppressed. Can be an integer or string.

    Examples
    --------
    >>> # Suppress INFO and DEBUG messages from torchao
    >>> with suppress_logging("torchao.quantization.quant_api"):
    ...     model = quantize_model(model)

    >>> # Suppress everything except errors and critical messages
    >>> with suppress_logging("transformers", "ERROR"):
    ...     model = AutoModel.from_pretrained(model_name)
    """
    with temporary_log_level(logger_name, level):
        yield

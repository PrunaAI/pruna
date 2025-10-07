from typing import Any
import logging
import pytest

from pruna.logging.logger import pruna_logger, set_logging_level


@pytest.mark.cpu
@pytest.mark.parametrize(
    "level, expected",
    [
        ("DEBUG", logging.DEBUG),
        ("INFO", logging.INFO),
        ("WARNING", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("CRITICAL", logging.CRITICAL),
    ],
)
def test_set_logging_level_programmatic(level: str, expected: int) -> None:
    """Test setting the pruna_logger level programmatically."""
    set_logging_level(level)
    assert pruna_logger.level == expected


@pytest.mark.cpu
@pytest.mark.parametrize(
    "env_level, expected",
    [
        ("DEBUG", logging.DEBUG),
        ("WARNING", logging.WARNING),
    ],
)
def test_set_logging_level_env(monkeypatch: Any, env_level: str, expected: int) -> None:
    """Test setting the pruna_logger level via environment variable."""
    monkeypatch.setenv("PRUNA_LOG_LEVEL", env_level)
    set_logging_level()
    assert pruna_logger.level == expected


@pytest.mark.cpu
def test_invalid_logging_level() -> None:
    """Test that invalid logging levels raise ValueError."""
    with pytest.raises(ValueError):
        set_logging_level("INVALID_LEVEL")

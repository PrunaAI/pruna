from typing import Any

# import all fixtures to make them avaliable for pytest
from .fixtures import *  # noqa: F403, F401
from .fixtures import HIGH_RESOURCE_FIXTURES, HIGH_RESOURCE_FIXTURES_CPU


def pytest_configure(config: Any) -> None:
    """Configure the pytest markers."""
    config.addinivalue_line("markers", "cpu: mark test to run on CPU")
    config.addinivalue_line("markers", "cuda: mark test to run only on GPU machines")
    config.addinivalue_line("markers", "distributed: mark test to run only on multi-GPU machines")
    config.addinivalue_line("markers", "high: mark test to run only on large GPUs")
    config.addinivalue_line("markers", "high_cpu: mark test to run only on large CPU systems")
    config.addinivalue_line("markers", "slow: mark test that run rather long")
    config.addinivalue_line("markers", "style: mark test that only check style")


def pytest_collection_modifyitems(session: Any, config: Any, items: list) -> None:
    """Hook that is called after test collection. Automatically adds markers to tests that use high-resource fixtures."""
    for item in items:
        if "model_fixture" in item.fixturenames:
            model_value = item.callspec.params["model_fixture"]
            # Convert model_value to a string or a hashable identifier
            if model_value in HIGH_RESOURCE_FIXTURES:
                item.add_marker("high")
            if model_value in HIGH_RESOURCE_FIXTURES_CPU:
                item.add_marker("high_cpu")

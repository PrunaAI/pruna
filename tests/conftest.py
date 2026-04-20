from typing import Any

import pytest

# import all fixtures to make them avaliable for pytest
from .fixtures import *  # noqa: F403, F401

HARDWARE_MARKS = {"cpu", "cuda", "multi_gpu"}


def pytest_configure(config: Any) -> None:
    """Configure the pytest markers."""
    # Hardware marks
    config.addinivalue_line("markers", "cpu: mark test to run on CPU")
    config.addinivalue_line("markers", "cuda: mark test to run only on GPU machines")
    config.addinivalue_line("markers", "multi_gpu: mark test to run only on multi-GPU machines")
    config.addinivalue_line("markers", "high_gpu: mark test to run only on large GPUs")  # e.g. H100
    # Dependency marks for external dependencies
    config.addinivalue_line("markers", "requires_gptq: mark test that needs pruna[gptq]")
    config.addinivalue_line("markers", "requires_awq: mark test that needs pruna[awq]")
    config.addinivalue_line("markers", "requires_stable_fast: mark test that needs pruna[stable-fast]")
    config.addinivalue_line("markers", "requires_vllm: mark test that needs pruna[vllm]")
    config.addinivalue_line("markers", "requires_intel: mark test that needs pruna[intel]")
    config.addinivalue_line("markers", "requires_lmharness: mark test that needs pruna[lmharness]")
    config.addinivalue_line("markers", "requires_whisper: mark test that needs pruna[whisper]")
    config.addinivalue_line("markers", "requires_rapidata: mark test that needs pruna[rapidata]")
    # Category marks
    config.addinivalue_line("markers", "slow: mark test that run rather long")
    config.addinivalue_line("markers", "style: mark test that only check style")
    config.addinivalue_line("markers", "integration: mark test that is an integration test")


def pytest_collection_modifyitems(session: Any, config: Any, items: list) -> None:
    """Hook that is called after test collection."""
    selected = []
    deselected = []
    for item in items:
        # Auto-tag unmarked tests as CPU
        if not any(mark in item.keywords for mark in HARDWARE_MARKS):
            item.add_marker(pytest.mark.cpu)
        # device_parametrized generates cpu/cuda/accelerate variants for every
        # algorithm test, even when the algorithm's runs_on excludes that device.
        # The incompatible variants get collected by the CI (e.g. -m "cpu") and
        # skip at runtime after expensive fixture setup. Deselecting them here
        # avoids that wasted overhead.
        if hasattr(item, "callspec") and "algorithm_tester" in item.callspec.params:
            tester = item.callspec.params["algorithm_tester"]
            device = item.callspec.params.get("device")
            if device and device not in tester.compatible_devices():
                deselected.append(item)
                continue
        selected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected

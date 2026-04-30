from typing import Any

import pytest

# import all fixtures to make them avaliable for pytest
from .fixtures import *  # noqa: F403, F401

DEVICE_MARKS = {
    "cpu": "mark test to run on CPU",
    "cuda": "mark test to run only on GPU machines",
    "multi_gpu": "mark test to run only on multi-GPU machines",
}
EXTRA_MARKS = {
    "requires_gptq": "mark test that needs pruna[gptq]",
    "requires_awq": "mark test that needs pruna[awq]",
    "requires_stable_fast": "mark test that needs pruna[stable-fast]",
    "requires_vllm": "mark test that needs pruna[vllm]",
    "requires_intel": "mark test that needs pruna[intel]",
    "requires_lmharness": "mark test that needs pruna[lmharness]",
    "requires_whisper": "mark test that needs pruna[whisper]",
    "requires_upscale": "mark test that needs pruna[upscale]",
    "requires_rapidata": "mark test that needs pruna[rapidata]",
}


def pytest_configure(config: Any) -> None:
    """Configure the pytest markers."""
    # Device marks
    for mark, description in DEVICE_MARKS.items():
        config.addinivalue_line("markers", f"{mark}: {description}")
    config.addinivalue_line("markers", "high_gpu: mark test to run only on large GPUs")
    # Dependency marks for external dependencies
    for mark, description in EXTRA_MARKS.items():
        config.addinivalue_line("markers", f"{mark}: {description}")
    config.addinivalue_line("markers", "no_extras: mark test that runs without optional dependency extras")
    # Category marks
    config.addinivalue_line("markers", "slow: mark test that run rather long")
    config.addinivalue_line("markers", "style: mark test that only check style")
    config.addinivalue_line("markers", "integration: mark test that is an integration test")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(session: Any, config: Any, items: list) -> None:
    """Hook that is called after test collection."""
    selected = []
    deselected = []
    for item in items:
        # Auto-tag unmarked tests as CPU
        if not any(mark in item.keywords for mark in DEVICE_MARKS):
            item.add_marker(pytest.mark.cpu)
        # Auto-tag tests that do not require optional dependency extras. This
        # keeps the default CI selection positive as new extras are added.
        if not any(mark.name in EXTRA_MARKS for mark in item.iter_markers()):
            item.add_marker(pytest.mark.no_extras)
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

import importlib.util
from typing import Any

import pytest
import torch

# import all fixtures to make them avaliable for pytest
from .fixtures import *  # noqa: F403, F401

EXTRA_SKIP_MAP = {
    "requires_gptq": ("gptqmodel", "pruna[gptq]"),
    "requires_awq": ("llmcompressor", "pruna[awq]"),
    "requires_stable_fast": ("sfast", "pruna[stable-fast]"),
    "requires_vllm": ("vllm", "pruna[vllm]"),
    "requires_intel": ("intel_extension_for_pytorch", "pruna[intel]"),
    "requires_lmharness": ("lm_eval", "pruna[lmharness]"),
    "requires_whisper": ("whisper", "pruna[whisper]"),

}


def pytest_configure(config: Any) -> None:
    """Configure the pytest markers."""
    # Hardware marks
    config.addinivalue_line("markers", "cpu: mark test to run on CPU")
    config.addinivalue_line("markers", "cuda: mark test to run only on GPU machines")
    config.addinivalue_line("markers", "multi_gpu: mark test to run only on multi-GPU machines")
    config.addinivalue_line("markers", "high_vram: mark test to run only on large GPUs")  # e.g. H100
    # Dependency marks for external dependencies
    config.addinivalue_line("markers", "requires_gptq: mark test that needs pruna[gptq]")
    config.addinivalue_line("markers", "requires_awq: mark test that needs pruna[awq]")
    config.addinivalue_line("markers", "requires_stable_fast: mark test that needs pruna[stable-fast]")
    config.addinivalue_line("markers", "requires_vllm: mark test that needs pruna[vllm]")
    config.addinivalue_line("markers", "requires_intel: mark test that needs pruna[intel]")
    config.addinivalue_line("markers", "requires_lmharness: mark test that needs pruna[lmharness]")
    config.addinivalue_line("markers", "requires_whisper: mark test that needs pruna[whisper]")
    # Category marks
    config.addinivalue_line("markers", "slow: mark test that run rather long")
    config.addinivalue_line("markers", "style: mark test that only check style")
    config.addinivalue_line("markers", "integration: mark test that is an integration test")


def _has_multi_gpu() -> bool:
    return torch.cuda.device_count() > 1


def _has_gpu() -> bool:
    return torch.cuda.is_available()


def pytest_collection_modifyitems(session: Any, config: Any, items: list) -> None:
    """Hook that is called after test collection."""
    cuda_skip = not _has_gpu()
    multi_gpu_skip = not _has_multi_gpu()

    # Build skip markers for missing optional dependencies
    extra_skips: dict[str, pytest.Mark] = {}
    for mark_name, (pkg, install_hint) in EXTRA_SKIP_MAP.items():
        if importlib.util.find_spec(pkg) is None:
            extra_skips[mark_name] = pytest.mark.skip(
                reason=f"{pkg} not installed. Install with: pip install '{install_hint}'"
            )

    for item in items:
        # Skip CUDA tests when no GPU
        if cuda_skip and "cuda" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="CUDA not available"))

        # Skip multi-GPU tests when < 2 GPUs
        if multi_gpu_skip and "multi_gpu" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="Requires multiple GPUs"))

        # Skip tests whose optional dependencies are not installed
        for mark_name, skip_marker in extra_skips.items():
            if mark_name in item.keywords:
                item.add_marker(skip_marker)

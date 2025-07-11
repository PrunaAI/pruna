from typing import Any

import pytest

from pruna import SmashConfig

from ..common import (
    device_parametrized,
    get_instances_from_module,
    run_full_integration,
)
from . import testers


@device_parametrized
@pytest.mark.parametrize(
    "algorithm_tester, model_fixture",
    get_instances_from_module(testers),
    indirect=["model_fixture"],
)
def test_full_integration(algorithm_tester: Any, device: str, model_fixture: tuple[Any, SmashConfig]) -> None:
    """Test the full integration of the algorithm."""
    run_full_integration(algorithm_tester, device, model_fixture)

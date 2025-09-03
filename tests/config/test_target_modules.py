from __future__ import annotations

import pytest
import re

from pruna import SmashConfig, smash
from pruna.config.target_modules import TARGET_MODULES_TYPE


@pytest.mark.cuda
@pytest.mark.parametrize(
    "model_fixture, algorithm_group, algorithm, target_modules, expected_number_of_targeted_modules",
    [
        ("flux_tiny_random", "quantizer", "quanto", None, 28),
        ("flux_tiny_random", "quantizer", "quanto", {"include": ["transformer*"]}, 28),
        ("flux_tiny_random", "quantizer", "quanto", {"include": ["transformer*"], "exclude": ["*norm*"]}, 24),
    ],
    indirect=["model_fixture"],
)
def test_target_modules(
    model_fixture: tuple, algorithm_group: str, algorithm: str, target_modules: TARGET_MODULES_TYPE | None, expected_number_of_targeted_modules: int
) -> None:
    model, smash_config = model_fixture
    smash_config[algorithm_group] = algorithm
    smash_config[f"{algorithm}_target_modules"] = target_modules
    smashed_model = smash(model, smash_config)

    num_targeted_modules = sum(
        1 for module in smashed_model.get_nn_modules().values()
        for submodule in module.modules()
        if submodule.__class__.__name__ == "QLinear"
    )
    assert num_targeted_modules == expected_number_of_targeted_modules

@pytest.mark.cpu
@pytest.mark.parametrize("target_modules", [
    {"include": ["test_pattern*", "other"]},
    {"include": ["this*"], "exclude": ["that"]},
])
def test_target_modules_format_accept(target_modules: dict[str, list[str]]):
    smash_config = SmashConfig()
    smash_config["quantizer"] = "quanto"
    smash_config["quanto_target_modules"] = target_modules
    assert smash_config['quanto_target_modules'] == target_modules

@pytest.mark.cpu
@pytest.mark.parametrize("target_modules, expected_error", [
    (["transformer*"], TypeError),  # not a dict
    ({"what_are_the_keywords": ["transformer"]}, ValueError),  # keys should be "include" or "exclude"
    ({"include": ["transformer*"], "exclude": 1}, TypeError),  # "exclude" value is not a list
    ({"include": ["transformer*"], "exclude": [1, "transformer*"]}, TypeError),  # lists can't contain anything but strings
])
def test_target_modules_format_reject(target_modules: TARGET_MODULES_TYPE, expected_error: type):
    smash_config = SmashConfig()
    smash_config["quantizer"] = "quanto"
    with pytest.raises(expected_error):
        smash_config["quanto_target_modules"] = target_modules

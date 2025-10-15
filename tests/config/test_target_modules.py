from __future__ import annotations

import pytest
import re

from pruna import SmashConfig, smash
from pruna.config.target_modules import TARGET_MODULES_TYPE


@pytest.mark.cuda
@pytest.mark.parametrize(
    "model_fixture, algorithm, target_modules, expected_number_of_targeted_modules",
    [
        ("flux_tiny_random", "quanto", None, 28),
        ("flux_tiny_random", "quanto", {"include": ["transformer*"]}, 28),
        ("flux_tiny_random", "quanto", {"include": ["transformer*"], "exclude": ["*norm*"]}, 24),
        ("flux_tiny_random", "diffusers_int8", None, 28),
        ("flux_tiny_random", "diffusers_int8", {"include": ["transformer*"], "exclude": ["*norm*"]}, 24),
        ("llama_3_tiny_random", "llm_int8", None, 14),
        ("llama_3_tiny_random_as_pipeline", "llm_int8", {"include": ["model.*"], "exclude": ["*v_proj*"]}, 13),
    ],
    indirect=["model_fixture"],
)
def test_target_modules(
    model_fixture: tuple, algorithm: str, target_modules: TARGET_MODULES_TYPE | None, expected_number_of_targeted_modules: int
) -> None:
    model, smash_config = model_fixture
    smash_config.add(algorithm)
    smash_config[f"{algorithm}_target_modules"] = target_modules
    smashed_model = smash(model, smash_config)

    is_algorithm_applied = {
        "quanto": lambda module: module.__class__.__name__ == "QLinear",
        "llm_int8": lambda module: "bitsandbytes" in str(type(module)).lower(),
        "diffusers_int8": lambda module: "bitsandbytes" in str(type(module)).lower(),
    }

    num_targeted_modules = sum(
        1 for module in smashed_model.get_nn_modules().values()
        for submodule in module.modules()
        if is_algorithm_applied[algorithm](submodule)
    )
    assert num_targeted_modules == expected_number_of_targeted_modules

@pytest.mark.cpu
@pytest.mark.parametrize("target_modules", [
    {"include": ["test_pattern*", "other"]},
    {"include": ["this*"], "exclude": ["that"]},
])
def test_target_modules_format_accept(target_modules: dict[str, list[str]]):
    smash_config = SmashConfig()
    smash_config.add("quanto")
    smash_config.add({"quanto_target_modules": target_modules})
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
    smash_config.add("quanto")
    with pytest.raises(expected_error):
        smash_config.add({"quanto_target_modules": target_modules})

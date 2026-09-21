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

from unittest.mock import patch

import pytest

from pruna.config.smash_config import SmashConfig
from pruna.engine.load import resmash


@pytest.mark.cpu
def test_save_load_preserves_custom_algorithm_order(tmp_path):
    """Test that a custom algorithm order survives SmashConfig serialization."""
    expected_order = ["torch_structured", "torch_compile"]
    config = SmashConfig(expected_order, device="cpu")
    config.overwrite_algorithm_order(expected_order)

    config.save_to_json(tmp_path)
    loaded_config = SmashConfig(device="cpu")
    loaded_config.load_from_json(tmp_path)

    assert loaded_config._algorithm_order == expected_order


@pytest.mark.cpu
def test_resmash_filters_custom_order_to_reapplied_algorithms():
    """Test that resmash retains custom order only for algorithms being reapplied."""
    config = SmashConfig(["torch_structured", "torch_compile"], device="cpu")
    config.overwrite_algorithm_order(["torch_structured", "torch_compile"])
    config.reapply_after_load = {"torch_structured": False, "torch_compile": True}
    observed = {}

    def fake_smash(model, smash_config):
        observed["order"] = smash_config._algorithm_order
        observed["active_algorithms"] = smash_config.get_active_algorithms()
        return model

    with patch("pruna.smash.smash", side_effect=fake_smash):
        model = object()
        assert resmash(model, config) is model

    assert observed["active_algorithms"] == ["torch_compile"]
    assert observed["order"] == ["torch_compile"]


@pytest.mark.cpu
def test_overwrite_algorithm_order_rejects_duplicates():
    """Test that a custom algorithm order contains each active algorithm exactly once."""
    config = SmashConfig(["torch_structured", "torch_compile"], device="cpu")

    with pytest.raises(ValueError, match="All active algorithms must be contained in the given algorithm order."):
        config.overwrite_algorithm_order(["torch_structured", "torch_compile", "torch_structured"])

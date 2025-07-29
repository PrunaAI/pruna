import json
from pathlib import Path
from unittest.mock import patch
from importlib_metadata import version

import pytest

from pruna.config.smash_config import SmashConfig
from pruna.logging.logger import pruna_logger


def test_version_mismatch_warning_and_missing_info(tmp_path):
    config = SmashConfig()
    config.save_to_json(tmp_path)
    config_file = tmp_path / "smash_config.json"

    # Simulate version mismatch
    with open(config_file, "r") as f:
        saved = json.load(f)
    saved["_pruna_version"] = "0.1.0"
    with open(config_file, "w") as f:
        json.dump(saved, f)
    with patch.object(pruna_logger, "warning") as mock_warn:
        SmashConfig().load_from_json(tmp_path)
        mock_warn.assert_called()
        # Check all warning calls for the version mismatch message
        found = False
        for call in mock_warn.call_args_list:
            msg = call[0][0]
            if "Version mismatch detected." in msg and "0.1.0" in msg:
                found = True
                break
        assert found, "Version mismatch warning not found in logger calls."

    # Simulate missing version info
    with open(config_file, "r") as f:
        saved = json.load(f)
    saved.pop("_pruna_version", None)
    with open(config_file, "w") as f:
        json.dump(saved, f)
    with patch.object(pruna_logger, "info") as mock_info:
        SmashConfig().load_from_json(tmp_path)
        mock_info.assert_called()
        assert "No pruna version information found" in mock_info.call_args[0][0]


def test_version_tracking_and_equality(tmp_path):
    config = SmashConfig()
    current_version = version("pruna")
    assert config._pruna_version == current_version

    config.save_to_json(tmp_path)
    config_file = tmp_path / "smash_config.json"
    with open(config_file) as f:
        saved = json.load(f)
    assert saved["_pruna_version"] == current_version

    loaded = SmashConfig()
    loaded.load_from_json(tmp_path)
    assert loaded._pruna_version == current_version

    # Equality and flush
    c1, c2 = SmashConfig(), SmashConfig()
    assert c1 == c2
    c1["cacher"] = "deepcache"
    c1.flush_configuration()
    assert c1._pruna_version == current_version

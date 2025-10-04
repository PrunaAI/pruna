from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from pruna.engine.civitai import download_civitai_artifact, is_civitai_source, load_pruna_model_from_civitai


def _fake_model_response(model_id: int = 123) -> dict[str, Any]:
    return {
        "id": model_id,
        "modelVersions": [
            {
                "id": 1,
                "files": [
                    {
                        "name": "pipeline.zip",
                        "format": "Diffusers",
                        "downloadUrl": "https://example.com/pipeline.zip",
                    }
                ],
            }
        ],
    }


def test_is_civitai_source():
    assert is_civitai_source("civitai:123")
    assert is_civitai_source("civitai:some-slug")
    assert not is_civitai_source("hf:org/model")


@patch("pruna.engine.civitai.requests.get")
@patch("pruna.engine.civitai._get_json")
def test_download_civitai_artifact_extracts_zip(mock_get_json: MagicMock, mock_req_get: MagicMock, tmp_path: Path):
    mock_get_json.return_value = _fake_model_response()

    # Fake streaming zip file containing model_index.json
    import io
    import zipfile

    bytes_io = io.BytesIO()
    with zipfile.ZipFile(bytes_io, "w") as zf:
        zf.writestr("model_index.json", json.dumps({"_class_name": "StableDiffusionPipeline"}))
    bytes_io.seek(0)

    resp = MagicMock()
    resp.iter_content = lambda chunk_size: [bytes_io.getvalue()]
    resp.__enter__.return_value = resp
    resp.raise_for_status = lambda: None
    mock_req_get.return_value = resp

    out_dir = download_civitai_artifact("123", cache_dir=tmp_path)
    assert (out_dir / "model_index.json").exists()


@patch("pruna.engine.civitai.download_civitai_artifact")
@patch("pruna.engine.civitai.load_diffusers_model")
def test_load_pruna_model_from_civitai_prefers_diffusers(mock_load_diffusers: MagicMock, mock_download: MagicMock, tmp_path: Path):
    local = tmp_path / "m"
    local.mkdir()
    (local / "model_index.json").write_text(json.dumps({"_class_name": "StableDiffusionPipeline"}))
    mock_download.return_value = local
    mock_load_diffusers.return_value = MagicMock()

    model, smash_cfg = load_pruna_model_from_civitai("civitai:123", cache_dir=tmp_path)
    assert model is not None
    assert smash_cfg.load_fns == ["diffusers"]




"""
Utilities to download and load models from Civitai.

This module provides a minimal client to resolve a Civitai model by id or name,
download the appropriate artifact (prefer Diffusers pipelines), unpack it into a
local cache directory, and then delegate loading to existing Pruna loaders.
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

from pruna import SmashConfig
from pruna.engine.load import load_diffusers_model, load_transformers_model
from pruna.logging.logger import pruna_logger


API_BASE_URL = "https://civitai.com/api/v1"


def is_civitai_source(source: Optional[str]) -> bool:
    """Return True if the source string denotes a civitai resource."""
    return isinstance(source, str) and source.lower().startswith("civitai:")


def _auth_headers() -> Dict[str, str]:
    api_key = os.environ.get("CIVITAI_API_KEY")
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    headers = {"Accept": "application/json"}
    headers.update(_auth_headers())
    response = requests.get(url, headers=headers, params=params, timeout=60)
    response.raise_for_status()
    return response.json()  # type: ignore[return-value]


def _pick_version(model_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pick a model version with a file that likely represents the full model.

    Preference order for file selection within a version:
    - file where `format` is "Diffusers" (usually a zip)
    - a file name ending with `.zip`
    - otherwise first file
    """
    versions = model_obj.get("modelVersions", [])
    if not versions:
        raise ValueError("Civitai model has no versions available")

    # Prefer latest (assume list is newest-first; otherwise sort by id)
    versions_sorted = sorted(versions, key=lambda v: v.get("id", 0), reverse=True)

    for version in versions_sorted:
        files = version.get("files", [])
        if not files:
            continue
        # Prefer Diffusers format
        diffusers_files = [f for f in files if str(f.get("format", "")).lower() == "diffusers"]
        if diffusers_files:
            version = dict(version)
            version["_picked_file"] = diffusers_files[0]
            return version
        # Fallback: any zip
        zip_files = [f for f in files if str(f.get("name", "")).lower().endswith(".zip")]
        if zip_files:
            version = dict(version)
            version["_picked_file"] = zip_files[0]
            return version
        # Otherwise take first file
        version = dict(version)
        version["_picked_file"] = files[0]
        return version

    raise ValueError("No downloadable files found for any Civitai version")


def _resolve_model(identifier: str) -> Dict[str, Any]:
    identifier = identifier.strip()
    # numeric id
    if identifier.isdigit():
        return _get_json(f"{API_BASE_URL}/models/{identifier}")
    # try slug/name search
    results = _get_json(f"{API_BASE_URL}/models", params={"limit": 1, "query": identifier})
    items = results.get("items", [])
    if not items:
        raise ValueError(f"No Civitai model found for query: {identifier}")
    return items[0]


def _download_to(path: Path, url: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = _auth_headers()
    with requests.get(url, headers=headers, stream=True, timeout=600) as r:  # 10 min timeout window
        r.raise_for_status()
        # Some files can be large; stream to disk
        with path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return path


def _extract_zip(archive_path: Path, target_dir: Path) -> None:
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(target_dir)


def _ensure_cache_dir(cache_dir: Optional[str | Path]) -> Path:
    if cache_dir is None:
        # default to user cache under ~/.cache/pruna/civitai
        base = Path.home() / ".cache" / "pruna" / "civitai"
    else:
        base = Path(cache_dir) / "civitai"
    base.mkdir(parents=True, exist_ok=True)
    return base


def download_civitai_artifact(identifier: str, cache_dir: Optional[str | Path] = None) -> Path:
    """
    Download a civitai model artifact and return the local directory path with files.

    The function prefers Diffusers-format artifacts and extracts them if zipped.
    """
    model_obj = _resolve_model(identifier)
    version = _pick_version(model_obj)
    picked = version["_picked_file"]

    download_url = picked.get("downloadUrl")
    if not download_url:
        raise ValueError("Civitai did not provide a downloadUrl for the selected file")

    version_id = str(version.get("id", "unknown"))
    dest_root = _ensure_cache_dir(cache_dir) / str(model_obj.get("id", "unknown")) / version_id
    completed_flag = dest_root / ".completed"

    # If already populated, reuse
    if completed_flag.exists():
        return dest_root

    dest_root.mkdir(parents=True, exist_ok=True)

    # Decide destination
    file_name = str(picked.get("name") or f"civitai_{version_id}")
    dest_file = dest_root / file_name

    pruna_logger.info(f"Downloading Civitai artifact: {download_url}")
    _download_to(dest_file, download_url)

    # If zip, extract into dest_root
    if dest_file.suffix.lower() == ".zip":
        _extract_zip(dest_file, dest_root)
        try:
            dest_file.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass

    # mark complete
    completed_flag.write_text("ok")
    return dest_root


def load_pruna_model_from_civitai(source: str, *, cache_dir: Optional[str | Path] = None, **kwargs: Any) -> Tuple[Any, SmashConfig]:
    """
    Resolve and download a model from Civitai and load it through existing loaders.

    The `source` must be in the form `civitai:<id-or-name>`.
    """
    identifier = source.split(":", 1)[1]
    local_dir = download_civitai_artifact(identifier, cache_dir)

    # Heuristically decide loader based on present files
    smash_config = SmashConfig()
    if (local_dir / "model_index.json").exists():
        model = load_diffusers_model(local_dir, smash_config, **kwargs)
        # prepare save/load functions for subsequent saves
        smash_config.load_fns = ["diffusers"]
    elif (local_dir / "config.json").exists():
        model = load_transformers_model(local_dir, smash_config, **kwargs)
        smash_config.load_fns = ["transformers"]
    else:
        # Unknown layout. Surface a helpful error.
        raise FileNotFoundError(
            "Downloaded Civitai artifact does not contain a recognizable model layout (missing model_index.json or config.json)."
        )

    return model, smash_config




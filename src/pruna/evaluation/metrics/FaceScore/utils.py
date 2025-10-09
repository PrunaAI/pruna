

# Copyright (c) 2025 PrunaAI. All rights reserved.
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

from __future__ import annotations

from pathlib import Path
from typing import Any
import torch
from ImageReward import ImageReward

_CACHE_DIR = Path.home() / ".cache" / "FaceScore"
LOCAL_MODEL_PATH = _CACHE_DIR / "FS_model.pt"
LOCAL_CONFIG_PATH = _CACHE_DIR / "med_config.json"

def available_models() -> list[str]:
    """Return the names of available FS models (local only)."""
    return ["FaceScore"]

def load(
    name: str = "FaceScore",
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    med_config: str | None = None,
) -> ImageReward:
    """
    Load a FaceScore model from local files only.

    Args:
        name: Model name or path.
        device: Device to load the model on.
        med_config: Path to config file.

    Returns:
        ImageReward: The loaded FaceScore model.
    """
    if name == "FaceScore":
        model_path = LOCAL_MODEL_PATH
    elif Path(name).is_file():
        model_path = Path(name)
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    if not model_path.is_file():
        raise FileNotFoundError(
            f"FaceScore model weights not found at {model_path}. Please download FS_model.pt and place it there."
        )

    state_dict = torch.load(model_path, map_location="cpu")

    if med_config is None:
        med_config = LOCAL_CONFIG_PATH
    med_config_path = Path(med_config)
    if not med_config_path.is_file():
        raise FileNotFoundError(
            f"FaceScore config not found at {med_config_path}. Please download med_config.json and place it there."
        )

    model = ImageReward(device=device, med_config=str(med_config_path)).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


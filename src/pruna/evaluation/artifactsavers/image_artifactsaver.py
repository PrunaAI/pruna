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

from __future__ import annotations

import secrets
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from pruna.evaluation.artifactsavers.artifactsaver import ArtifactSaver
    
class ImageArtifactSaver(ArtifactSaver):
    """
    Save image artifacts.

    Parameters
    ----------
    root: Path | str | None = None
        The root directory to save the artifacts.
    export_format: str | None = "png"
        The format to save the artifacts (e.g. "png", "jpg", "jpeg", "webp").
    """

    export_format: str | None
    root: Path | str | None

    def __init__(self, root: Path | str | None = None, export_format: str | None = "png") -> None:
        self.root = Path(root) if root else Path.cwd()
        (self.root / "canonical").mkdir(parents=True, exist_ok=True)
        self.export_format = export_format if export_format else "png"
        if self.export_format not in ["png", "jpg", "jpeg", "webp"]:
            raise ValueError(f"Invalid format: {self.export_format}. Valid formats are: png, jpg, jpeg, webp.")

    def save_artifact(self, data: Any, saving_kwargs: dict = dict()) -> Path:
        """
        Save the image artifact.

        Parameters
        ----------
        data: Any
            The data to save.
        saving_kwargs: dict
            The additional kwargs to pass to the saving utility function.

        Returns
        -------
        Path
            The path to the saved artifact.
        """
        canonical_filename = f"{int(time.time())}_{secrets.token_hex(4)}.{self.export_format}"
        canonical_path = Path(str(self.root)) / "canonical" / canonical_filename

        # We save the image as a PIL Image, so we need to convert the data to a PIL Image.
        # Usually, the data is already a PIL.Image, so we don't need to convert it.
        if isinstance(data, torch.Tensor):
            data = np.transpose(data.cpu().numpy(), (1, 2, 0))
            data = np.clip(data * 255, 0, 255).astype(np.uint8)
        if isinstance(data, np.ndarray):
            data = Image.fromarray(data.astype(np.uint8))
        # Now data must be a PIL Image
        if not isinstance(data, Image.Image):
            raise ValueError("Model outputs must be torch.Tensor, numpy.ndarray, or PIL.Image.")

        # Save the image (export format is determined by the file extension)
        data.save(canonical_path, **saving_kwargs.copy())

        return canonical_path


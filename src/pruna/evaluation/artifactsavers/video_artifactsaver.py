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
from diffusers.utils import export_to_gif, export_to_video
from PIL import Image

from pruna.evaluation.artifactsavers.artifactsaver import ArtifactSaver


class VideoArtifactSaver(ArtifactSaver):
    """
    Save video artifacts.

    Parameters
    ----------
    root: Path | str | None = None
        The root directory to save the artifacts.
    export_format: str | None = "mp4"
        The format to save the artifacts.
    """

    export_format: str | None
    root: Path | str | None

    def __init__(self, root: Path | str | None = None, export_format: str | None = "mp4") -> None:
        self.root = Path(root) if root else Path.cwd()
        (self.root / "canonical").mkdir(parents=True, exist_ok=True)
        self.export_format = export_format if export_format else "mp4"

    def save_artifact(self, data: Any, saving_kwargs: dict = dict()) -> Path:
        """
        Save the video artifact.

        Parameters
        ----------
        data: Any
            The data to save.

        Returns
        -------
        Path
            The path to the saved artifact.
        """
        canonical_filename = f"{int(time.time())}_{secrets.token_hex(4)}.{self.export_format}"
        canonical_path = Path(str(self.root)) / "canonical" / canonical_filename

        #  all diffusers saving utility functions accept a list of PIL.Images, so we convert to PIL to be safe.
        if isinstance(data, torch.Tensor):
            data = np.transpose(data.cpu().numpy(), (0, 2, 3, 1))
            data = np.clip(data * 255, 0, 255).astype(np.uint8)
        if isinstance(data, np.ndarray):
            data = [Image.fromarray(frame.astype(np.uint8)) for frame in data]

        if self.export_format == "mp4":
            export_to_video(data, str(canonical_path), **saving_kwargs.copy())
        elif self.export_format == "gif":
            export_to_gif(data, str(canonical_path), **saving_kwargs.copy())
        else:
            raise ValueError(f"Invalid format: {self.export_format}")
        return canonical_path

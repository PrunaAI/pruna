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

from pathlib import Path

from pruna.evaluation.artifactsavers.artifactsaver import ArtifactSaver
from pruna.evaluation.artifactsavers.video_artifactsaver import VideoArtifactSaver

from pruna.evaluation.artifactsavers.image_artifactsaver import ImageArtifactSaver


def assign_artifact_saver(
    modality: str, root: Path | str | None = None, export_format: str | None = None
) -> ArtifactSaver:
    """
    Assign the appropriate artifact saver based on the modality.

    Parameters
    ----------
    modality: str
        The modality of the data.
    root: str
        The root directory to save the artifacts.
    export_format: str
        The format to save the artifacts.

    Returns
    -------
    ArtifactSaver
        The appropriate artifact saver.
    """
    if modality == "video":
        return VideoArtifactSaver(root=root, export_format=export_format)
    if modality == "image":
        return ImageArtifactSaver(root=root, export_format=export_format)
    else:
        raise ValueError(f"Modality {modality} is not supported")

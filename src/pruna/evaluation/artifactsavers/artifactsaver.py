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

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ArtifactSaver(ABC):
    """
    Abstract class for artifact savers.

    The artifact saver is responsible for saving the inference outputs during evaluation.

    There needs to be a subclass for each metric modality (e.g. video, image, text, etc.).

    Parameters
    ----------
    export_format: str | None
        The format to export the artifacts in.
    root: Path | str | None
        The root directory to save the artifacts in.
    """

    export_format: str | None = None
    root: Path | str | None = None

    @abstractmethod
    def save_artifact(self, data: Any) -> Path:
        """
        Implement this method to save the artifact.

        Parameters
        ----------
        data: Any
            The data to save.

        Returns
        -------
        Path
            The full path to the saved artifact.
        """
        pass

    def create_alias(self, source_path: Path | str, filename: str) -> Path:
        """
        Create an alias for the artifact.

        The evaluation agent will save the inference outputs with a canonical file
        formatting style that makes sense for the general case.

        If your metric requires a different file naming convention for evaluation,
        you can use this method to create an alias for the artifact.

        This way we prevent duplicate artifacts from being saved and save storage space.

        By default, the alias will be created as a hardlink to the source artifact.
        If the hardlink fails, a symlink will be created.

        Parameters
        ----------
        source_path : Path | str
            The path to the source artifact.
        filename : str
            The filename to create the alias for.

        Returns
        -------
        Path
            The full path to the alias.
        """
        alias = Path(str(self.root)) / f"{filename}.{self.export_format}"
        alias.parent.mkdir(parents=True, exist_ok=True)
        try:
            if alias.exists():
                alias.unlink()
            alias.hardlink_to(source_path)
        except Exception:
            try:
                if alias.exists():
                    alias.unlink()
                alias.symlink_to(source_path)
            except Exception as e:
                raise e
        return alias

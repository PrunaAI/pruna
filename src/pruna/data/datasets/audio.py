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

import os
from pathlib import Path
from typing import List, Tuple

from datasets import Dataset, load_dataset

from pruna.logging.logger import pruna_logger


def setup_commonvoice_dataset() -> List[Dataset]:
    """
    Setup the Common Voice dataset.

    License: CC0-1.0

    Returns
    -------
    List[Dataset]
        The Common Voice dataset.
    """
    return load_dataset(
        "mozilla-foundation/common_voice_1_0",
        "fr",
        revision="refs/pr/5",
        split=["train", "validation", "test"],
        trust_remote_code=True,
    )


def setup_podcast_dataset() -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the AI Podcast dataset.

    License: unspecified

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The AI Podcast dataset.
    """
    return _download_audio_and_select_sample("sam_altman_lex_podcast_367.flac")


def setup_mini_presentation_audio_dataset() -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the Mini Audio dataset.

    License: unspecified

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The AI Podcast dataset.
    """
    return _download_audio_and_select_sample("4469669-10.mp3")


def _download_audio_and_select_sample(file_id: str) -> Tuple[Dataset, Dataset, Dataset]:
    load_dataset("reach-vb/random-audios", trust_remote_code=True)
    cache_path = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if cache_path is None:
        cache_path = str(Path.home() / ".cache" / "huggingface" / "hub")

    dataset_path = Path(cache_path) / "datasets--reach-vb--random-audios"
    path_to_podcast_file = str(list(dataset_path.rglob(file_id))[0])

    ds = Dataset.from_dict({"audio": [{"path": path_to_podcast_file}], "sentence": [""]})
    pruna_logger.info(
        "The AI Podcast dataset only consists of one sample, returning same data for training, validation and testing."
    )
    return ds, ds, ds

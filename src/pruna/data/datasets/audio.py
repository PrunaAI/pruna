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

from pathlib import Path
from typing import Tuple

from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download

from pruna.data.utils import split_train_into_train_val_test
from pruna.logging.logger import pruna_logger


def setup_librispeech_dataset(seed: int) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the LibriSpeech dataset.

    License: MIT

    Parameters
    ----------
    seed : int
        The seed to use for the split.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The LibriSpeech dataset with explicit audio paths.
    """
    ds = load_dataset("argmaxinc/librispeech-200", split="train", streaming=False)
    ds = ds.map(lambda batch: {"sentence": ""})

    ds_train, ds_val, ds_test = split_train_into_train_val_test(ds, seed)
    return ds_train, ds_val, ds_test


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
    dataset_path = snapshot_download("reach-vb/random-audios", repo_type="dataset")
    path_to_podcast_file = str(Path(dataset_path) / file_id)
    ds = Dataset.from_dict({"audio": [{"path": path_to_podcast_file}], "sentence": [""]})
    pruna_logger.info(
        "The AI Podcast dataset only consists of one sample, returning same data for training, validation and testing."
    )
    return ds, ds, ds

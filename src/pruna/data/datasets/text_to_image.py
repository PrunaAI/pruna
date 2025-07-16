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

import urllib.request
import zipfile
from functools import partial
from pathlib import Path
from typing import Any, Dict, Tuple

from datasets import Dataset, config, load_dataset
from PIL import Image

from pruna.data.utils import split_train_into_train_val_test, split_val_into_val_test
from pruna.logging.logger import pruna_logger


def setup_open_image_dataset(seed: int) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the OpenImage dataset.

    License: Apache 2.0

    Parameters
    ----------
    seed : int
        The seed to use.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The OpenImage dataset.
    """
    dataset = load_dataset("data-is-better-together/open-image-preferences-v1", trust_remote_code=True)["cleaned"]
    dataset = dataset.rename_column("image_quality_dev", "image")
    dataset = dataset.rename_column("quality_prompt", "text")
    return split_train_into_train_val_test(dataset, seed)


def setup_laion256_dataset(seed: int) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Set up the LAION256 dataset.

    Parameters
    ----------
    seed : int
        The seed to use for splitting the dataset.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The LAION256 dataset.
    """
    dataset = load_dataset("nannullna/laion_subset", trust_remote_code=True)["artwork"]
    return split_train_into_train_val_test(dataset, seed)


def setup_coco_dataset(seed: int) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Set up the COCO dataset.

    Parameters
    ----------
    seed : int
        The seed to use for splitting the dataset.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The COCO dataset.
    """
    # original license Creative Commons Attribution 4.0 License (CC BY 4.0)
    directory_dataset = Path(config.HF_DATASETS_CACHE) / "coco"
    if not directory_dataset.exists():
        directory_dataset.mkdir(parents=True)
        pruna_logger.info(f"Downloading COCO dataset to {directory_dataset}...")
        pruna_logger.info("Downloading COCO can take up to 15 minutes...")

        url = "http://images.cocodataset.org/zips/"
        for target in ["train2017.zip", "val2017.zip"]:
            zip_path = directory_dataset / target
            urllib.request.urlretrieve(url + target, zip_path)

            # Unzip the files
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(directory_dataset)

            zip_path.unlink()

    dataset = load_dataset("phiyodr/coco2017", trust_remote_code=True)

    def _process_example(example: Dict[str, Any], directory_dataset: str) -> Dict[str, Any]:
        """
        Helper function to load image from disk and rename fields.

        Parameters
        ----------
        example : Dict[str, Any]
            The example to process.
        directory_dataset : str
            The directory to load the image from.

        Returns
        -------
        Dict[str, Any]
            The processed example.
        """
        image_path = Path(directory_dataset) / example["file_name"]
        example["image"] = Image.open(image_path)
        example["text"] = example["captions"][0]
        return example

    train_dataset = dataset["train"].map(partial(_process_example, directory_dataset=directory_dataset))
    val_dataset = dataset["validation"].map(partial(_process_example, directory_dataset=directory_dataset))
    val_dataset, test_dataset = split_val_into_val_test(val_dataset, seed)
    return train_dataset, val_dataset, test_dataset

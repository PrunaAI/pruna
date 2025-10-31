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

from typing import Tuple

from datasets import load_dataset
from torch.utils.data import Dataset

from pruna.data.utils import (
    define_sample_size_for_dataset,
    split_train_into_train_val,
    split_val_into_val_test,
    stratify_dataset,
)


def setup_mnist_dataset(
    seed: int, fraction: float = 1.0, train_sample_size: int | None = None, test_sample_size: int | None = None
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the MNIST dataset.

    License: MIT

    Parameters
    ----------
    seed : int
        The seed to use.
    fraction : float
        The fraction of the dataset to use.
    train_sample_size : int | None
        The sample size to use for the train dataset.
    test_sample_size : int | None
        The sample size to use for the test dataset.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The MNIST dataset.
    """
    train_ds, test_ds = load_dataset("ylecun/mnist", split=["train", "test"])  # type: ignore[misc]

    train_sample_size = define_sample_size_for_dataset(train_ds, fraction, train_sample_size)
    test_sample_size = define_sample_size_for_dataset(test_ds, fraction, test_sample_size)

    train_ds = stratify_dataset(train_ds, train_sample_size, seed)
    test_ds = stratify_dataset(test_ds, test_sample_size, seed)

    train_ds, val_ds = split_train_into_train_val(train_ds, seed)
    val_ds, test_ds = split_val_into_val_test(val_ds, seed)

    return train_ds, val_ds, test_ds  # type: ignore[return-value]


def setup_imagenet_dataset(
    seed: int, fraction: float = 1.0, train_sample_size: int | None = None, test_sample_size: int | None = None
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the ImageNet dataset.

    License: unspecified

    Parameters
    ----------
    seed : int
        The seed to use.
    fraction : float
        The fraction of the dataset to use.
    train_sample_size : int | None
        The sample size to use for the train dataset.
    test_sample_size : int | None
        The sample size to use for the test dataset.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The ImageNet dataset.
    """
    train_ds, val = load_dataset("zh-plus/tiny-imagenet", split=["train", "valid"])  # type: ignore[misc]
    train_sample_size = define_sample_size_for_dataset(train_ds, fraction, train_sample_size)
    train_ds = stratify_dataset(train_ds, train_sample_size, seed)
    val_ds, test_ds = split_val_into_val_test(val, seed)
    test_sample_size = define_sample_size_for_dataset(test_ds, fraction, test_sample_size)
    test_ds = stratify_dataset(test_ds, test_sample_size, seed)
    return train_ds, val_ds, test_ds  # type: ignore[return-value]


def setup_cifar10_dataset(
    seed: int, fraction: float = 1.0, train_sample_size: int | None = None, test_sample_size: int | None = None
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the CIFAR-10 dataset.

    The original CIFAR-10 dataset from uoft-cs/cifar10 has an 'img' column,
    but this function renames it to 'image' to ensure compatibility with
    the image_classification_collate function which expects an 'image' column.

    License: unspecified

    Parameters
    ----------
    seed : int
        The seed to use.
    fraction : float
        The fraction of the dataset to use.
    train_sample_size : int | None
        The sample size to use for the train dataset.
    test_sample_size : int | None
        The sample size to use for the test dataset.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The CIFAR-10 dataset with columns: 'image' (PIL Image) and 'label' (int).
    """
    train_ds, test_ds = load_dataset("uoft-cs/cifar10", split=["train", "test"])

    # Rename 'img' column to 'image' to match collate function expectations
    # This ensures compatibility with image_classification_collate function
    train_ds = train_ds.rename_column("img", "image")
    test_ds = test_ds.rename_column("img", "image")

    train_sample_size = define_sample_size_for_dataset(train_ds, fraction, train_sample_size)
    test_sample_size = define_sample_size_for_dataset(test_ds, fraction, test_sample_size)

    train_ds = stratify_dataset(train_ds, train_sample_size, seed)
    test_ds = stratify_dataset(test_ds, test_sample_size, seed)

    train_ds, val_ds = split_train_into_train_val(train_ds, seed)
    return train_ds, val_ds, test_ds  # type: ignore[return-value]

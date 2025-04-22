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

from typing import Any, Tuple, Union

import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from pruna.logging.logger import pruna_logger


class TokenizerMissingError(Exception):
    """
    Custom exception raised when a tokenizer is required but not provided.

    Parameters
    ----------
    message : str, optional
        The message to display when the exception is raised.
    """

    def __init__(self, message: str = "Tokenizer is missing. Please provide a valid tokenizer.") -> None:
        super().__init__(message)


def split_train_into_train_val_test(dataset: Dataset, seed: int) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split the training dataset into train, validation, and test.

    Parameters
    ----------
    dataset : Dataset
        The dataset to split.
    seed : int
        The seed to use for splitting the dataset.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The train, validation, and test datasets.
    """
    pruna_logger.info("Loaded only training, splitting train 80/10/10 into train, validation and test...")
    train_val_split = dataset.train_test_split(test_size=0.2, seed=seed)
    train_ds, val_test_split = train_val_split["train"], train_val_split["test"]
    train_val_split = val_test_split.train_test_split(test_size=0.5, seed=seed)
    val_ds, test_ds = train_val_split["train"], train_val_split["test"]
    return train_ds, val_ds, test_ds


def split_train_into_train_val(dataset: Dataset, seed: int) -> Tuple[Dataset, Dataset]:
    """
    Split the trainingdataset into train and validation.

    Parameters
    ----------
    dataset : Dataset
        The dataset to split.
    seed : int
        The seed to use for splitting the dataset.

    Returns
    -------
    Tuple[Dataset, Dataset]
        The train and validation datasets.
    """
    pruna_logger.info("Loaded only training and test, splitting train 90/10 into train and validation...")
    train_val_split = dataset.train_test_split(test_size=0.1, seed=seed)
    train_ds, val_ds = train_val_split["train"], train_val_split["test"]
    return train_ds, val_ds


def split_val_into_val_test(dataset: Dataset, seed: int) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into validation and test.

    Parameters
    ----------
    dataset : Dataset
        The dataset to split.
    seed : int
        The seed to use for splitting the dataset.

    Returns
    -------
    Tuple[Dataset, Dataset]
        The validation and test datasets.
    """
    val_test_split = dataset.train_test_split(test_size=0.5, seed=seed)
    val_ds, test_ds = val_test_split["train"], val_test_split["test"]
    return val_ds, test_ds


def move_batch_to_device(batch: Any, device: Union[torch.device, str]) -> Any:
    """
    Recursively move all tensors in the batch to the specified device.

    Parameters
    ----------
    batch : Any
        The batch to be processed, which can be a tensor, dict, list, or tuple.
    device : torch.device | str
        The device to which tensors should be moved.

    Returns
    -------
    Any
        The batch with all tensors moved to the specified device.
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return [move_batch_to_device(v, device) for v in batch]
    else:
        return batch


def wrap_batch_for_model_call(batch: Any, model: Any, device: Union[torch.device, str]) -> None:
    """
    Wrap the batch for model call, casting tensors to the specified device.

    Parameters
    ----------
    batch : Any
        The batch to be wrapped, potentially containing tensors.
    model : Any
        The model to be called.
    device : torch.device, optional
        The device to which tensors should be moved (default is 'cuda').
    """
    batch = move_batch_to_device(batch, device)

    if isinstance(batch, dict):
        model(**batch)
    elif isinstance(batch, (tuple, list)):
        model(batch[0])
    else:
        pruna_logger.error("Batch is provided in unexpected format.")
        raise ValueError("Unexpected batch format")


def recover_text_from_dataloader(dataloader: DataLoader, tokenizer: Any) -> list:
    """
    Recover text from the dataloader.

    Parameters
    ----------
    dataloader : DataLoader
        The dataloader to recover text from.
    tokenizer : Any
        The tokenizer to use for decoding.

    Returns
    -------
    list
        The recovered texts.
    """
    texts = []
    for i, batch in enumerate(dataloader):
        if isinstance(batch, (list, tuple)):
            out = tokenizer.batch_decode(torch.cat((batch[0], batch[1]), dim=1))
        elif isinstance(batch, dict):
            out = tokenizer.batch_decode(torch.cat((batch["input"], batch["target"]), dim=1))
        else:
            pruna_logger.error("Batch is provided in unexpected format.")
            raise ValueError()
        texts.extend(out)
    return texts

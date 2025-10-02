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

import copy
from typing import Tuple

from datasets import load_dataset, Dataset

from pruna.data.utils import split_train_into_train_val, split_train_into_train_val_test
from pruna.logging.logger import pruna_logger


def setup_wikitext_dataset() -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the WikiText dataset.

    License: unspecified, original license Creative Commons Attribution-ShareAlike License (CC BY-SA)

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The WikiText dataset.
    """
    train_dataset, val_dataset, test_dataset = load_dataset(
        path="mikasenghaas/wikitext-2",
        split=["train", "validation", "test"]
    )
    return train_dataset, val_dataset, test_dataset


def setup_wikitext_tiny_dataset(seed: int = 42, num_rows: int = 960) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the TinyWikiText dataset. Splits the dataset .8/.1/.1 into train/val/test subsets, respectively.

    License: unspecified, original license Creative Commons Attribution-ShareAlike License (CC BY-SA).

    Parameters
    ----------
    seed : int
        The seed to use (default 42).
    num_rows : int
        The maximum total number of rows in the tiny dataset (default 960).

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The TinyWikiText dataset split .8/.1/.1 into train/val/test subsets, respectively.
    """
    assert 10 <= num_rows < 1000, 'the total number of rows, r, for the tiny wikitext dataset must be 10 <= r < 1000'

    # load the 'mikasenghaas/wikitext-2' dataset with a total of 21,580 rows using the setup_wikitext_dataset() function
    train_ds, val_ds, test_ds = setup_wikitext_dataset()

    # assert the wikitext dataset train/val/test splits each have enough rows for reducing to .8/.1/.1, respectively
    assert (
        train_ds.num_rows >= int(num_rows * 0.8),
        f'wikitext cannot be reduced to {num_rows} rows, train split too small'
    )
    assert (
        val_ds.num_rows >= int(num_rows * 0.1),
        f'wikitext cannot be reduced to {num_rows} rows, val split too small'
    )
    assert (
        test_ds.num_rows >= int(num_rows * 0.1),
        f'wikitext cannot be reduced to {num_rows} rows, test split too small'
    )

    # randomly select from the wikitext dataset a total number of rows below 1000 split .8/.1/.1 between train/val/test
    train_dataset_tiny = train_ds.shuffle(seed=seed).select(range(int(num_rows * 0.8)))
    val_dataset_tiny = val_ds.shuffle(seed=seed).select(range(int(num_rows * 0.1)))
    test_dataset_tiny = test_ds.shuffle(seed=seed).select(range(int(num_rows * 0.1)))
    return train_dataset_tiny, val_dataset_tiny, test_dataset_tiny


def setup_smoltalk_dataset(seed: int) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the SmolTalk dataset.

    License: Apache 2.0

    Parameters
    ----------
    seed : int
        The seed to use.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The SmolTalk dataset.
    """
    train_full = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split="train")
    test_data = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split="test")

    train_ds, val_ds = split_train_into_train_val(train_full, seed)

    def _prepare_text(example: dict) -> dict:
        """
        Converts the 'messages' list of {role, content} dicts into a single text string under 'text'.

        Parameters
        ----------
        example : dict
            The example to prepare.

        Returns
        -------
        dict
            The prepared example.
        """
        message_data = example["messages"]
        # replicate the logic from __getitem__ method
        text = " ".join(f"{message['role']}\n{message['content']}\n" for message in message_data)
        return {"text": text}

    # Apply map function to transform 'messages' into a single 'text' field
    train_ds = train_ds.map(_prepare_text)
    val_ds = val_ds.map(_prepare_text)
    test_ds = test_data.map(_prepare_text)

    return train_ds, val_ds, test_ds


def setup_smolsmoltalk_dataset(seed: int) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the SmolSmolTalk dataset.

    License: Apache 2.0

    Parameters
    ----------
    seed : int
        The seed to use.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The SmolSmolTalk dataset.
    """
    train_full = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
    test_ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="test")

    train_ds, val_ds = split_train_into_train_val(train_full, seed)

    def _prepare_text(example: dict) -> dict:
        """
        Converts the 'messages' list of {role, content} dicts into a single text string under 'text'.

        Parameters
        ----------
        example : dict
            The example to prepare.

        Returns
        -------
        dict
            The prepared example.
        """
        message_data = example["messages"]
        # replicate the logic from __getitem__ method
        text = " ".join(f"{message['role']}\n{message['content']}\n" for message in message_data)
        return {"text": text}

    # Apply map function to transform 'messages' into a single 'text' field
    train_ds = train_ds.map(_prepare_text)
    val_ds = val_ds.map(_prepare_text)
    test_ds = test_ds.map(_prepare_text)

    return train_ds, val_ds, test_ds


def setup_pubchem_dataset(seed: int) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the PubChem dataset.

    License: unspecified

    Parameters
    ----------
    seed : int
        The seed to use.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The PubChem dataset.
    """
    dataset = load_dataset("alxfgh/PubChem10M_SELFIES")["train"]
    dataset = dataset.rename_column("SELFIES", "text")
    return split_train_into_train_val_test(dataset, seed)


def setup_openassistant_dataset(seed: int) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the OpenAssistant dataset.

    License: Apache 2.0

    Parameters
    ----------
    seed : int
        The seed to use.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The OpenAssistant dataset.
    """
    train_dataset, test_dataset = load_dataset("timdettmers/openassistant-guanaco", split=["train", "test"])
    train_ds, val_ds = split_train_into_train_val(train_dataset, seed)
    return train_ds, val_ds, test_dataset


def setup_c4_dataset() -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the C4 dataset.

    License: Open Data Commons Attribution License (ODC-BY)

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The C4 dataset.
    """
    train_dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    val_dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    pruna_logger.info("Received only train and val datasets as iterable datasets, copying validation dataset to test.")
    test_dataset = copy.deepcopy(val_dataset)
    return train_dataset, val_dataset, test_dataset

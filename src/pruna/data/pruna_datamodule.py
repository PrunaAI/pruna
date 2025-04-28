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

import inspect
from functools import partial
from typing import Callable, List, Tuple, Union

from datasets import Dataset, IterableDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset as TorchDataset
from transformers.tokenization_utils import PreTrainedTokenizer as AutoTokenizer

from pruna.data import base_datasets
from pruna.data.collate import pruna_collate_fns
from pruna.data.utils import TokenizerMissingError
from pruna.logging.logger import pruna_logger


class PrunaDataModule(LightningDataModule):
    """
    A PrunaDataModule is a wrapper around a PyTorch Lightning DataModule that allows for easy loading of datasets.

    Parameters
    ----------
    train_ds : Union[IterableDataset, Dataset, TorchDataset]
        The training dataset.
    val_ds : Union[IterableDataset, Dataset, TorchDataset]
        The validation dataset.
    test_ds : Union[IterableDataset, Dataset, TorchDataset]
        The test dataset.
    collate_fn : Callable
        The collate function to use.
    dataloader_args : dict
        The arguments for the dataloader.
    """

    def __init__(
        self,
        train_ds: Union[IterableDataset, Dataset, TorchDataset],
        val_ds: Union[IterableDataset, Dataset, TorchDataset],
        test_ds: Union[IterableDataset, Dataset, TorchDataset],
        collate_fn: Callable,
        dataloader_args: dict,
    ) -> None:
        super().__init__()
        self.train_dataset: Union[IterableDataset, Dataset, TorchDataset] = train_ds
        self.val_dataset: Union[IterableDataset, Dataset, TorchDataset] = val_ds
        self.test_dataset: Union[IterableDataset, Dataset, TorchDataset] = test_ds
        self.collate_fn = collate_fn
        self.dataloader_args = dataloader_args

    @classmethod
    def from_datasets(  # type: ignore[override]
        cls,
        datasets: (
            Tuple[Union[IterableDataset, Dataset, TorchDataset], ...]
            | List[Union[IterableDataset, Dataset, TorchDataset]]
        ),
        collate_fn: str,
        tokenizer: AutoTokenizer | None = None,
        collate_fn_args: dict = dict(),
        dataloader_args: dict = dict(),
    ) -> "PrunaDataModule":
        """
        Create a PrunaDataModule from the individual datasets.

        Parameters
        ----------
        datasets : tuple | list
            The datasets.
        collate_fn : str
            The Pruna collate function to use.
        tokenizer : AutoTokenizer | None
            The tokenizer to use.
        collate_fn_args : dict
            Any additional arguments for the collate function.
        dataloader_args : dict
            Any additional arguments for the dataloader.

        Returns
        -------
        PrunaDataModule
            The PrunaDataModule.
        """
        if len(datasets) != 3:
            pruna_logger.error("Datasets must contain exactly 3 elements: train, validation, and test.")
            raise ValueError()

        if tokenizer is not None:
            collate_fn_args["tokenizer"] = tokenizer
            if "max_seq_len" not in collate_fn_args:
                try:
                    max_seq_len = tokenizer.model_max_length
                    # some models define a max_seq_len that is too large when the tokenizer converts this to float
                    if max_seq_len > 1e10:
                        max_seq_len = None
                    collate_fn_args["max_seq_len"] = max_seq_len
                    pruna_logger.info(f"Using max_seq_len of tokenizer: {max_seq_len}")
                except AttributeError:
                    pass

        train_ds, val_ds, test_ds = datasets
        collate_func = get_collate_fn(collate_fn, collate_fn_args)

        collate_func_name = collate_func.func.__name__ if isinstance(collate_func, partial) else collate_func.__name__
        pruna_logger.info(f"Testing compatibility with {collate_func_name}...")
        try:
            for ds in [train_ds, val_ds, test_ds]:
                # try collating two samples to test if batching works with the dataset
                collate_func([next(iter(ds))] * 2)
        except Exception as e:
            raise ValueError(f"Compatibility test failed with error: {e}")

        return cls(train_ds, val_ds, test_ds, collate_func, dataloader_args)

    @classmethod
    def from_string(
        cls,
        dataset_name: str,
        tokenizer: AutoTokenizer | None = None,
        collate_fn_args: dict = dict(),
        dataloader_args: dict = dict(),
        seed: int = 42,
    ) -> "PrunaDataModule":
        """
        Create a PrunaDataModule from the dataset name with preimplemented dataset loading.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset.
        tokenizer : AutoTokenizer | None
            The tokenizer to use.
        collate_fn_args : dict
            Any additional arguments for the collate function.
        dataloader_args : dict
            Any additional arguments for the dataloader.
        seed : int
            The seed to use.

        Returns
        -------
        PrunaDataModule
            The PrunaDataModule.
        """
        setup_fn, collate_fn_name, default_collate_fn_args = base_datasets[dataset_name]

        # use default collate_fn_args and override with user-provided ones
        default_collate_fn_args.update(collate_fn_args)
        collate_fn_args = default_collate_fn_args

        if "seed" in inspect.signature(setup_fn).parameters:
            setup_fn = partial(setup_fn, seed=seed)
        train_ds, val_ds, test_ds = setup_fn()

        return cls.from_datasets(
            (train_ds, val_ds, test_ds), collate_fn_name, tokenizer, collate_fn_args, dataloader_args
        )

    def limit_datasets(self, limit: int | list[int] | tuple[int, int, int]) -> None:
        """
        Limit the dataset to the given number of samples.

        Parameters
        ----------
        limit : int | list[int] | tuple[int, int, int]
            The number of samples to limit the dataset to.
        """
        if isinstance(limit, int):
            train_limit, val_limit, test_limit = limit, limit, limit
        else:
            if len(limit) != 3:
                pruna_logger.error("Limit must be a list of 3 integers for train, val, and test.")
                raise ValueError
            train_limit, val_limit, test_limit = limit

        if isinstance(self.train_dataset, Dataset):
            self.train_dataset = self.train_dataset.select(range(train_limit))
            self.val_dataset = self.val_dataset.select(range(val_limit))  # type: ignore[union-attr]
            self.test_dataset = self.test_dataset.select(range(test_limit))  # type: ignore[union-attr]
        elif isinstance(self.train_dataset, IterableDataset):
            self.train_dataset = self.train_dataset[:train_limit]
            self.val_dataset = self.val_dataset[:val_limit]
            self.test_dataset = self.test_dataset[:test_limit]
        elif isinstance(self.train_dataset, TorchDataset):
            train_indices = list(range(min(len(self.train_dataset), train_limit)))  # type: ignore[arg-type]
            val_indices = list(range(min(len(self.val_dataset), val_limit)))  # type: ignore[arg-type]
            test_indices = list(range(min(len(self.test_dataset), test_limit)))  # type: ignore[arg-type]
            self.train_dataset = Subset(self.train_dataset, train_indices)
            self.val_dataset = Subset(self.val_dataset, val_indices)
            self.test_dataset = Subset(self.test_dataset, test_indices)
        else:
            raise ValueError("Dataset could not be limited.")

    def train_dataloader(self, **kwargs) -> DataLoader:
        """
        Return the training data loader.

        Parameters
        ----------
        **kwargs : dict
            Any additional arguments used when loading data, overriding the default values provided in the constructor.
            Examples: img_size: int would override the collate function default for image generation,
            while batch_size: int, shuffle: bool, pin_memory: bool, ... would override the dataloader defaults.

        Returns
        -------
        DataLoader
            The training data loader.
        """
        if "shuffle" not in kwargs:
            kwargs["shuffle"] = True  # dispatched to the dataloader

        return self.__construct_dataloader(self.train_dataset, **kwargs)

    def val_dataloader(self, **kwargs) -> DataLoader:
        """
        Return the validation data loader.

        Parameters
        ----------
        **kwargs : dict
            Any additional arguments used when loading data, overriding the default values provided in the constructor.
            Examples: img_size: int would override the collate function default for image generation,
            while batch_size: int, shuffle: bool, pin_memory: bool, ... would override the dataloader defaults.

        Returns
        -------
        DataLoader
            The validation data loader.
        """
        if "shuffle" not in kwargs:
            kwargs["shuffle"] = False  # dispatched to the dataloader

        return self.__construct_dataloader(self.val_dataset, **kwargs)

    def test_dataloader(self, **kwargs) -> DataLoader:
        """
        Return the test data loader.

        Parameters
        ----------
        **kwargs : dict
            Any additional arguments used when loading data, overriding the default values provided in the constructor.
            Examples: img_size: int would override the collate function default for image generation,
            while batch_size: int, shuffle: bool, pin_memory: bool, ... would override the dataloader defaults.

        Returns
        -------
        DataLoader
            The test data loader.
        """
        if "shuffle" not in kwargs:
            kwargs["shuffle"] = False  # dispatched to the dataloader

        return self.__construct_dataloader(self.test_dataset, **kwargs)

    def __construct_dataloader(
        self,
        dataset: IterableDataset | Dataset,
        shuffle: bool,
        **kwargs: dict,
    ) -> DataLoader:
        # dispatch each kwarg to either collate_fn or dataloader (priority is given to collate_fn)
        collate_fn = self.collate_fn
        collate_fn_possible_args = inspect.signature(collate_fn).parameters.keys()
        collate_fn_args = {k: v for k, v in kwargs.items() if k in collate_fn_possible_args}
        dataloader_args = {k: v for k, v in kwargs.items() if k not in collate_fn_possible_args}

        # update collate_fn with the user-provided arguments
        if collate_fn_args:
            # add arguments to the collate function or override those previously provided
            collate_fn = partial(collate_fn, **collate_fn_args)

        # use self.dataloader_args as defaults, and overwrite with local dataloader_args
        combined_dataloader_args = self.dataloader_args.copy()
        combined_dataloader_args.update(dataloader_args)

        # when loading a dataset with streaming=True, resulting in Iterable, we can not specify the shuffle option
        if isinstance(dataset, IterableDataset):
            return DataLoader(dataset, collate_fn=collate_fn, **combined_dataloader_args)
        else:
            return DataLoader(dataset, shuffle=shuffle, collate_fn=collate_fn, **combined_dataloader_args)


def get_collate_fn(collate_fn_name: str, collate_fn_args: dict) -> Callable:
    """
    Get and prepare the collate function.

    Parameters
    ----------
    collate_fn_name : str
        The name of the collate function.
    collate_fn_args : dict
        Any additional arguments for the collate function.

    Returns
    -------
    Callable
        The collate function.
    """
    collate_fn = pruna_collate_fns[collate_fn_name]

    # Inspect the signature of the underlying function
    signature = inspect.signature(collate_fn)

    # Gather required parameters (no default) that are not in collate_fn_args
    # excluding the "data" parameter
    missing_required_params = []
    for param_name, param in signature.parameters.items():
        if param_name == "data":
            # We exclude `data` from this check
            continue

        if param_name == "tokenizer" and "tokenizer" not in collate_fn_args:
            raise TokenizerMissingError(
                "Tokenizer is required but not provided. Please provide a tokenizer with the keyword 'tokenizer'."
            )

        # If the parameter has no default value AND it is not in collate_fn_args
        if (param.default is inspect.Parameter.empty) and (param_name not in collate_fn_args):
            missing_required_params.append(param_name)

    if missing_required_params:
        raise ValueError(f"The following required parameters are missing in collate_fn_args: {missing_required_params}")

    # Create a partial with the given arguments
    collate_fn = partial(collate_fn, **collate_fn_args)
    return collate_fn

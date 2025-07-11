from typing import Any, Callable

import pytest
from transformers import AutoTokenizer

from pruna.data.datasets.image import setup_imagenet_dataset
from pruna.data.pruna_datamodule import PrunaDataModule
from datasets import Dataset
import torch
from torch.utils.data import TensorDataset

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def iterate_dataloaders(datamodule: PrunaDataModule) -> None:
    """Iterate through the dataloaders of the datamodule."""
    next(iter(datamodule.train_dataloader()))
    next(iter(datamodule.val_dataloader()))
    next(iter(datamodule.test_dataloader()))


@pytest.mark.cpu
@pytest.mark.parametrize(
    "dataset_name, collate_fn_args",
    [
        pytest.param("COCO", dict(img_size=512), marks=pytest.mark.slow),
        pytest.param("LAION256", dict(img_size=512), marks=pytest.mark.slow),
        pytest.param("OpenImage", dict(img_size=512), marks=pytest.mark.slow),
        pytest.param("CommonVoice", dict(), marks=pytest.mark.slow),
        pytest.param("AIPodcast", dict(), marks=pytest.mark.slow),
        ("ImageNet", dict(img_size=512)),
        pytest.param("MNIST", dict(img_size=512), marks=pytest.mark.slow),
        ("WikiText", dict(tokenizer=bert_tokenizer)),
        pytest.param("SmolTalk", dict(tokenizer=bert_tokenizer), marks=pytest.mark.slow),
        pytest.param("SmolSmolTalk", dict(tokenizer=bert_tokenizer), marks=pytest.mark.slow),
        pytest.param("PubChem", dict(tokenizer=bert_tokenizer), marks=pytest.mark.slow),
        pytest.param("OpenAssistant", dict(tokenizer=bert_tokenizer), marks=pytest.mark.slow),
        pytest.param("C4", dict(tokenizer=bert_tokenizer), marks=pytest.mark.slow),
        pytest.param("Polyglot", dict(tokenizer=bert_tokenizer), marks=pytest.mark.slow),
    ],
)
def test_dm_from_string(dataset_name: str, collate_fn_args: dict[str, Any]) -> None:
    """Test the datamodule from a string."""
    # get tokenizer if available
    tokenizer = collate_fn_args.get("tokenizer", None)

    # get the datamodule from the string
    datamodule = PrunaDataModule.from_string(dataset_name, collate_fn_args=collate_fn_args, tokenizer=tokenizer)

    # iterate through the dataloaders
    iterate_dataloaders(datamodule)


@pytest.mark.cpu
@pytest.mark.parametrize(
    "setup_fn, collate_fn, collate_fn_args",
    [(setup_imagenet_dataset, "image_classification_collate", dict(img_size=512))],
)
def test_dm_from_dataset(setup_fn: Callable, collate_fn: Callable, collate_fn_args: dict[str, Any]) -> None:
    """Test the datamodule from a dataset."""
    # get datamodule with datasets and collate function as input
    datasets = setup_fn(seed=123)
    datamodule = PrunaDataModule.from_datasets(datasets, collate_fn, collate_fn_args=collate_fn_args)

    # iterate through the dataloaders
    iterate_dataloaders(datamodule)
@pytest.mark.cpu
@pytest.mark.parametrize("limit_len", [2, (2, 3, 1), (10, 10, 10)])
@pytest.mark.parametrize("ds_type", ["hf", "torch"])
def test_limit_datasets(ds_type: str, limit_len: int | tuple[int, int, int]) -> None:
    """Test the `limit_datasets` method of PrunaDataModule with various dataset types and limits."""

    # Create small datasets
    if ds_type == "hf":
        data = {"text": ["a", "b", "c", "d", "e"]}
        train_ds = val_ds = test_ds = Dataset.from_dict(data)
    elif ds_type == "torch":
        x = torch.arange(5).float().unsqueeze(1)
        y = torch.arange(5)
        train_ds = val_ds = test_ds = TensorDataset(x, y)
    else:
        raise ValueError("Unsupported dataset type")

    def dummy_collate(batch):
        return batch

    datamodule = PrunaDataModule(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        collate_fn=dummy_collate,
        dataloader_args={"batch_size": 1},
    )

    original_len = len(train_ds)
    datamodule.limit_datasets(limit_len)

    train_len = len(datamodule.train_dataset)
    val_len = len(datamodule.val_dataset)
    test_len = len(datamodule.test_dataset)

    if isinstance(limit_len, int):
        expected = min(original_len, limit_len)
        assert train_len == val_len == test_len == expected
    else:
        assert train_len == min(original_len, limit_len[0])
        assert val_len == min(original_len, limit_len[1])
        assert test_len == min(original_len, limit_len[2])
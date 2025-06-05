from typing import Any, Callable

import pytest
from transformers import AutoTokenizer
from datasets import Dataset 
from torch.utils.data import TensorDataset
import torch
from pruna.data.datasets.image import setup_imagenet_dataset
from pruna.data.pruna_datamodule import PrunaDataModule

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
    datamodule.limit_datasets(10)


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
    datamodule.limit_datasets(10)

    # iterate through the dataloaders
    iterate_dataloaders(datamodule)
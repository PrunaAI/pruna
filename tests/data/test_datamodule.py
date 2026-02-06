from typing import Any, Callable

import pytest
import torch
from transformers import AutoTokenizer

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
        pytest.param("LibriSpeech", dict(), marks=pytest.mark.slow),
        pytest.param("AIPodcast", dict(), marks=pytest.mark.slow),
        ("ImageNet", dict(img_size=512)),
        ("TinyCIFAR10", dict(img_size=32)),
        ("TinyImageNet", dict(img_size=224)),
        ("TinyMNIST", dict(img_size=28)),
        pytest.param("MNIST", dict(img_size=512), marks=pytest.mark.slow),
        ("WikiText", dict(tokenizer=bert_tokenizer)),
        pytest.param("TinyWikiText", dict(tokenizer=bert_tokenizer), marks=pytest.mark.slow),
        pytest.param("SmolTalk", dict(tokenizer=bert_tokenizer), marks=pytest.mark.slow),
        pytest.param("SmolSmolTalk", dict(tokenizer=bert_tokenizer), marks=pytest.mark.slow),
        pytest.param("PubChem", dict(tokenizer=bert_tokenizer), marks=pytest.mark.slow),
        pytest.param("OpenAssistant", dict(tokenizer=bert_tokenizer), marks=pytest.mark.slow),
        pytest.param("C4", dict(tokenizer=bert_tokenizer), marks=pytest.mark.slow),
        pytest.param("Polyglot", dict(tokenizer=bert_tokenizer), marks=pytest.mark.slow),
        pytest.param("DrawBench", dict(), marks=pytest.mark.slow),
        pytest.param("PartiPrompts", dict(), marks=pytest.mark.slow),
        pytest.param("GenAIBench", dict(), marks=pytest.mark.slow),
        pytest.param("TinyIMDB", dict(tokenizer=bert_tokenizer), marks=pytest.mark.slow),
        pytest.param("VBench", dict(), marks=pytest.mark.slow),
        pytest.param("ImgEdit", dict(), marks=pytest.mark.slow),
        pytest.param("GEditBench", dict(), marks=pytest.mark.slow),
    ],
)
def test_dm_from_string(dataset_name: str, collate_fn_args: dict[str, Any]) -> None:
    """Test the datamodule from a string."""
    # get tokenizer if available
    tokenizer = collate_fn_args.get("tokenizer")

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
    batch = next(iter(datamodule.train_dataloader()))
    images, labels = batch
    assert images.shape[1] == 3
    assert images.shape[2] == collate_fn_args["img_size"]
    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64
    # iterate through the dataloaders
    iterate_dataloaders(datamodule)


@pytest.mark.slow
def test_parti_prompts_with_category_filter():
    """Test PartiPrompts loading with category filter."""
    dm = PrunaDataModule.from_string("PartiPrompts", category="Animals", dataloader_args={"batch_size": 4})
    dm.limit_datasets(10)
    batch = next(iter(dm.test_dataloader()))
    prompts, auxiliaries = batch

    assert len(prompts) == 4
    assert all(isinstance(p, str) for p in prompts)
    assert all(aux["Category"] == "Animals" for aux in auxiliaries)


@pytest.mark.slow
def test_imgedit_with_category_filter():
    """Test ImgEdit loading with category filter."""
    dm = PrunaDataModule.from_string("ImgEdit", category="replace", dataloader_args={"batch_size": 4})
    dm.limit_datasets(10)
    batch = next(iter(dm.test_dataloader()))
    prompts, auxiliaries = batch

    assert len(prompts) == 4
    assert all(isinstance(p, str) for p in prompts)
    assert all(aux["category"] == "replace" for aux in auxiliaries)
    assert all("judge_prompt" in aux for aux in auxiliaries)


@pytest.mark.slow
def test_geditbench_with_category_filter():
    """Test GEditBench loading with category filter."""
    dm = PrunaDataModule.from_string("GEditBench", category="background_change", dataloader_args={"batch_size": 4})
    dm.limit_datasets(10)
    batch = next(iter(dm.test_dataloader()))
    prompts, auxiliaries = batch

    assert len(prompts) == 4
    assert all(isinstance(p, str) for p in prompts)
    assert all(aux["category"] == "background_change" for aux in auxiliaries)

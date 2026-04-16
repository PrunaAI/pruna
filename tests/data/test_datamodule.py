import importlib.util
from typing import Any, Callable

import pytest
import torch
from transformers import AutoTokenizer

from pruna.data import base_datasets
from pruna.data.datasets.image import setup_imagenet_dataset
from pruna.data.pruna_datamodule import PrunaDataModule
from pruna.data.utils import get_literal_values_from_param

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def iterate_dataloaders(datamodule: PrunaDataModule) -> None:
    """Iterate through the dataloaders of the datamodule."""
    next(iter(datamodule.train_dataloader()))
    next(iter(datamodule.val_dataloader()))
    next(iter(datamodule.test_dataloader()))


def _assert_at_least_one_sample(datamodule: PrunaDataModule) -> None:
    """Assert train, val, and test splits each have at least one sample."""
    for name, ds in zip(
        ("train", "val", "test"),
        (datamodule.train_dataset, datamodule.val_dataset, datamodule.test_dataset),
    ):
        try:
            n = len(ds)
        except TypeError:
            continue
        assert n >= 1, f"{name} split has 0 samples"


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
        pytest.param("HPS", dict(), marks=pytest.mark.slow),
        pytest.param("ImgEdit", dict(), marks=pytest.mark.slow),
        pytest.param("LongTextBench", dict(), marks=pytest.mark.slow),
        pytest.param("GEditBench", dict(), marks=pytest.mark.slow),
        pytest.param("OneIG", dict(), marks=pytest.mark.slow),
        pytest.param("OneIGAnimeStylization", dict(), marks=pytest.mark.slow),
        pytest.param("OneIGGeneralObject", dict(), marks=pytest.mark.slow),
        pytest.param("OneIGKnowledgeReasoning", dict(), marks=pytest.mark.slow),
        pytest.param("OneIGMultilingualism", dict(), marks=pytest.mark.slow),
        pytest.param("OneIGPortrait", dict(), marks=pytest.mark.slow),
        pytest.param("OneIGTextRendering", dict(), marks=pytest.mark.slow),
        pytest.param("GenEval", dict(), marks=pytest.mark.slow),
        pytest.param("DPG", dict(), marks=pytest.mark.slow),
    ],
)
def test_dm_from_string(dataset_name: str, collate_fn_args: dict[str, Any]) -> None:
    """Test the datamodule from a string."""
    # get tokenizer if available
    tokenizer = collate_fn_args.get("tokenizer")

    # get the datamodule from the string
    datamodule = PrunaDataModule.from_string(dataset_name, collate_fn_args=collate_fn_args, tokenizer=tokenizer)
    _assert_at_least_one_sample(datamodule)
    datamodule.limit_datasets(10)

    # iterate through the dataloaders
    iterate_dataloaders(datamodule)


@pytest.mark.cpu
@pytest.mark.parametrize(
    "setup_fn, collate_fn, collate_fn_args",
    [(setup_imagenet_dataset, "image_classification_collate", dict(img_size=512))],
)
def test_dm_from_dataset(setup_fn: Callable, collate_fn: str, collate_fn_args: dict[str, Any]) -> None:
    """Test the datamodule from a dataset."""
    # get datamodule with datasets and collate function as input
    datasets = setup_fn(seed=123)
    datamodule = PrunaDataModule.from_datasets(datasets, collate_fn, collate_fn_args=collate_fn_args)
    _assert_at_least_one_sample(datamodule)
    datamodule.limit_datasets(10)
    batch = next(iter(datamodule.train_dataloader()))
    images, labels = batch
    assert images.shape[1] == 3
    assert images.shape[2] == collate_fn_args["img_size"]
    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64
    # iterate through the dataloaders
    iterate_dataloaders(datamodule)


_PREFERRED_SMOKE_CATEGORY: dict[str, str] = {
    # Prefer top-level categories that are guaranteed to have many samples.
    # Fine-grained art styles (e.g. "3d rendering") sort first alphabetically
    # but may have < 4 samples, which would break the batch-size assertion.
    "OneIG": "Anime_Stylization",
}


def _benchmark_category_smoke() -> list[tuple[str, str]]:
    """One (dataset, category) per benchmark that supports ``category`` (stable, small smoke set)."""
    result = []
    for name in sorted(base_datasets):
        if name == "VBench" and importlib.util.find_spec("vbench") is None:
            continue
        setup_fn = base_datasets[name][0]
        literal_values = get_literal_values_from_param(setup_fn, "category")
        if literal_values:
            category = _PREFERRED_SMOKE_CATEGORY.get(name) or sorted(literal_values)[0]
            result.append((name, category))
    return result


@pytest.mark.cpu
@pytest.mark.slow
@pytest.mark.parametrize("dataset_name, category", _benchmark_category_smoke())
def test_benchmark_category_filter(dataset_name: str, category: str) -> None:
    """Category filter loads and batches match the chosen category (one category per dataset)."""
    dm = PrunaDataModule.from_string(dataset_name, category=category, dataloader_args={"batch_size": 4})
    _assert_at_least_one_sample(dm)
    dm.limit_datasets(10)
    batch = next(iter(dm.test_dataloader()))
    prompts, auxiliaries = batch

    # Some categories have fewer than 4 samples; assert at least one rather than exactly four.
    assert 1 <= len(prompts) <= 4
    assert all(isinstance(p, str) for p in prompts)

    def _category_in_aux(aux: dict, cat: str) -> bool:
        for v in aux.values():
            if v == cat:
                return True
            if isinstance(v, (list, tuple)) and cat in v:
                return True
        return False

    assert all(_category_in_aux(aux, category) for aux in auxiliaries)


@pytest.mark.cpu
@pytest.mark.slow
def test_prompt_benchmark_auxiliaries() -> None:
    """Prompt-based benchmarks expose expected aux keys."""
    for dataset_name, required_aux_key in (
        ("LongTextBench", "text_content"),
        ("OneIG", "text_content"),
    ):
        dm = PrunaDataModule.from_string(dataset_name, dataloader_args={"batch_size": 4})
        dm.limit_datasets(10)
        batch = next(iter(dm.test_dataloader()))
        prompts, auxiliaries = batch

        assert len(prompts) == 4
        assert all(isinstance(p, str) for p in prompts)
        assert all(required_aux_key in aux for aux in auxiliaries)

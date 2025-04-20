from typing import Any, Callable

import pytest

from pruna import SmashConfig
from pruna.data.datasets.image import setup_imagenet_dataset
from pruna.data.pruna_datamodule import PrunaDataModule

from ..data.test_datamodule import iterate_dataloaders


@pytest.mark.cpu
@pytest.mark.parametrize("dataset_name, collate_fn_args", [("ImageNet", dict(img_size=512))])
def test_dm_from_string_to_config(dataset_name: str, collate_fn_args: dict[str, Any]) -> None:
    """Test the datamodule from a string to config."""
    smash_config = SmashConfig()
    smash_config.add_data(dataset_name, collate_fn_args=collate_fn_args)
    iterate_dataloaders(smash_config.data)


@pytest.mark.cpu
@pytest.mark.parametrize("dataset_name, collate_fn_args", [("ImageNet", dict(img_size=512))])
def test_dm_to_config(dataset_name: str, collate_fn_args: dict[str, Any]) -> None:
    """Test the datamodule to config."""
    datamodule = PrunaDataModule.from_string(dataset_name, collate_fn_args=collate_fn_args)
    smash_config = SmashConfig()
    smash_config.add_data(datamodule)
    iterate_dataloaders(smash_config.data)


@pytest.mark.cpu
@pytest.mark.parametrize(
    "setup_fn, collate_fn, collate_fn_args",
    [(setup_imagenet_dataset, "image_classification_collate", dict(img_size=512))],
)
def test_dm_from_datasets_to_config(setup_fn: Callable, collate_fn: Callable, collate_fn_args: dict[str, Any]) -> None:
    """Test the datamodule from a dataset."""
    datasets = setup_fn(seed=123)
    smash_config = SmashConfig()
    smash_config.add_data(datasets, collate_fn=collate_fn, collate_fn_args=collate_fn_args)
    iterate_dataloaders(smash_config.data)


@pytest.mark.cpu
@pytest.mark.parametrize(
    "setup_fn, collate_fn_defaults, args_override",
    [(setup_imagenet_dataset, dict(img_size=224))],
)
def test_img_args_override(
    setup_fn: Callable, collate_fn_defaults: dict[str, Any], args_override: dict[str, Any]
) -> None:
    """Test the datamodule from a dataset."""
    datasets = setup_fn(seed=123)
    smash_config = SmashConfig()
    smash_config.add_data(datasets, collate_fn="image_classification_collate", collate_fn_args=collate_fn_defaults)
    for get_dataloader in [smash_config.train_dataloader, smash_config.val_dataloader, smash_config.test_dataloader]:
        dataloader = get_dataloader(**args_override)
        image, _ = next(iter(dataloader))
        assert image.shape[0] == args_override["batch_size"]
        assert image.shape[2] == args_override["img_size"]
        assert image.shape[3] == args_override["img_size"]

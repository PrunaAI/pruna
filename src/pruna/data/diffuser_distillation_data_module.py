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

import functools
import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pruna.algorithms.global_utils.recovery.finetuners.diffusers.distillation_arg_utils import (
    get_latent_extractor_fn,
)
from pruna.algorithms.global_utils.recovery.finetuners.diffusers.utils import get_denoiser_attr
from pruna.data.pruna_datamodule import PrunaDataModule
from pruna.logging.logger import pruna_logger


class DiffusionDistillationDataModule(PrunaDataModule):
    """
    A distillation datamodule containing captions, random seeds and input-output pairs during the diffusion process.

    Parameters
    ----------
    pipeline : Any
        The diffusion pipeline used to generate the distillation data. The diffuser is expected to have a denoiser
        (either `unet` or `transformer` attribute) whose output is either a tuple with latent output in index 0, or a
        dict-like object with the latent output in the `sample` key.
    caption_datamodule : pl.LightningDataModule
        The caption datamodule. Each batch is expected to be an iterable of captions or a tuple whose first
        element is an iterable of captions.
    save_path : str | Path
        The path to save the distillation data.
    seed : int | None
        The random seed used to generate unique ids and random seeds for the distillation data.
        If None, a random seed will be generated. This seed will be saved in the `parameters.json` file for future runs.
    pipeline_kwargs : dict[str, Any]
        Additional keyword arguments to pass to the pipeline, such as `guidance_scale` or `num_inference_steps`.
    """

    def __init__(
        self,
        pipeline: Any,
        caption_datamodule: PrunaDataModule,
        save_path: str | Path = "distillation_data",
        seed: int | None = None,
        pipeline_kwargs: dict[str, Any] = {},
    ):
        self.pipeline = pipeline
        self.pipeline_kwargs = pipeline_kwargs
        self.collection_helper: InternalStateCollectionHelper | None = InternalStateCollectionHelper(self.pipeline)

        self.caption_datamodule = caption_datamodule
        self.save_path = Path(save_path)

        if seed is None:
            param_path = self.save_path / "parameters.json"
            if param_path.exists():  # load previously saved seed
                with open(param_path, "r") as f:
                    seed = json.load(f)["seed"]
                if not isinstance(seed, int):
                    raise ValueError(f"Seed must be an integer, but got {seed} from the parameters.json file.")
                else:
                    self.seed = seed
            else:
                self.seed = _get_random_seed()
        else:
            self.seed = seed
        generator = torch.Generator().manual_seed(self.seed)
        self.dataloader_generators = {
            subset: torch.Generator().manual_seed(_get_random_seed(generator)) for subset in ["train", "val", "test"]
        }
        self.seed_making_generators = {
            subset: torch.Generator().manual_seed(_get_random_seed(generator)) for subset in ["train", "val", "test"]
        }

        train_filenames, val_filenames, test_filenames = self.prepare_distillation_dataset()

        super().__init__(
            train_ds=DiffusionDistillationDataset(self.save_path / "train", train_filenames),
            val_ds=DiffusionDistillationDataset(self.save_path / "val", val_filenames),
            test_ds=DiffusionDistillationDataset(self.save_path / "test", test_filenames),
            collate_fn=DiffusionDistillationDataset.collate_fn,
            dataloader_args={},
        )

    def prepare_distillation_dataset(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Prepare the distillation data.

        Returns
        -------
        Tuple[List[str], List[str], List[str]]
            The filenames of the train, val and test datasets.
        """
        if self.pipeline is None or self.collection_helper is None:
            # this can happen because those attributes are set to None at the end of this method
            raise ValueError("prepare_distillation_dataset() can only be called once.")

        self.collection_helper.enable()

        # save progress bar state to restore it later
        if hasattr(self.pipeline, "_progress_bar_config"):
            progress_bar_state = dict(self.pipeline._progress_bar_config)
        else:
            progress_bar_state = {}
        self.pipeline.set_progress_bar_config(disable=True)

        self.save_path.mkdir(exist_ok=True, parents=True)
        parameters = {
            "pipeline_kwargs": self.pipeline_kwargs,
            "seed": self.seed,
        }
        with Path(self.save_path / "parameters.json").open("w") as f:
            json.dump(parameters, f)

        train_filenames = self._prepare_one_dataset(
            self.caption_datamodule.train_dataloader(generator=self.dataloader_generators["train"]),
            "train",
        )
        val_filenames = self._prepare_one_dataset(
            self.caption_datamodule.val_dataloader(generator=self.dataloader_generators["val"]),
            "val",
        )
        test_filenames = self._prepare_one_dataset(
            self.caption_datamodule.test_dataloader(generator=self.dataloader_generators["test"]),
            "test",
        )
        self.pipeline.set_progress_bar_config(**progress_bar_state)
        self.collection_helper.disable()

        # pipeline should not be needed by this module anymore, so we drop the reference so it can be deleted elsewhere
        self.collection_helper = None
        self.pipeline = None
        return train_filenames, val_filenames, test_filenames

    def _prepare_one_dataset(
        self,
        dataloader: Optional[DataLoader] | None,
        subdir_name: str,
    ) -> List[str]:
        """
        Setup a single dataset and save it to the path.

        Parameters
        ----------
        dataloader : Optional[DataLoader]
            The dataloader to use to prepare the dataset.
        subdir_name : str
            The name of the subdirectory to save the dataset to, in ["train", "val", "test"].

        Returns
        -------
        List[str]
            The filenames of the dataset.
        """
        if dataloader is None:
            pruna_logger.warning(f"Missing dataloader for {subdir_name} data")
            return None
        Path(self.save_path / subdir_name).mkdir(exist_ok=True, parents=True)
        desc = f"Prepare {subdir_name} distillation dataset"
        filenames: List[str] = []

        for batch in tqdm(dataloader, desc=desc):
            captions = batch if isinstance(batch[0], str) else batch[0]
            for caption in captions:
                filename = f"{len(filenames)}.pt"
                self._prepare_one_sample(filename, caption, subdir_name)
                filenames.append(filename)
        return filenames

    @torch.no_grad()
    def _prepare_one_sample(self, filename: str, caption: str, subdir_name: str) -> None:
        """
        Prepare a single sample and save it to the path.

        Parameters
        ----------
        filename : str
            The filename of the sample.
        caption : str
            The caption of the sample.
        subdir_name : str
            The name of the subdirectory to save the sample to, in ["train", "val", "test"].
        """
        assert (
            self.pipeline is not None and self.collection_helper is not None
        ), "prepare_one_sample() can only be called once."

        seed = _get_random_seed(self.seed_making_generators[subdir_name])
        filepath = self.save_path / subdir_name / filename
        if filepath.exists():
            return  # file was generated in a previous run

        self.collection_helper.new_sample()
        self.pipeline(caption, generator=torch.Generator().manual_seed(seed), **self.pipeline_kwargs)
        inputs, outputs = self.collection_helper.get_sample()

        sample = {
            "caption": caption,
            "inputs": inputs,
            "outputs": outputs,
            "seed": seed,
        }
        torch.save(sample, filepath)


class DiffusionDistillationDataset(Dataset):
    """
    Dataset for distilling a diffusion pipeline, containing captions, latent inputs, latent outputs and seeds.

    Parameters
    ----------
    path : Path
        The path to the distillation data.
    filenames : Optional[List[str]]
        The filenames to load from the path. If None, all files in the path will be loaded.
    """

    def __init__(self, path: Path, filenames: Optional[List[str]] = None):
        self.path = path
        if filenames is None:
            self.filenames = [p.name for p in self.path.iterdir()]
        else:
            self.filenames = filenames

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, torch.Tensor, int]:
        """Get an item from the dataset."""
        filepath = self.path / self.filenames[idx]
        # This is the most generic way to load the data, but may cause a bottleneck because of continuous disk access
        # Loading the whole dataset into memory is often possible given the typically small size of distillation datasets
        # This can be explored if this is identified as a causing a latency bottleneck
        sample = torch.load(filepath)
        return sample["caption"], sample["inputs"], sample["outputs"], sample["seed"]

    @staticmethod
    def collate_fn(
        samples: List[Tuple[str, torch.Tensor, torch.Tensor, int]],
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[int]]:
        """
        Collate the samples into a batch.

        Parameters
        ----------
        samples : List[Tuple[str, torch.Tensor, torch.Tensor, int]]
            The samples to collate, composed of a caption, a latent input, a latent output and a seed.

        Returns
        -------
        Tuple[List[str], torch.Tensor, torch.Tensor, List[int]]
            The collated samples.
        """
        captions = [sample[0] for sample in samples]
        inputs = torch.stack([sample[1] for sample in samples], dim=0)
        outputs = torch.stack([sample[2] for sample in samples], dim=0)
        seeds = [sample[3] for sample in samples]
        return captions, inputs, outputs, seeds


class InternalStateCollectionHelper:
    """
    Helper class to collect internal states from the pipeline.

    When enabled, the denoiser's forward will be monkey patched to save its inputs and outputs.
    They can be collected by calling `new_sample`, running the pipeline and then calling `get_sample`.
    The helper can be disabled to restore the original forward.

    Parameters
    ----------
    pipeline : Any
        The pipeline to use as example for distillation.
    """

    def __init__(self, pipeline: Any) -> None:
        denoiser, _ = get_denoiser_attr(pipeline)
        if denoiser is None:
            raise ValueError("Could not find a denoiser in the pipeline.")
        self.denoiser: torch.nn.Module = denoiser
        self.latent_extractor_fn = get_latent_extractor_fn(pipeline)

        self.original_forward = self.denoiser.forward
        self.inputs: List[torch.Tensor] = []
        self.outputs: List[torch.Tensor] = []

    def new_sample(self) -> None:
        """Reset the state of the forward hook before calling the pipeline, must be called before each new sample."""
        self.inputs = []
        self.outputs = []

    def get_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the saved internal states after the pipeline has been called.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The saved inputs and outputs.
        """
        inputs = torch.stack(self.inputs, dim=0)
        outputs = torch.stack(self.outputs, dim=0)
        return inputs, outputs

    def enable(self) -> None:
        """Enable the collection of internal states when running the pipeline."""

        @functools.wraps(self.denoiser.forward)
        def forward(*args, **kwargs):
            latent = self.latent_extractor_fn(args, kwargs)
            self.inputs.append(latent)
            results = self.original_forward(*args, **kwargs)
            output = results["sample"] if ("return_dict" in kwargs and kwargs["return_dict"]) else results[0]
            self.outputs.append(output)
            return results

        self.denoiser.forward = forward

    def disable(self) -> None:
        """Disable the collection of internal states."""
        self.denoiser.forward = self.original_forward


def _get_random_seed(generator: torch.Generator | None = None) -> int:
    """
    Randomly generate a random seed.

    Parameters
    ----------
    generator : torch.Generator | None
        The generator to use to generate the seed. If None, the current rng state will be used.

    Returns
    -------
    int
        The generated seed.
    """
    return int(torch.randint(0, 2**32 - 1, (1,), generator=generator).item())

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
import hashlib
import re
from pathlib import Path
from typing import Any, Callable, Iterable, List

import numpy as np
import torch
from diffusers.utils import export_to_gif, export_to_video, load_image
from diffusers.utils import load_video as diffusers_load_video
from PIL.Image import Image
from torchvision.transforms import ToTensor

from pruna.data.pruna_datamodule import PrunaDataModule
from pruna.data.utils import define_sample_size_for_dataset, stratify_dataset
from pruna.engine.utils import safe_memory_cleanup, set_to_best_available_device
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.result import MetricResult
from pruna.logging.logger import pruna_logger


class VBenchMixin:
    """
    Mixin class for VBench metrics.

    Handles benchmark specific initilizations and artifact saving conventions.
    """

    def create_filename(self, prompt: str, idx: int, file_extension: str, special_str: str = "") -> str:
        """
        Create filename according to VBench formatting conventions.

        Parameters
        ----------
        prompt : str
            The prompt to create the filename from.
        idx : int
            The index of the video. Vbench uses 5 seeds for each prompt.
        file_extension : str
            The file extension to use. Vbench supports mp4 and gif.
        special_str : str
            A special string to add to the filename if you wish to add a specific identifier.

        Returns
        -------
        str
            The filename.
        """
        return create_vbench_file_name(sanitize_prompt(prompt), idx, special_str, file_extension)

    def validate_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Make sure that the video tensor has correct dimensions.

        Parameters
        ----------
        batch : torch.Tensor
            The video tensor.

        Returns
        -------
        torch.Tensor
            The video tensor.
        """
        if batch.ndim == 4:
            return batch.unsqueeze(0)
        elif batch.ndim != 5:
            raise ValueError(f"Batch must be 4 or 5 dimensional video tensor with B,T,C,H,W, got {batch.ndim}")
        return batch


def get_sample_seed(experiment_name: str, prompt: str, index: int) -> int:
    """
    Get a sample seed for a given experiment name, prompt, and index.

    Parameters
    ----------
    experiment_name : str
        The name of the experiment. To replicate the same experiment, use the same experiment name.
    prompt : str
        The prompt to sample from.
    index : int
        The index of the sample. To get different samples from the same prompt, use different indices.

    Returns
    -------
    int
        The seed.
    """
    key = f"{experiment_name}_{prompt}_{index}".encode('utf-8')

    return int(hashlib.sha256(key).hexdigest(), 16) % (2**32)


def is_file_exists(path: str | Path, filename: str) -> bool:
    """
    Check if a file with the given filename exists in the given path.

    Parameters
    ----------
    path : str | Path
        The path to the folder.
    filename : str
        The name of the file.

    Returns
    -------
    bool
        True if the file exists, False otherwise.
    """
    folder = Path(path)
    full_path = folder / filename

    return full_path.is_file()


def load_video(path: str | Path, return_type: str = "pt") -> List[Image] | np.ndarray | torch.Tensor:
    """
    Load videos from a path.

    Parameters
    ----------
    path : str | Path
        The path to the videos.
    return_type : str
        The type to return the videos as. Can be "pt", "np", "pil".

    Returns
    -------
    List[torch.Tensor]
        The videos.
    """
    video = diffusers_load_video(str(path))
    if return_type == "pt":
        return torch.stack([ToTensor()(frame) for frame in video])
    elif return_type == "np":
        return np.stack([np.array(frame) for frame in video])
    elif return_type == "pil":
        return video
    else:
        raise ValueError(f"Invalid return_type: {return_type}. Use 'pt', 'np', or 'pil'.")


def load_videos_from_path(path: str | Path) -> torch.Tensor:
    """
    Load entire directory of mp4 videos as a single tensor ready to be passed to evaluation.

    Parameters
    ----------
    path : str | Path
        The path to the directory of videos.

    Returns
    -------
    torch.Tensor
        The videos.
    """
    path = Path(str(path))
    videos = torch.stack([load_video(p) for p in path.glob("*.mp4")])
    return videos


def sanitize_prompt(prompt: str) -> str:
    """
    Return a filesystem-safe version of a prompt.

    Replaces characters illegal in filenames and collapses whitespace so that
    generated files are portable across file systems.

    Parameters
    ----------
    prompt : str
        The prompt to sanitize.

    Returns
    -------
    str
        The sanitized prompt.
    """
    prompt = re.sub(r"[\\/:*?\"<>|]", " ", prompt)  # remove illegal chars
    prompt = re.sub(r"\s+", " ", prompt)  # collapse multiple spaces
    prompt = prompt.strip()  # remove leading and trailing whitespace
    return prompt


def resize_to_max_area(image: Image, target_height: int, target_width: int, mod_value: int) -> Image:
    """
    Resize the image to the maximum area.

    Parameters
    ----------
    image : Image
        The image to resize.
    target_height : int
        The target height of the image.
    target_width : int
        The target width of the image.
    mod_value : int
        The modulo value to use for the resizing.

    Returns
    -------
    Image
        The resized image.
    """
    aspect_ratio = image.height / image.width
    target_area = target_height * target_width
    height = round(np.sqrt(target_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(target_area / aspect_ratio)) // mod_value * mod_value
    return image.resize((width, height))


def prepare_batch(batch: str | tuple[str | List[str], Any], task: str = "t2v", target_height: int = 720,
 target_width: int = 1280, mod_value: int = 4) -> tuple(str, Image):
    """
    Prepare the batch to be used in the generate_videos function.

    Pruna datamodules are expected to yield tuples where the first element is
    a sequence of inputs; this utility enforces batch_size == 1 for simplicity.

    Parameters
    ----------
    batch : str | tuple[str | List[str], Any]
        The batch to prepare.

    Returns
    -------
    str
        The prompt string.
    Image
        The resized image.
    """
    if isinstance(batch, str):
        return batch, None
    # for pruna datamodule. always returns a tuple where the first element is the input (list of prompts) to the model.
    elif isinstance(batch, tuple):
        if not hasattr(batch[0], "__len__"):
            raise ValueError(f"Batch[0] is not a sequence (got {type(batch[0])})")
        if len(batch[0]) != 1:
            raise ValueError(f"Only batch size 1 is supported; got {len(batch[0])}")
    else:
        raise ValueError(f"Invalid batch type: {type(batch)}")
    if task == "i2v":
        image = load_image(batch[1])
        image = resize_to_max_area(image, target_height, target_width, mod_value)
        return batch[0][0], image
    elif task == "t2v":
        return batch[0][0], None
    else:
        raise ValueError(f"Invalid task: {task}. Use 'i2v' or 't2v'.")


def _normalize_save_format(save_format: str, save_fn: Callable = None) -> tuple[str, Callable]:
    """
    Normalize the save format to be used in the generate_videos function.

    Parameters
    ----------
    save_format : str
        The format to save the videos in. VBench supports mp4 and gif.

    Returns
    -------
    tuple[str, Callable]
        The normalized save format and the save function.
    """
    
    save_format = save_format.lower().strip()
    if not save_format.startswith("."):
        save_format = "." + save_format
    
    if save_fn is not None:
        return save_format, save_fn
    if save_format == ".mp4":
        return ".mp4", export_to_video
    if save_format == ".gif":
        return ".gif", export_to_gif
    raise ValueError(f"Invalid save_format: {save_format}. Use 'mp4' or 'gif'.")


def _normalize_prompts(
    prompts: str | List[str] | PrunaDataModule, split: str = "test",
    batch_size: int = 1, num_samples: int | None = None, fraction: float = 1.0,
    data_partition_strategy: str = "indexed", partition_index: int = 0, seed: int = 42,
) -> Iterable[str]:
    """
    Normalize prompts to an iterable format to be used in the generate_videos function.

    Parameters
    ----------
    prompts : str | List[str] | PrunaDataModule
        The prompts to normalize.
    split : str
        The dataset split to sample from.
    batch_size : int
        The batch size to sample from.
    num_samples : int | None
        The number of samples to sample from.
    fraction : float
        The fraction of the dataset to sample from.
    data_partition_strategy : str
        The strategy to use for partitioning the dataset. Can be "indexed" or "random".
    partition_index : int
        The index to use for partitioning the dataset.
    seed : int
        The seed to use for partitioning the dataset.

    Returns
    -------
    Iterable[str]
        The normalized prompts.
    """
    if isinstance(prompts, str):
        return [prompts]
    elif isinstance(prompts, PrunaDataModule):
        target_dataset = getattr(prompts, f"{split}_dataset")
        sample_size = define_sample_size_for_dataset(target_dataset, fraction, num_samples)
        setattr(prompts, f"{split}_dataset", stratify_dataset(target_dataset,
        sample_size, seed, data_partition_strategy, partition_index))
        return getattr(prompts, f"{split}_dataloader")(batch_size=batch_size)
    else:  # list of prompts, already iterable
        if num_samples is not None:
            prompts = prompts[:num_samples]
        return prompts


def _ensure_dir(p: Path) -> None:
    """
    Ensure the directory exists.

    Parameters
    ----------
    p : Path
        The path to ensure the directory exists.
    """
    p.mkdir(parents=True, exist_ok=True)


def create_vbench_file_name(
    prompt: str, idx: int, special_str: str = "", save_format: str = ".mp4", max_filename_length: int = 255
) -> str:
    """
    Create a file name for the video in accordance with the VBench format.

    Parameters
    ----------
    prompt : str
        The prompt to create the file name from.
    idx : int
        The index of the video. Vbench uses 5 seeds for each prompt.
    special_str : str
        A special string to add to the file name if you wish to add a specific identifier.
    save_format : str
        The format of the video file. Vbench supports mp4 and gif.
    max_filename_length : int
        The maximum length allowed for the file name.

    Returns
    -------
    str
        The file name for the video.
    """
    filename = f"{prompt}{special_str}-{str(idx)}{save_format}"
    if len(filename) > max_filename_length:
        pruna_logger.debug(
            f"File name {filename} is too long. Maximum length is {max_filename_length} characters. Truncating filename."
        )
        filename = filename[:max_filename_length]
    return filename


def sample_video_from_pipelines(model: Any, seeder: Any, prompt: str, **kwargs):
    """
    Sample a video from diffusers pipeline.

    Parameters
    ----------
    model : Any
        The pipeline to sample from.
    seeder : Any
        The seeding generator.
    prompt : str
        The prompt to sample from.
    **kwargs : Any
        Additional keyword arguments to pass to the pipeline.

    Returns
    -------
    torch.Tensor
        The video tensor.
    """
    is_return_dict = kwargs.pop("return_dict", True)
    with torch.inference_mode():
        if is_return_dict:
            out = model(prompt=prompt, generator=seeder, **kwargs).frames[0]
        else:
            # If return_dict is False, the pipeline returns a tuple of (frames, metadata).
            out = model(prompt=prompt, generator=seeder, **kwargs)[0]

    return out

def seed_rng(x: int) -> torch.Generator:
    """
    Seed the RNG.
    Parameters
    ----------
    x : int
        The seed to seed the RNG with.

    Returns
    -------
    torch.Generator
        The seeded RNG.
    """
    return torch.Generator("cpu").manual_seed(x)

def sample_seed(x: int) -> int:
    """
    Sample a seed.
    Parameters
    ----------
    x : int
        The seed to sample.

    Returns
    -------
    int
        The sampled seed.
    """
    return torch.randint(low=0, high=2**31 -1, generator=seed_rng(x), dtype=torch.int64, size=(1,)).item()


def generate_videos(
    inputs: str | List[str] | PrunaDataModule,
    num_samples: int | None = None,
    model: Any | None = None,
    samples_fraction: float = 1.0,
    data_partition_strategy: str = "indexed",
    partition_index: int = 0,
    split: str = "test",
    unique_sample_per_video_count: int = 1,
    sampling_fn: Callable[..., Any] = sample_video_from_pipelines,
    save_fn: Callable = None,
    save_dir: str | Path = "./saved_videos",
    save_format: str = "mp4",
    filename_fn: Callable = create_vbench_file_name,
    special_str: str = "",
    experiment_name: str = "",
    sampling_seed_fn: Callable[..., Any] = get_sample_seed,
    get_next_seed_fn: Callable[..., Any] = seed_rng,
    pipeline_task: str = "t2v",
    save_kwargs: dict = {},
    **model_kwargs,
) -> None:
    """
    Generate N samples per prompt and save them to disk.


    Parameters
    ----------
    inputs : str | List[str] | List[tuple(str, Any)]| PrunaDataModule
        The inputs to sample from.
    num_samples : int | None
        The number of samples to generate. If unique_sample_per_video_count is greater than 1,
        the total number of outputs will be num_samples * unique_sample_per_video_count.
    model : Any | None
        The model to sample from.
    samples_fraction : float
        The fraction of the dataset to sample from. Only supported for PrunaDataModule.
    data_partition_strategy : str
        The strategy to use for partitioning the dataset. Can be "indexed" or "random". Only supported for PrunaDataModule.
        Indexed means that the dataset will be partitioned and the partition_index will be used to select the partition.
        Random means that the dataset will be shuffled and the first num_samples or samples_fraction will be used.
        Indexed is the default strategy.
    partition_index : int
        The index to use for partitioning the dataset. Only supported for PrunaDataModule.
    split : str
        The split to sample from. Only supported for PrunaDataModule.
        Default is "test" since most benchmarking datamodules in Pruna are configured to use the test split.
    unique_sample_per_video_count : int
        The number of unique samples per video.
    sampling_fn : Callable[..., Any]
        The sampling function to use.
    save_fn : Callable
        The function to save the videos.
    save_dir : str | Path
        The directory to save the videos to.
    save_format : str
        The format to save the videos in. Pruna supports mp4 and gif.
    filename_fn : Callable
        The function to create the file name.
    special_str : str
        A special string to add to the file name if you wish to add a specific identifier.
    experiment_name : str
        The name of the experiment. Used to create the seed.
    sampling_seed_fn : Callable[..., Any]
        The function to create the global seed.
    get_next_seed_fn : Callable[..., Any]
        The function to create the next seed.
    pipeline_task: str
        The task to perform with the pipeline. Can be "i2v" or "t2v".
    save_kwargs : dict
        Additional keyword arguments to pass to the save function.
    **model_kwargs : Any
        Additional keyword arguments to pass to the sampling function.
    """
    file_extension, save_fn = _normalize_save_format(save_format, save_fn)

    prompt_iterable = _normalize_prompts(inputs, split, batch_size=1, num_samples=num_samples,
    fraction=samples_fraction, data_partition_strategy=data_partition_strategy,
    partition_index=partition_index)

    save_dir = Path(save_dir)
    _ensure_dir(save_dir)

    target_height = model_kwargs.get("height",None)
    target_width = model_kwargs.get("width",None)
    if target_height is None or target_width is None:
        if model:
            model_call_params = inspect.signature(getattr(model, "__call__", lambda: None)).parameters
            if "height" in model_call_params:
                target_height = model_call_params["height"].default
            if "width" in model_call_params:
                target_width = model_call_params["width"].default



    mod_value = 1
    if pipeline_task == "i2v" and model:
        vae_scale = getattr(model, "vae_scale_factor_spatial", 1)
        patch_size = getattr(getattr(model, "transformer", None), "config", {}).get("patch_size", [1, 1])[1]
        mod_value = vae_scale * patch_size

    for batch in prompt_iterable:
        prompt, image = prepare_batch(batch, task=pipeline_task,
        target_height=target_height, target_width=target_width, mod_value=mod_value)
        if model and image:
            model_kwargs.update({"height": image.height, "width": image.width})
        sampling_params = model_kwargs.copy()
        sampling_params.update({"model": model, "prompt": prompt, "seeder": None, "image": image})
        for idx in range(unique_sample_per_video_count):
            file_name = filename_fn(sanitize_prompt(prompt), idx, special_str, file_extension)
            out_path = save_dir / file_name

            if is_file_exists(save_dir, file_name):
                continue
            else:
                seed = sampling_seed_fn(experiment_name, prompt, idx)
                seeder = get_next_seed_fn(seed)
                sampling_params["seeder"] = seeder
                vid = sampling_fn(**sampling_params)

                save_fn(vid, out_path, **save_kwargs)

                del vid
                safe_memory_cleanup()


def evaluate_videos(
    data: Any, metrics: StatefulMetric | List[StatefulMetric], prompts: Any | None = None
) -> List[MetricResult]:
    """
    Evaluation loop helper.

    Parameters
    ----------
    data : Any
        The data to evaluate.
    metrics : StatefulMetric | List[StatefulMetric]
        The metrics to evaluate.
    prompts : Any | None
        The prompts to evaluate.

    Returns
    -------
    List[MetricResult]
        The results of the evaluation.
    """
    results = []
    if isinstance(metrics, StatefulMetric):
        metrics = [metrics]
    if any(metric.call_type != "y" for metric in metrics) and prompts is None:
        raise ValueError(
            "You are trying to evaluate metrics that require more than the outputs, but didn't provide prompts."
        )
    for metric in metrics:
        for batch in data:
            if prompts is None:
                prompts = batch
            metric.update(prompts, batch, batch)
            prompts = None
        results.append(metric.compute())
    return results

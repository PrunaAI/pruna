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


def _normalize_save_format(save_format: str) -> tuple[str, Callable]:
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
    if save_format == "mp4":
        return ".mp4", export_to_video
    if save_format == "gif":
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


def sample_video_from_pipelines(pipeline: Any, seeder: Any, prompt: str, **kwargs):
    """
    Sample a video from diffusers pipeline.

    Parameters
    ----------
    pipeline : Any
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
            out = pipeline(prompt=prompt, generator=seeder, **kwargs).frames[0]
        else:
            # If return_dict is False, the pipeline returns a tuple of (frames, metadata).
            out = pipeline(prompt=prompt, generator=seeder, **kwargs)[0]

    return out


def _wrap_sampler(model: Any, sampling_fn: Callable[..., Any]) -> Callable[..., Any]:
    """
    Wrap a user-provided sampling function into a uniform callable.

    The returned callable has a keyword-only signature:
        sampler(*, prompt: str, seeder: Any, device: str|torch.device, **kwargs)

    This wrapper always passes `model` as the first positional argument, so
    custom functions can name their first parameter `model` or `pipeline`, etc.

    Parameters
    ----------
    model : Any
        The model to sample from.
    sampling_fn : Callable[..., Any]
        The sampling function to wrap.

    Returns
    -------
    Callable[..., Any]
        The wrapped sampling function.
    """
    if sampling_fn != sample_video_from_pipelines:
        pruna_logger.info(
            "Using custom sampling function. Ensure it accepts (model, *, prompt, seeder, device, **kwargs)."
        )

    # The sampling function may expect the model as "pipeline" so we pass it as an arg and not a kwarg.
    def sampler(*, prompt: str, seeder: Any, **kwargs: Any) -> Any:
        return sampling_fn(model, prompt=prompt, seeder=seeder, **kwargs)

    return sampler


def generate_videos(
    model: Any,
    inputs: str | List[str] | PrunaDataModule,
    num_samples: int | None = None,
    samples_fraction: float = 1.0,
    data_partition_strategy: str = "indexed",
    partition_index: int = 0,
    split: str = "test",
    unique_sample_per_video_count: int = 1,
    global_seed: int = 42,
    sampling_fn: Callable[..., Any] = sample_video_from_pipelines,
    fps: int = 16,
    save_dir: str | Path = "./saved_videos",
    save_format: str = "mp4",
    filename_fn: Callable = create_vbench_file_name,
    special_str: str = "",
    device: str | torch.device = None,
    experiment_name: str = "",
    sampling_seed_fn: Callable[..., Any] = get_sample_seed,
    pipeline_task: str = "t2v",
    **model_kwargs,
) -> None:
    """
    Generate N samples per prompt and save them to disk with seed tracking.

    This function:
      - Normalizes prompts (string, list, or datamodule).
      - Uses an RNG seeded with `global_seed` for reproducibility across runs.
      - Saves videos as MP4 or GIF.

    Parameters
    ----------
    model : Any
        The model to sample from.
    inputs : str | List[str] | List[tuple(str, Any)]| PrunaDataModule
        The inputs to sample from.
    num_samples : int | None
        The number of samples to generate. If unique_sample_per_video_count is greater than 1,
        the total number of outputs will be num_samples * unique_sample_per_video_count.
    samples_fraction : float
        The fraction of the dataset to sample from.
    data_partition_strategy : str
        The strategy to use for partitioning the dataset. Can be "indexed" or "random".
        Indexed means that the dataset will be partitioned and the partition_index will be used to select the partition.
        Random means that the dataset will be shuffled and the first num_samples or samples_fraction will be used.
        Indexed is the default strategy.
    partition_index : int
        The index to use for partitioning the dataset.
    split : str
        The split to sample from.
        Default is "test" since most benchmarking datamodules in Pruna are configured to use the test split.
    unique_sample_per_video_count : int
        The number of unique samples per video.
    global_seed : int
        The global seed to sample from.
    sampling_fn : Callable[..., Any]
        The sampling function to use.
    fps : int
        The frames per second of the exported video.
    save_dir : str | Path
        The directory to save the videos to.
    save_format : str
        The format to save the videos in. Pruna supports mp4 and gif.
    filename_fn : Callable
        The function to create the file name.
    special_str : str
        A special string to add to the file name if you wish to add a specific identifier.
    device : str | torch.device | None
        The device to sample on. If None, the best available device will be used.
    experiment_name : str
        The name of the experiment. Used to create the seed.
    sampling_seed_fn : Callable[..., Any]
        The function to create the seed.
    pipeline_task: str
        The task to perform with the pipeline. Can be "i2v" or "t2v".
    **model_kwargs : Any
        Additional keyword arguments to pass to the sampling function.
    """
    file_extension, save_fn = _normalize_save_format(save_format)

    device = set_to_best_available_device(device)

    prompt_iterable = _normalize_prompts(inputs, split, batch_size=1, num_samples=num_samples,
    fraction=samples_fraction, data_partition_strategy=data_partition_strategy,
    partition_index=partition_index)

    save_dir = Path(save_dir)
    _ensure_dir(save_dir)

    # set a run-level seed (VBench suggests this) (important for reproducibility)
    def seed_rng(x: int) -> torch.Generator:
        return torch.Generator("cpu").manual_seed(x)
    sampler = _wrap_sampler(model=model, sampling_fn=sampling_fn)

    target_height = model_kwargs.get("target_height", 720)
    target_width = model_kwargs.get("target_width", 1280)
    mod_value = 1
    if pipeline_task == "i2v":
        mod_value = model.vae_scale_factor_spatial * model.transformer.config.patch_size[1]

    for batch in prompt_iterable:
        prompt, image = prepare_batch(batch, task=pipeline_task,
        target_height=target_height, target_width=target_width, mod_value=mod_value)
        for idx in range(unique_sample_per_video_count):
            file_name = filename_fn(sanitize_prompt(prompt), idx, special_str, file_extension)
            out_path = save_dir / file_name

            if is_file_exists(save_dir, file_name):
                continue
            else:
                seed = sampling_seed_fn(experiment_name, prompt, idx)
                if pipeline_task == "i2v":
                    vid = sampler(prompt=prompt, image=image, seeder=seed_rng(seed), **model_kwargs)
                elif pipeline_task == "t2v":
                    vid = sampler(prompt=prompt, seeder=seed_rng(seed), **model_kwargs)
                else:
                    raise ValueError(f"Invalid task: {pipeline_task}. Use 'i2v' or 't2v'.")
                save_fn(vid, out_path, fps=fps)

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

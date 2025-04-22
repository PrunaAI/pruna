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

from typing import Any, Callable, List, Tuple

import torch
from torchvision import transforms
from transformers.tokenization_utils import PreTrainedTokenizer as AutoTokenizer


def image_format_to_transforms(output_format: str, img_size: int) -> transforms.Compose:
    """
    Compose torchvision transforms to convert PIL images to the desired type and range.

    Parameters
    ----------
    output_format : str
        The output format, in ["int", "float", "normalized"].
        With "int", output tensors have integer values between 0 and 255. With "float", they have float values
        between 0 and 1. With "normalized", they have float values between -1 and 1.
    img_size : int
        The size of the image to resize to.

    Returns
    -------
    transforms.Compose
        The composed transforms to convert PIL images to Tensors in the desired format.
    """
    # resize smallest edge to img_size and center crop to get a square image with highest chance of containing the object
    resize = [
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
    ]
    if output_format == "int":
        return transforms.Compose(resize + [transforms.PILToTensor()])
    elif output_format == "float":
        return transforms.Compose(resize + [transforms.ToTensor()])
    elif output_format == "normalized":
        return transforms.Compose(resize + [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    else:
        raise ValueError(f"Invalid output format: {output_format}")


def image_generation_collate(data: Any, img_size: int, output_format: str = "int") -> Tuple[List[str], torch.Tensor]:
    """
    Custom collation function for text-to-image generation datasets.

    Expects a ``image`` column containing PIL images and a ``text`` column containing the clear-textprompt for the image
    generation in the dataset.

    Parameters
    ----------
    data : Any
        The data to collate.
    img_size : int
        The size of the image to resize to.
    output_format : str
        The output format, in ["int", "float", "normalized"].
        With "int", output tensors have integer values between 0 and 255. With "float", they have float values
        between 0 and 1. With "normalized", they have float values between -1 and 1.

    Returns
    -------
    Tuple[torch.Tensor, Any]
        The collated data with size img_size and normalized to [0, 1].
    """
    transformations = image_format_to_transforms(output_format, img_size)
    images, texts = [], []

    for item in data:
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_tensor = transformations(image)
        images.append(image_tensor)
        texts.append(item["text"])

    images_tensor = torch.stack(images)
    return texts, images_tensor


def audio_collate(data: Any) -> Tuple[List[str], List[str]]:
    """
    Custom collation function for audio datasets.

    Expects a ``audio/path`` column containing the path to the audio samples and a ``sentence`` column containing the
    clear-text transcription of the audio samples in the dataset.

    Parameters
    ----------
    data : Any
        The data to collate.

    Returns
    -------
    List[str]
        The collated data.
    """
    return [item["audio"]["path"] for item in data], [item["sentence"] for item in data]


def image_classification_collate(
    data: Any, img_size: int, output_format: str = "int"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collation function for image classification datasets.

    Expects a ``image`` column containing PIL images and a ``label`` column containing the class label in the dataset.

    Parameters
    ----------
    data : Any
        The data to collate.
    img_size : int
        The size of the image to resize to.
    output_format : str
        The output format, in ["int", "float", "normalized"].
        With "int", output tensors have integer values between 0 and 255. With "float", they have float values
        between 0 and 1. With "normalized", they have float values between -1 and 1.

    Returns
    -------
    Tuple[List[str], torch.Tensor]
        The collated data with size img_size and normalized to [0, 1].
    """
    transformations = image_format_to_transforms(output_format, img_size)
    images, labels = [], []

    for item in data:
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_tensor = transformations(image)
        images.append(image_tensor)
        labels.append(item["label"])

    images_tensor = torch.stack(images).float()
    return images_tensor, torch.tensor(labels)


def text_generation_collate(
    data: Any, max_seq_len: int | None, tokenizer: AutoTokenizer
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collation function for text generation datasets.

    Expects a ``text`` column containing clear-text samples in the dataset.

    Parameters
    ----------
    data : Any
        The data to collate.
    max_seq_len : int | None
        The maximum sequence length.
    tokenizer : AutoTokenizer
        The tokenizer to use.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The collated data.
    """
    input_ids = []
    for sample in data:
        input_ids.append(
            tokenizer(
                sample["text"],
                max_length=max_seq_len,
                truncation=True,
                padding="max_length" if max_seq_len else False,
            )["input_ids"]
        )

    return torch.tensor(input_ids)[:, :-1], torch.tensor(input_ids)[:, 1:]


def question_answering_collate(
    data: Any, max_seq_len: int, tokenizer: AutoTokenizer
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collation function for question answering datasets.

    Expects a ``question`` and ``answer`` column containing the clear-text question and answer in the dataset.

    Parameters
    ----------
    data : Any
        The data to collate.
    max_seq_len : int
        The maximum sequence length.
    tokenizer : AutoTokenizer
        The tokenizer to use.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The collated data.
    """
    questions, answers = [], []
    for sample in data:
        questions.append(
            tokenizer(sample["question"], max_length=max_seq_len, truncation=True, padding="max_length")["input_ids"]
        )
        answers.append(
            tokenizer(sample["answer"], max_length=max_seq_len, truncation=True, padding="max_length")["input_ids"]
        )
    return torch.tensor(questions), torch.tensor(answers)


pruna_collate_fns: dict[str, Callable] = {
    "image_generation_collate": image_generation_collate,
    "audio_collate": audio_collate,
    "image_classification_collate": image_classification_collate,
    "text_generation_collate": text_generation_collate,
    "question_answering_collate": question_answering_collate,
}

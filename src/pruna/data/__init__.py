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

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Tuple

from pruna.data.datasets.audio import (
    setup_librispeech_dataset,
    setup_mini_presentation_audio_dataset,
    setup_podcast_dataset,
)
from pruna.data.datasets.image import (
    setup_cifar10_dataset,
    setup_imagenet_dataset,
    setup_mnist_dataset,
)
from pruna.data.datasets.prompt import (
    setup_drawbench_dataset,
    setup_genai_bench_dataset,
    setup_imgedit_dataset,
    setup_parti_prompts_dataset,
)
from pruna.data.datasets.question_answering import setup_polyglot_dataset
from pruna.data.datasets.text_generation import (
    setup_c4_dataset,
    setup_openassistant_dataset,
    setup_pubchem_dataset,
    setup_smolsmoltalk_dataset,
    setup_smoltalk_dataset,
    setup_tiny_imdb_dataset,
    setup_wikitext_dataset,
    setup_wikitext_tiny_dataset,
)
from pruna.data.datasets.text_to_image import (
    setup_coco_dataset,
    setup_laion256_dataset,
    setup_open_image_dataset,
)
from pruna.data.datasets.text_to_video import setup_vbench_dataset

base_datasets: dict[str, Tuple[Callable, str, dict[str, Any]]] = {
    "COCO": (setup_coco_dataset, "image_generation_collate", {"img_size": 512}),
    "LAION256": (setup_laion256_dataset, "image_generation_collate", {"img_size": 512}),
    "LibriSpeech": (setup_librispeech_dataset, "audio_collate", {}),
    "AIPodcast": (setup_podcast_dataset, "audio_collate", {}),
    "MiniPresentation": (setup_mini_presentation_audio_dataset, "audio_collate", {}),
    "ImageNet": (
        setup_imagenet_dataset,
        "image_classification_collate",
        {"img_size": 224},
    ),
    "MNIST": (setup_mnist_dataset, "image_classification_collate", {"img_size": 28}),
    "WikiText": (setup_wikitext_dataset, "text_generation_collate", {}),
    "TinyWikiText": (setup_wikitext_tiny_dataset, "text_generation_collate", {}),
    "SmolTalk": (setup_smoltalk_dataset, "text_generation_collate", {}),
    "SmolSmolTalk": (setup_smolsmoltalk_dataset, "text_generation_collate", {}),
    "PubChem": (setup_pubchem_dataset, "text_generation_collate", {}),
    "OpenAssistant": (setup_openassistant_dataset, "text_generation_collate", {}),
    "C4": (setup_c4_dataset, "text_generation_collate", {}),
    "Polyglot": (setup_polyglot_dataset, "question_answering_collate", {}),
    "OpenImage": (
        setup_open_image_dataset,
        "image_generation_collate",
        {"img_size": 1024},
    ),
    "CIFAR10": (
        setup_cifar10_dataset,
        "image_classification_collate",
        {"img_size": 32},
    ),
    # our full CIFAR10 has 50k train and 10k test
    "TinyCIFAR10": (
        partial(setup_cifar10_dataset, train_sample_size=800, test_sample_size=100),
        "image_classification_collate",
        {"img_size": 32},
    ),
    #  our full MNIST has 60k train and 10k test
    "TinyMNIST": (
        partial(setup_mnist_dataset, train_sample_size=800, test_sample_size=100),
        "image_classification_collate",
        {"img_size": 28},
    ),
    # our full ImageNet has 100k train and 10k val
    "TinyImageNet": (
        partial(setup_imagenet_dataset, train_sample_size=1000, test_sample_size=100),
        "image_classification_collate",
        {"img_size": 224},
    ),
    "DrawBench": (setup_drawbench_dataset, "prompt_collate", {}),
    "PartiPrompts": (setup_parti_prompts_dataset, "prompt_with_auxiliaries_collate", {}),
    "GenAIBench": (setup_genai_bench_dataset, "prompt_collate", {}),
    "ImgEdit": (setup_imgedit_dataset, "prompt_with_auxiliaries_collate", {}),
    "TinyIMDB": (setup_tiny_imdb_dataset, "text_generation_collate", {}),
    "VBench": (setup_vbench_dataset, "prompt_with_auxiliaries_collate", {}),
}


@dataclass
class BenchmarkInfo:
    """
    Metadata for a benchmark dataset.

    Parameters
    ----------
    name : str
        Internal identifier for the benchmark.
    display_name : str
        Human-readable name for display purposes.
    description : str
        Description of what the benchmark evaluates.
    metrics : list[str]
        List of metric names used for evaluation.
    task_type : str
        Type of task the benchmark evaluates (e.g., 'text_to_image').
    subsets : list[str]
        Optional list of benchmark subset names.
    """

    name: str
    display_name: str
    description: str
    metrics: list[str]
    task_type: str
    subsets: list[str] = field(default_factory=list)


benchmark_info: dict[str, BenchmarkInfo] = {
    "PartiPrompts": BenchmarkInfo(
        name="parti_prompts",
        display_name="Parti Prompts",
        description=(
            "Over 1,600 diverse English prompts across 12 categories with 11 challenge aspects "
            "ranging from basic to complex, enabling comprehensive assessment of model capabilities "
            "across different domains and difficulty levels."
        ),
        metrics=["arniqa", "clip_score", "clipiqa", "sharpness"],
        task_type="text_to_image",
        subsets=[
            "Abstract",
            "Animals",
            "Artifacts",
            "Arts",
            "Food & Beverage",
            "Illustrations",
            "Indoor Scenes",
            "Outdoor Scenes",
            "People",
            "Produce & Plants",
            "Vehicles",
            "World Knowledge",
            "Basic",
            "Complex",
            "Fine-grained Detail",
            "Imagination",
            "Linguistic Structures",
            "Perspective",
            "Properties & Positioning",
            "Quantity",
            "Simple Detail",
            "Style & Format",
            "Writing & Symbols",
        ],
    ),
    "DrawBench": BenchmarkInfo(
        name="drawbench",
        display_name="DrawBench",
        description="A comprehensive benchmark for evaluating text-to-image generation models.",
        metrics=["clip_score", "clipiqa", "sharpness"],
        task_type="text_to_image",
    ),
    "GenAIBench": BenchmarkInfo(
        name="genai_bench",
        display_name="GenAI Bench",
        description="A benchmark for evaluating generative AI models.",
        metrics=["clip_score", "clipiqa", "sharpness"],
        task_type="text_to_image",
    ),
    "VBench": BenchmarkInfo(
        name="vbench",
        display_name="VBench",
        description="A benchmark for evaluating video generation models.",
        metrics=["clip_score"],
        task_type="text_to_video",
    ),
    "COCO": BenchmarkInfo(
        name="coco",
        display_name="COCO",
        description="Microsoft COCO dataset for image generation evaluation with real image-caption pairs.",
        metrics=["fid", "clip_score", "clipiqa"],
        task_type="text_to_image",
    ),
    "ImageNet": BenchmarkInfo(
        name="imagenet",
        display_name="ImageNet",
        description="Large-scale image classification benchmark with 1000 classes.",
        metrics=["accuracy"],
        task_type="image_classification",
    ),
    "WikiText": BenchmarkInfo(
        name="wikitext",
        display_name="WikiText",
        description="Language modeling benchmark based on Wikipedia articles.",
        metrics=["perplexity"],
        task_type="text_generation",
    ),
    "ImgEdit": BenchmarkInfo(
        name="imgedit",
        display_name="ImgEdit",
        description="Comprehensive image editing benchmark with 8 edit types: replace, add, remove, adjust, extract, style, background, compose.",
        metrics=["accuracy"],
        task_type="image_edit",
        subsets=["replace", "add", "remove", "adjust", "extract", "style", "background", "compose"],
    ),
}


def list_benchmarks(task_type: str | None = None) -> list[str]:
    """
    List available benchmark names.

    Parameters
    ----------
    task_type : str | None
        Filter by task type (e.g., 'text_to_image', 'text_to_video').
        If None, returns all benchmarks.

    Returns
    -------
    list[str]
        List of benchmark names.
    """
    if task_type is None:
        return list(benchmark_info.keys())
    return [name for name, info in benchmark_info.items() if info.task_type == task_type]


def get_benchmark_info(name: str) -> BenchmarkInfo:
    """
    Get benchmark metadata by name.

    Parameters
    ----------
    name : str
        The benchmark name.

    Returns
    -------
    BenchmarkInfo
        The benchmark metadata.

    Raises
    ------
    KeyError
        If benchmark name is not found.
    """
    if name not in benchmark_info:
        available = ", ".join(benchmark_info.keys())
        raise KeyError(f"Benchmark '{name}' not found. Available: {available}")
    return benchmark_info[name]

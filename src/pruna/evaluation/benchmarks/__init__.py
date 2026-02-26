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

from pruna.data import base_datasets
from pruna.evaluation.metrics import MetricRegistry


@dataclass
class Benchmark:
    """
    Metadata for a benchmark dataset.

    Parameters
    ----------
    name : str
        Human-readable name for display and base_datasets lookup.
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
    description: str
    metrics: list[str]
    task_type: str
    subsets: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Populate subsets from base_datasets when a matching lookup key exists."""
        lookup_key = self.name.replace(" ", "")
        if lookup_key in base_datasets:
            self.subsets = base_datasets[lookup_key][3]


class BenchmarkRegistry:
    """
    Registry for benchmarks.

    Provides lookup and discovery of benchmark metadata.
    """

    _registry: list[Benchmark] = [
        Benchmark(
            name="Parti Prompts",
            description=(
                "Over 1,600 diverse English prompts across 12 categories with 11 challenge aspects "
                "ranging from basic to complex, enabling comprehensive assessment of model capabilities "
                "across different domains and difficulty levels."
            ),
            metrics=["arniqa", "clip_score", "clipiqa", "sharpness"],
            task_type="text_to_image",
        ),
        Benchmark(
            name="DrawBench",
            description="A comprehensive benchmark for evaluating text-to-image generation models.",
            metrics=["clip_score", "clipiqa", "sharpness"],
            task_type="text_to_image",
        ),
        Benchmark(
            name="GenAI Bench",
            description="A benchmark for evaluating generative AI models.",
            metrics=["clip_score", "clipiqa", "sharpness"],
            task_type="text_to_image",
        ),
        Benchmark(
            name="VBench",
            description="A benchmark for evaluating video generation models.",
            metrics=["clip_score"],
            task_type="text_to_video",
        ),
        Benchmark(
            name="COCO",
            description="Microsoft COCO dataset for image generation evaluation with real image-caption pairs.",
            metrics=["fid", "clip_score", "clipiqa"],
            task_type="text_to_image",
        ),
        Benchmark(
            name="ImageNet",
            description="Large-scale image classification benchmark with 1000 classes.",
            metrics=["accuracy"],
            task_type="image_classification",
        ),
        Benchmark(
            name="WikiText",
            description="Language modeling benchmark based on Wikipedia articles.",
            metrics=["perplexity"],
            task_type="text_generation",
        ),
    ]

    @classmethod
    def _validate(cls, name: str, benchmark: Benchmark) -> None:
        missing = [m for m in benchmark.metrics if m not in MetricRegistry._registry]
        if missing:
            raise ValueError(
                f"Benchmark '{name}' references metrics not in MetricRegistry: {missing}. "
                f"Available metrics: {list(MetricRegistry._registry.keys())}"
            )
        lookup_key = benchmark.name.replace(" ", "")
        if lookup_key not in base_datasets:
            available = ", ".join(base_datasets.keys())
            raise ValueError(
                f"Benchmark '{name}' name '{benchmark.name}' does not align with base_datasets. "
                f"Expected lookup key '{lookup_key}'. Available: {available}"
            )

    @classmethod
    def get(cls, name: str) -> Benchmark:
        """
        Get benchmark metadata by name.

        Parameters
        ----------
        name : str
            The benchmark name.

        Returns
        -------
        Benchmark
            The benchmark metadata.

        Raises
        ------
        KeyError
            If benchmark name is not found.
        ValueError
            If a benchmark metric is not registered in MetricRegistry.
        ValueError
            If name does not align with a base_datasets key.
        """
        lookup_key = name.replace(" ", "")
        for benchmark in cls._registry:
            if benchmark.name.replace(" ", "") == lookup_key:
                cls._validate(lookup_key, benchmark)
                return benchmark
        available = ", ".join(b.name.replace(" ", "") for b in cls._registry)
        raise KeyError(f"Benchmark '{name}' not found. Available: {available}")

    @classmethod
    def list(cls, task_type: str | None = None) -> list[str]:
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
            return [b.name.replace(" ", "") for b in cls._registry]
        return [b.name.replace(" ", "") for b in cls._registry if b.task_type == task_type]


for _benchmark in BenchmarkRegistry._registry:
    BenchmarkRegistry._validate(_benchmark.name.replace(" ", ""), _benchmark)

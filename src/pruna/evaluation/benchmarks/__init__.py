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

from dataclasses import dataclass

from pruna.data import base_datasets
from pruna.data.utils import get_literal_values_from_param
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
    category : str | list[str] | None
        Category filter: valid values derived from the setup function's Literal (list),
        or None when the benchmark has no category parameter.
    reference : str | None
        URL to the canonical paper (e.g., arXiv) for this benchmark.
    """

    name: str
    description: str
    metrics: list[str]
    task_type: str
    category: str | list[str] | None = None
    reference: str | None = None

    @property
    def lookup_key(self) -> str:
        """Key for base_datasets lookup (name with spaces removed)."""
        return self.name.replace(" ", "")

    def __post_init__(self) -> None:
        """Populate category from setup function's Literal."""
        if self.lookup_key in base_datasets:
            setup_fn = base_datasets[self.lookup_key][0]
            literal_values = get_literal_values_from_param(setup_fn, "category")
            self.category = literal_values if literal_values else None


class BenchmarkRegistry:
    """
    Registry for benchmarks.

    Provides lookup and discovery of benchmark metadata. Metrics per benchmark
    are set to those explicitly used in the reference paper (see reference URL).
    Verified from full text / evaluation sections where available:
    - Parti/DrawBench: human evaluation only (abstract + HF summary).
    - GenAI Bench: VQAScore only (2406.13743; not in Pruna).
    - VBench: 16 dimensions, each with tailored method (DINO, CLIP, ViCLIP, LAION, MUSIQ, GRiT, etc.; 2311.17982 ar5iv).
    - COCO: FID + CLIP for image-text alignment (Imagen 2205.11487).
    - ImageNet: top-1/top-5 accuracy (1409.0575). WikiText: perplexity (1609.07843).
    - GenEval: Mask2Former + CLIP color pipeline, binary GenEval score (2310.11513 arxiv HTML).
    - HPS/ImgEdit/LongText/GEditBench/OneIG/DPG: paper-specific metrics not in Pruna.
    """

    _registry: list[Benchmark] = [
        Benchmark(
            name="Parti Prompts",
            description=(
                "Holistic benchmark from Google Research with over 1,600 English prompts across 12 categories "
                "and 11 challenge aspects. Evaluates text-to-image models on abstract thinking, world knowledge, "
                "perspectives, and symbol rendering from basic to complex compositions."
            ),
            metrics=[],  # Paper uses human evaluation only; pass explicit metrics if needed
            task_type="text_to_image",
            reference="https://arxiv.org/abs/2206.10789",
        ),
        Benchmark(
            name="DrawBench",
            description=(
                "Comprehensive benchmark from the Imagen team for rigorous evaluation of text-to-image models. "
                "Enables side-by-side comparison on sample quality and image-text alignment with human raters."
            ),
            metrics=[],  # Paper uses human evaluation only; pass explicit metrics if needed
            task_type="text_to_image",
            reference="https://arxiv.org/abs/2205.11487",
        ),
        Benchmark(
            name="GenAI Bench",
            description=(
                "1,600 prompts from professional designers for compositional text-to-visual generation. "
                "Covers basic skills (scene, attributes, spatial relationships) to advanced reasoning "
                "(counting, comparison, logic/negation) with over 24k human ratings."
            ),
            metrics=[],  # Paper uses VQAScore only; not in Pruna
            task_type="text_to_image",
            reference="https://arxiv.org/abs/2406.13743",
        ),
        Benchmark(
            name="VBench",
            description=(
                "Comprehensive benchmark suite for video generative models. Decomposes video quality into "
                "16 disentangled dimensions: temporal flickering, motion smoothness, subject consistency, "
                "spatial relationship, color, aesthetic quality, and more."
            ),
            metrics=[],  # Paper uses dimension-specific automated metrics; not all in Pruna
            task_type="text_to_video",
            reference="https://arxiv.org/abs/2311.17982",
        ),
        Benchmark(
            name="COCO",
            description=(
                "Microsoft COCO dataset for image generation evaluation. Real image-caption pairs "
                "enabling FID and alignment metrics on distribution-level and instance-level quality."
            ),
            metrics=["fid", "clip_score"],  # Paper: FID + CLIP VIT-L/14 for alignment
            task_type="text_to_image",
            reference="https://arxiv.org/abs/2205.11487",
        ),
        Benchmark(
            name="ImageNet",
            description=(
                "Large-scale image classification benchmark with 1,000 classes. Standard evaluation "
                "for vision model accuracy on object recognition."
            ),
            metrics=["accuracy"],
            task_type="image_classification",
            reference="https://arxiv.org/abs/1409.0575",
        ),
        Benchmark(
            name="WikiText",
            description=(
                "Language modeling benchmark based on Wikipedia articles. Standard evaluation "
                "for text generation quality via perplexity."
            ),
            metrics=["perplexity"],
            task_type="text_generation",
            reference="https://arxiv.org/abs/1609.07843",
        ),
        Benchmark(
            name="GenEval",
            description=(
                "Compositional text-to-image benchmark with 6 categories: single object, two object, "
                "counting, colors, position, color attributes. Evaluates fine-grained alignment "
                "between prompts and generated images via VQA-style questions."
            ),
            metrics=[],  # Paper: Mask2Former + CLIP ViT-L/14 pipeline; not in Pruna
            task_type="text_to_image",
            reference="https://arxiv.org/abs/2310.11513",
        ),
        Benchmark(
            name="HPS",
            description=(
                "HPD (Human Preference Dataset) v2 for HPS (Human Preference Score) evaluation. "
                "Covers anime, concept-art, paintings, and photo styles with human preference data."
            ),
            metrics=[],  # Paper uses HPS scoring model; not in Pruna
            task_type="text_to_image",
            reference="https://arxiv.org/abs/2306.09341",
        ),
        Benchmark(
            name="ImgEdit",
            description=(
                "Image editing benchmark with 8 edit types: replace, add, remove, adjust, extract, "
                "style, background, compose. Evaluates instruction-following for inpainting and editing."
            ),
            metrics=[],  # Paper uses GPT-4o/ImgEdit-Judge; not in Pruna
            task_type="text_to_image",
            reference="https://arxiv.org/abs/2505.20275",
        ),
        Benchmark(
            name="Long Text Bench",
            description=(
                "Text-to-image benchmark for long, detailed prompts. Evaluates model ability to "
                "handle complex multi-clause descriptions and maintain coherence across long instructions."
            ),
            metrics=[],  # Paper uses text_score/TIT-Score; not in Pruna
            task_type="text_to_image",
            reference="https://arxiv.org/abs/2507.22058",
        ),
        Benchmark(
            name="GEditBench",
            description=(
                "General image editing benchmark with 11 task types: background change, color alter, "
                "material alter, motion change, style change, subject add/remove/replace, text change, "
                "tone transfer, and human retouching."
            ),
            metrics=[],  # Paper uses VIEScore; not in Pruna
            task_type="text_to_image",
            reference="https://arxiv.org/abs/2504.17761",
        ),
        Benchmark(
            name="OneIG",
            description=(
                "Omni-dimensional benchmark for text-to-image evaluation. Six dataset categories "
                "(Anime_Stylization, General_Object, Knowledge_Reasoning, Multilingualism, Portrait, "
                "Text_Rendering) plus fine-grained style classes. Includes alignment questions."
            ),
            metrics=[],  # Paper uses dimension-specific metrics; not in Pruna
            task_type="text_to_image",
            reference="https://arxiv.org/abs/2506.07977",
        ),
        Benchmark(
            name="DPG",
            description=(
                "Dense Prompt Graph benchmark. Evaluates entity, attribute, relation, "
                "global, and other descriptive aspects with natural-language questions for alignment."
            ),
            metrics=[],  # Paper uses custom evaluation; not in Pruna
            task_type="text_to_image",
            reference="https://arxiv.org/abs/2403.05135",
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
        if benchmark.lookup_key not in base_datasets:
            available = ", ".join(base_datasets.keys())
            raise ValueError(
                f"Benchmark '{benchmark.name}' (lookup key '{benchmark.lookup_key}') is not in base_datasets. "
                f"Available: {available}"
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
            if benchmark.lookup_key == lookup_key:
                cls._validate(lookup_key, benchmark)
                return benchmark
        available = ", ".join(b.lookup_key for b in cls._registry)
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
            return [b.lookup_key for b in cls._registry]
        return [b.lookup_key for b in cls._registry if b.task_type == task_type]


for _benchmark in BenchmarkRegistry._registry:
    BenchmarkRegistry._validate(_benchmark.lookup_key, _benchmark)

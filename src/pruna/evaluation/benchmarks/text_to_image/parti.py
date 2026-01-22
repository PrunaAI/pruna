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

from typing import Iterator, List, cast

from datasets import Dataset, load_dataset

from pruna.evaluation.benchmarks.base import TASK, Benchmark, BenchmarkEntry


class PartiPrompts(Benchmark):
    """Parti Prompts benchmark for text-to-image generation."""

    def __init__(
        self,
        seed: int = 42,
        num_samples: int | None = None,
        subset: str | None = None,
    ):
        """
        Initialize the Parti Prompts benchmark.

        Parameters
        ----------
        seed : int
            Random seed for shuffling. Default is 42.
        num_samples : int | None
            Number of samples to select. If None, uses all samples. Default is None.
            subset : str | None
            Filter by a subset of the dataset. For PartiPrompts, this can be either:

            **Categories:**
            - "Abstract"
            - "Animals"
            - "Artifacts"
            - "Arts"
            - "Food & Beverage"
            - "Illustrations"
            - "Indoor Scenes"
            - "Outdoor Scenes"
            - "People"
            - "Produce & Plants"
            - "Vehicles"
            - "World Knowledge"

            **Challenges:**
            - "Basic"
            - "Complex"
            - "Fine-grained Detail"
            - "Imagination"
            - "Linguistic Structures"
            - "Perspective"
            - "Properties & Positioning"
            - "Quantity"
            - "Simple Detail"
            - "Style & Format"
            - "Writing & Symbols"

            If None, includes all samples. Default is None.
        """
        super().__init__()
        self._seed = seed
        self._num_samples = num_samples

        # Determine if subset refers to a dataset category or challenge
        # Check against known challenges
        self.subset = subset

    def _load_prompts(self) -> List[dict]:
        """Load prompts from the dataset."""
        dataset_dict = load_dataset("nateraw/parti-prompts")  # type: ignore
        dataset = cast(Dataset, dataset_dict["train"])  # type: ignore
        if self.subset is not None:
            dataset = dataset.filter(lambda x: x["Category"] == self.subset or x["Challenge"] == self.subset)
        shuffled_dataset = dataset.shuffle(seed=self._seed)
        if self._num_samples is not None:
            selected_dataset = shuffled_dataset.select(range(min(self._num_samples, len(shuffled_dataset))))
        else:
            selected_dataset = shuffled_dataset
        return list(selected_dataset)

    def __iter__(self) -> Iterator[BenchmarkEntry]:
        """Iterate over benchmark entries."""
        for i, row in enumerate(self._load_prompts()):
            yield BenchmarkEntry(
                model_inputs={"prompt": row["Prompt"]},
                model_outputs={},
                path=f"{i}.png",
                additional_info={
                    "category": row["Category"],
                    "challenge": row["Challenge"],
                    "note": row.get("Note", ""),
                },
                task_type=self.task_type,
            )

    @property
    def name(self) -> str:
        """Return the unique name identifier."""
        if self.subset is None:
            return "parti_prompts"
        normalized = (
            self.subset.lower().replace(" & ", "_").replace(" ", "_").replace("&", "_").replace("__", "_").rstrip("_")
        )
        return f"parti_prompts_{normalized}"

    @property
    def display_name(self) -> str:
        """Return the human-readable display name."""
        if self.subset is None:
            return "Parti Prompts"
        return f"Parti Prompts ({self.subset})"

    def __len__(self) -> int:
        """Return the number of entries in the benchmark."""
        return len(self._load_prompts())

    @property
    def metrics(self) -> List[str]:
        """Return the list of recommended metrics."""
        return ["arniqa", "clip", "clip_iqa", "sharpness"]

    @property
    def task_type(self) -> TASK:
        """Return the task type."""
        return "text_to_image"

    @property
    def description(self) -> str:
        """Return a description of the benchmark."""
        return (
            "Over 1,600 diverse English prompts across 12 categories with 11 challenge aspects "
            "ranging from basic to complex, enabling comprehensive assessment of model capabilities "
            "across different domains and difficulty levels."
        )


# Category-based subclasses
class PartiPromptsAbstract(PartiPrompts):
    """Parti Prompts filtered by Abstract category."""

    def __init__(self, seed: int = 42, num_samples: int | None = None, subset: str | None = None):
        super().__init__(seed=seed, num_samples=num_samples, subset="Abstract" if subset is None else subset)


class PartiPromptsAnimals(PartiPrompts):
    """Parti Prompts filtered by Animals category."""

    def __init__(self, seed: int = 42, num_samples: int | None = None, subset: str | None = None):
        super().__init__(seed=seed, num_samples=num_samples, subset="Animals" if subset is None else subset)


class PartiPromptsArtifacts(PartiPrompts):
    """Parti Prompts filtered by Artifacts category."""

    def __init__(self, seed: int = 42, num_samples: int | None = None, subset: str | None = None):
        super().__init__(seed=seed, num_samples=num_samples, subset="Artifacts" if subset is None else subset)


class PartiPromptsArts(PartiPrompts):
    """Parti Prompts filtered by Arts category."""

    def __init__(self, seed: int = 42, num_samples: int | None = None, subset: str | None = None):
        super().__init__(seed=seed, num_samples=num_samples, subset="Arts" if subset is None else subset)


class PartiPromptsFoodBeverage(PartiPrompts):
    """Parti Prompts filtered by Food & Beverage category."""

    def __init__(self, seed: int = 42, num_samples: int | None = None, subset: str | None = None):
        super().__init__(seed=seed, num_samples=num_samples, subset="Food & Beverage" if subset is None else subset)


class PartiPromptsIllustrations(PartiPrompts):
    """Parti Prompts filtered by Illustrations category."""

    def __init__(self, seed: int = 42, num_samples: int | None = None, subset: str | None = None):
        super().__init__(seed=seed, num_samples=num_samples, subset="Illustrations" if subset is None else subset)


class PartiPromptsIndoorScenes(PartiPrompts):
    """Parti Prompts filtered by Indoor Scenes category."""

    def __init__(self, seed: int = 42, num_samples: int | None = None, subset: str | None = None):
        super().__init__(seed=seed, num_samples=num_samples, subset="Indoor Scenes" if subset is None else subset)


class PartiPromptsOutdoorScenes(PartiPrompts):
    """Parti Prompts filtered by Outdoor Scenes category."""

    def __init__(self, seed: int = 42, num_samples: int | None = None, subset: str | None = None):
        super().__init__(seed=seed, num_samples=num_samples, subset="Outdoor Scenes" if subset is None else subset)


class PartiPromptsPeople(PartiPrompts):
    """Parti Prompts filtered by People category."""

    def __init__(self, seed: int = 42, num_samples: int | None = None, subset: str | None = None):
        super().__init__(seed=seed, num_samples=num_samples, subset="People" if subset is None else subset)


class PartiPromptsProducePlants(PartiPrompts):
    """Parti Prompts filtered by Produce & Plants category."""

    def __init__(self, seed: int = 42, num_samples: int | None = None, subset: str | None = None):
        super().__init__(seed=seed, num_samples=num_samples, subset="Produce & Plants" if subset is None else subset)


class PartiPromptsVehicles(PartiPrompts):
    """Parti Prompts filtered by Vehicles category."""

    def __init__(self, seed: int = 42, num_samples: int | None = None, subset: str | None = None):
        super().__init__(seed=seed, num_samples=num_samples, subset="Vehicles" if subset is None else subset)


class PartiPromptsWorldKnowledge(PartiPrompts):
    """Parti Prompts filtered by World Knowledge category."""

    def __init__(self, seed: int = 42, num_samples: int | None = None, subset: str | None = None):
        super().__init__(seed=seed, num_samples=num_samples, subset="World Knowledge" if subset is None else subset)


# Challenge-based subclasses
class PartiPromptsBasic(PartiPrompts):
    """Parti Prompts filtered by Basic challenge."""

    def __init__(self, seed: int = 42, num_samples: int | None = None, subset: str | None = None):
        # subset can be a category to further filter when challenge is already set
        super().__init__(seed=seed, num_samples=num_samples, subset="Basic" if subset is None else subset)


class PartiPromptsComplex(PartiPrompts):
    """Parti Prompts filtered by Complex challenge."""

    def __init__(self, seed: int = 42, num_samples: int | None = None, subset: str | None = None):
        super().__init__(seed=seed, num_samples=num_samples, subset="Complex" if subset is None else subset)


class PartiPromptsFineGrainedDetail(PartiPrompts):
    """Parti Prompts filtered by Fine-grained Detail challenge."""

    def __init__(self, seed: int = 42, num_samples: int | None = None, subset: str | None = None):
        super().__init__(seed=seed, num_samples=num_samples, subset="Fine-grained Detail" if subset is None else subset)


class PartiPromptsImagination(PartiPrompts):
    """Parti Prompts filtered by Imagination challenge."""

    def __init__(self, seed: int = 42, num_samples: int | None = None, subset: str | None = None):
        super().__init__(seed=seed, num_samples=num_samples, subset="Imagination" if subset is None else subset)


class PartiPromptsLinguisticStructures(PartiPrompts):
    """Parti Prompts filtered by Linguistic Structures challenge."""

    def __init__(self, seed: int = 42, num_samples: int | None = None, subset: str | None = None):
        super().__init__(
            seed=seed, num_samples=num_samples, subset="Linguistic Structures" if subset is None else subset
        )


class PartiPromptsPerspective(PartiPrompts):
    """Parti Prompts filtered by Perspective challenge."""

    def __init__(self, seed: int = 42, num_samples: int | None = None, subset: str | None = None):
        super().__init__(seed=seed, num_samples=num_samples, subset="Perspective" if subset is None else subset)


class PartiPromptsPropertiesPositioning(PartiPrompts):
    """Parti Prompts filtered by Properties & Positioning challenge."""

    def __init__(self, seed: int = 42, num_samples: int | None = None, subset: str | None = None):
        super().__init__(
            seed=seed, num_samples=num_samples, subset="Properties & Positioning" if subset is None else subset
        )


class PartiPromptsQuantity(PartiPrompts):
    """Parti Prompts filtered by Quantity challenge."""

    def __init__(self, seed: int = 42, num_samples: int | None = None, subset: str | None = None):
        super().__init__(seed=seed, num_samples=num_samples, subset="Quantity" if subset is None else subset)


class PartiPromptsSimpleDetail(PartiPrompts):
    """Parti Prompts filtered by Simple Detail challenge."""

    def __init__(self, seed: int = 42, num_samples: int | None = None, subset: str | None = None):
        super().__init__(seed=seed, num_samples=num_samples, subset="Simple Detail" if subset is None else subset)


class PartiPromptsStyleFormat(PartiPrompts):
    """Parti Prompts filtered by Style & Format challenge."""

    def __init__(self, seed: int = 42, num_samples: int | None = None, subset: str | None = None):
        super().__init__(seed=seed, num_samples=num_samples, subset="Style & Format" if subset is None else subset)


class PartiPromptsWritingSymbols(PartiPrompts):
    """Parti Prompts filtered by Writing & Symbols challenge."""

    def __init__(self, seed: int = 42, num_samples: int | None = None, subset: str | None = None):
        super().__init__(seed=seed, num_samples=num_samples, subset="Writing & Symbols" if subset is None else subset)

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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Literal

TASK = Literal[
    "text_to_image",
    "text_generation",
    "audio",
    "image_classification",
    "question_answering",
]


@dataclass
class BenchmarkEntry:
    """A single entry in a benchmark dataset."""

    model_inputs: dict[str, Any]
    model_outputs: dict[str, Any] = field(default_factory=dict)
    path: str = ""
    additional_info: dict[str, Any] = field(default_factory=dict)
    task_type: TASK = "text_to_image"


class Benchmark(ABC):
    """Base class for all benchmark datasets."""

    def __init__(self):
        """Initialize the benchmark. Override to load data lazily or eagerly."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[BenchmarkEntry]:
        """Iterate over benchmark entries."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name identifier for this benchmark."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Return the human-readable display name for this benchmark."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of items in the benchmark."""
        pass

    @property
    @abstractmethod
    def metrics(self) -> List[str]:
        """Return the list of metric names recommended for this benchmark."""
        pass

    @property
    @abstractmethod
    def task_type(self) -> TASK:
        """Return the task type for this benchmark."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of this benchmark."""
        pass

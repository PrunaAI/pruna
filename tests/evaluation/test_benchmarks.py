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

"""Tests for the benchmarks module."""

import pytest

from pruna.evaluation.benchmarks.base import Benchmark, BenchmarkEntry, TASK
from pruna.evaluation.benchmarks.registry import BenchmarkRegistry
from pruna.evaluation.benchmarks.adapter import benchmark_to_datasets
from pruna.evaluation.benchmarks.text_to_image.parti import PartiPrompts


def test_benchmark_entry_creation():
    """Test creating a BenchmarkEntry with all fields."""
    entry = BenchmarkEntry(
        model_inputs={"prompt": "test prompt"},
        model_outputs={"image": "test_image.png"},
        path="test.png",
        additional_info={"category": "test"},
        task_type="text_to_image",
    )
    
    assert entry.model_inputs == {"prompt": "test prompt"}
    assert entry.model_outputs == {"image": "test_image.png"}
    assert entry.path == "test.png"
    assert entry.additional_info == {"category": "test"}
    assert entry.task_type == "text_to_image"


def test_benchmark_entry_defaults():
    """Test BenchmarkEntry with default values."""
    entry = BenchmarkEntry(model_inputs={"prompt": "test"})
    
    assert entry.model_inputs == {"prompt": "test"}
    assert entry.model_outputs == {}
    assert entry.path == ""
    assert entry.additional_info == {}
    assert entry.task_type == "text_to_image"


def test_task_type_literal():
    """Test that TASK type only accepts valid task types."""
    # Valid task types
    valid_tasks: list[TASK] = [
        "text_to_image",
        "text_generation",
        "audio",
        "image_classification",
        "question_answering",
    ]
    
    for task in valid_tasks:
        entry = BenchmarkEntry(model_inputs={}, task_type=task)
        assert entry.task_type == task


def test_benchmark_registry_get():
    """Test getting a benchmark from the registry."""
    benchmark_class = BenchmarkRegistry.get("parti_prompts")
    assert benchmark_class is not None
    assert issubclass(benchmark_class, Benchmark)


def test_benchmark_registry_list_all():
    """Test listing all registered benchmarks."""
    all_benchmarks = BenchmarkRegistry.list_all()
    assert isinstance(all_benchmarks, dict)
    assert len(all_benchmarks) > 0
    assert "parti_prompts" in all_benchmarks


def test_benchmark_registry_get_nonexistent():
    """Test getting a non-existent benchmark returns None."""
    benchmark_class = BenchmarkRegistry.get("nonexistent_benchmark")
    assert benchmark_class is None


def test_parti_prompts_creation():
    """Test creating a PartiPrompts benchmark instance."""
    benchmark = PartiPrompts(seed=42, num_samples=5)
    
    assert benchmark.name == "parti_prompts"
    assert benchmark.display_name == "Parti Prompts"
    assert benchmark.task_type == "text_to_image"
    assert len(benchmark.metrics) > 0
    assert isinstance(benchmark.description, str)


def test_parti_prompts_iteration():
    """Test iterating over PartiPrompts entries."""
    benchmark = PartiPrompts(seed=42, num_samples=5)
    entries = list(benchmark)
    
    assert len(entries) == 5
    for entry in entries:
        assert isinstance(entry, BenchmarkEntry)
        assert "prompt" in entry.model_inputs
        assert entry.task_type == "text_to_image"
        assert entry.model_outputs == {}


def test_parti_prompts_length():
    """Test PartiPrompts __len__ method."""
    benchmark = PartiPrompts(seed=42, num_samples=10)
    assert len(benchmark) == 10


def test_parti_prompts_subset():
    """Test PartiPrompts with a subset filter."""
    benchmark = PartiPrompts(seed=42, num_samples=5, subset="Animals")
    
    assert "animals" in benchmark.name.lower()
    assert "Animals" in benchmark.display_name
    
    entries = list(benchmark)
    for entry in entries:
        assert entry.additional_info.get("category") == "Animals"


def test_benchmark_to_datasets():
    """Test converting a benchmark to datasets."""
    benchmark = PartiPrompts(seed=42, num_samples=5)
    train_ds, val_ds, test_ds = benchmark_to_datasets(benchmark)
    
    assert len(test_ds) == 5
    assert len(train_ds) == 1  # Dummy dataset
    assert len(val_ds) == 1  # Dummy dataset
    
    # Check that test dataset has the expected fields
    sample = test_ds[0]
    assert "prompt" in sample or "text" in sample


def test_benchmark_entry_task_type_validation():
    """Test that BenchmarkEntry validates task_type."""
    # This should work
    entry = BenchmarkEntry(
        model_inputs={},
        task_type="text_to_image",
    )
    assert entry.task_type == "text_to_image"
    
    # Test other valid task types
    for task in ["text_generation", "audio", "image_classification", "question_answering"]:
        entry = BenchmarkEntry(model_inputs={}, task_type=task)
        assert entry.task_type == task

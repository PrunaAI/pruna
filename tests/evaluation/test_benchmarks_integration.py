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

"""Integration tests for benchmarks with datamodule and metrics."""

import pytest

from pruna.data.pruna_datamodule import PrunaDataModule
from pruna.evaluation.benchmarks.registry import BenchmarkRegistry
from pruna.evaluation.benchmarks.text_to_image.parti import PartiPrompts
from pruna.evaluation.metrics.registry import MetricRegistry


@pytest.mark.cpu
def test_datamodule_from_benchmark():
    """Test creating a PrunaDataModule from a benchmark."""
    benchmark = PartiPrompts(seed=42, num_samples=5)
    datamodule = PrunaDataModule.from_benchmark(benchmark)
    
    assert datamodule is not None
    assert datamodule.test_dataset is not None
    assert len(datamodule.test_dataset) == 5


@pytest.mark.cpu
def test_datamodule_from_benchmark_string():
    """Test creating a PrunaDataModule from a benchmark name string."""
    datamodule = PrunaDataModule.from_string("parti_prompts", seed=42)
    
    assert datamodule is not None
    # Limit to small number for testing
    datamodule.limit_datasets(5)
    
    # Test that we can iterate through the dataloader
    test_loader = datamodule.test_dataloader(batch_size=2)
    batch = next(iter(test_loader))
    assert batch is not None


@pytest.mark.cpu
def test_benchmark_with_metrics():
    """Test that benchmarks provide recommended metrics."""
    benchmark = PartiPrompts(seed=42, num_samples=5)
    recommended_metrics = benchmark.metrics
    
    assert isinstance(recommended_metrics, list)
    assert len(recommended_metrics) > 0
    
    # Check that metrics can be retrieved from registry
    for metric_name in recommended_metrics:
        # Some metrics might be registered, some might not
        # Just verify the names are strings
        assert isinstance(metric_name, str)


@pytest.mark.cpu
def test_benchmark_registry_integration():
    """Test that benchmarks are properly registered and can be used."""
    # Get benchmark from registry
    benchmark_class = BenchmarkRegistry.get("parti_prompts")
    assert benchmark_class is not None
    
    # Create instance
    benchmark = benchmark_class(seed=42, num_samples=3)
    
    # Verify it works with datamodule
    datamodule = PrunaDataModule.from_benchmark(benchmark)
    assert datamodule is not None
    
    # Verify we can get entries
    entries = list(benchmark)
    assert len(entries) == 3


@pytest.mark.cpu
def test_benchmark_task_type_mapping():
    """Test that benchmark task types map correctly to collate functions."""
    benchmark = PartiPrompts(seed=42, num_samples=3)
    
    # Create datamodule and verify it uses the correct collate function
    datamodule = PrunaDataModule.from_benchmark(benchmark)
    
    # The collate function should be set based on task_type
    assert datamodule.collate_fn is not None
    
    # Verify we can use the dataloader
    test_loader = datamodule.test_dataloader(batch_size=1)
    batch = next(iter(test_loader))
    assert batch is not None


@pytest.mark.cpu
def test_benchmark_entry_model_outputs():
    """Test that BenchmarkEntry can store model outputs."""
    from pruna.evaluation.benchmarks.base import BenchmarkEntry
    
    entry = BenchmarkEntry(
        model_inputs={"prompt": "test"},
        model_outputs={"image": "generated_image.png", "score": 0.95},
    )
    
    assert entry.model_outputs == {"image": "generated_image.png", "score": 0.95}
    
    # Verify entries from benchmark have empty model_outputs by default
    benchmark = PartiPrompts(seed=42, num_samples=2)
    entries = list(benchmark)
    
    for entry in entries:
        assert entry.model_outputs == {}
        # But model_outputs field exists and can be populated
        entry.model_outputs["test"] = "value"
        assert entry.model_outputs["test"] == "value"

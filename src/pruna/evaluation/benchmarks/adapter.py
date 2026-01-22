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

from typing import Tuple

from datasets import Dataset

from pruna.evaluation.benchmarks.base import Benchmark
from pruna.logging.logger import pruna_logger


def benchmark_to_datasets(benchmark: Benchmark) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Convert a Benchmark instance to train/val/test datasets compatible with PrunaDataModule.

    Parameters
    ----------
    benchmark : Benchmark
        The benchmark instance to convert.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        Train, validation, and test datasets. For test-only benchmarks,
        train and val are dummy datasets with a single item.
    """
    entries = list(benchmark)
    
    # Convert BenchmarkEntries to dict format expected by datasets
    # For prompt-based benchmarks, we need "text" field for prompt_collate
    data = []
    for entry in entries:
        row = entry.model_inputs.copy()
        row.update(entry.additional_info)
        
        # Ensure "text" field exists for prompt collate functions
        if "text" not in row and "prompt" in row:
            row["text"] = row["prompt"]
        elif "text" not in row:
            # If neither exists, use the first string value
            for key, value in row.items():
                if isinstance(value, str):
                    row["text"] = value
                    break
        
        # Add path if needed for some collate functions
        if "path" not in row:
            row["path"] = entry.path
        data.append(row)

    dataset = Dataset.from_list(data)
    
    # For test-only benchmarks (like PartiPrompts), create dummy train/val
    pruna_logger.info(f"{benchmark.display_name} is a test-only dataset. Do not use it for training or validation.")
    dummy = dataset.select([0]) if len(dataset) > 0 else dataset
    
    return dummy, dummy, dataset

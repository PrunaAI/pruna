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

from pruna.evaluation.benchmarks.base import Benchmark, BenchmarkEntry, TASK
from pruna.evaluation.benchmarks.registry import BenchmarkRegistry

# Auto-register all benchmarks
from pruna.evaluation.benchmarks import text_to_image  # noqa: F401

# Auto-register all benchmark subclasses
BenchmarkRegistry.auto_register_subclasses(text_to_image)

__all__ = ["Benchmark", "BenchmarkEntry", "BenchmarkRegistry", "TASK"]

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

import inspect
from typing import Type

from pruna.evaluation.benchmarks.base import Benchmark


class BenchmarkRegistry:
    """Registry for automatically discovering and registering benchmark classes."""

    _registry: dict[str, Type[Benchmark]] = {}

    @classmethod
    def register(cls, benchmark_class: Type[Benchmark]) -> Type[Benchmark]:
        """Register a benchmark class by its name property."""
        # Create instance with default args to get the name
        # This assumes benchmarks have default or no required arguments
        try:
            instance = benchmark_class()
            name = instance.name
        except Exception as e:
            raise ValueError(
                f"Failed to create instance of {benchmark_class.__name__} for registration: {e}. "
                "Ensure the benchmark class can be instantiated with default arguments."
            ) from e
        
        if name in cls._registry:
            raise ValueError(f"Benchmark with name '{name}' is already registered.")
        cls._registry[name] = benchmark_class
        return benchmark_class

    @classmethod
    def get(cls, name: str) -> Type[Benchmark] | None:
        """Get a benchmark class by name."""
        return cls._registry.get(name)

    @classmethod
    def list_all(cls) -> dict[str, Type[Benchmark]]:
        """List all registered benchmarks."""
        return cls._registry.copy()

    @classmethod
    def auto_register_subclasses(cls, module) -> None:
        """Automatically register all Benchmark subclasses in a module."""
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, Benchmark)
                and obj is not Benchmark
                and (obj.__module__ == module.__name__ or obj.__module__.startswith(module.__name__ + "."))
            ):
                cls.register(obj)

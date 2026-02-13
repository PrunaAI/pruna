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

from enum import Enum
from typing import Any, Dict


class AlgorithmTag(Enum):
    """
    Enum for algorithm tags with metadata and documentation.

    This enum provides a type-safe way to categorize algorithms with
    built-in documentation and metadata support.

    Parameters
    ----------
    value : callable
        The load function to be called.
    names : str
        The name of the enum member.
    module : str
        The module where the enum is defined.
    qualname : str
        The qualified name of the enum.
    type : type
        The type of the enum.
    start : int
        The start index for auto-numbering enum values.
    boundary : enum.FlagBoundary or None
        Boundary handling mode used by the Enum functional API for Flag and
        IntFlag enums.
    """

    QUANTIZER = (
        "quantizer",
        "Quantization reduces the precision of the model’s weights and activations, making them much smaller in terms of memory required.",
    )
    PRUNER = (
        "pruner",
        "Pruning removes less important or redundant connections and neurons from a model, resulting in a sparser, more efficient network.",
    )
    FACTORIZER = (
        "factorizer",
        "Factorization batches several small matrix multiplications into one large fused operation or splits matrix operations into smaller ones.",
    )
    KERNEL = (
        "kernel",
        "Kernels are compact and highly optimized routines that run as fast and efficient as possible on a given hardware.",
    )
    CACHER = (
        "cacher",
        "Caching is a technique used to store intermediate results of computations to speed up subsequent operations, particularly useful in reducing inference time for machine learning models by reusing previously computed results.",
    )
    COMPILER = ("compiler", "Compilation optimizes the model for specific hardware.")
    BATCHER = (
        "batcher",
        "Batching groups multiple inputs together to be processed simultaneously, improving computational efficiency and reducing overall processing time.",
    )
    DISTILLER = (
        "distiller",
        "Distillation trains a smaller, simpler model to mimic a larger, more complex model.",
    )
    ENHANCER = (
        "enhancer",
        "Enhancers improve the quality of the model's output. Enhancers can range from post-processing to test time compute algorithms.",
    )
    RESAMPLER = (
        "resampler",
        "Resamplers change the shape of image or video latents during generation to speed up inference.",
    )
    DECODER = (
        "decoder",
        "Decoders speed up autoregressive generation by changing their decoding strategy to be more parallelizable.",
    )

    def __init__(self, name: str, description: str):
        """
        Initialize an algorithm tag with name and description.

        Parameters
        ----------
        name : str
            The tag name identifier.
        description : str
            Human-readable description of what this tag represents.
        """
        self.group_name = name
        self.description = description

    def __str__(self) -> str:
        """
        Return the tag name as string.

        Returns
        -------
        str
            The tag name.
        """
        return self.group_name

    def get_documentation(self) -> Dict[str, Any]:
        """
        Get comprehensive documentation for this tag.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing tag metadata including name, description,
            and usage examples.

        Examples
        --------
        >>> tag = AlgorithmTag.QUANTIZER
        >>> doc = tag.get_documentation()
        >>> print(doc['description'])
        Algorithms that reduce model precision (e.g., INT8, FP16)
        """
        return {
            "name": self.group_name,
            "description": self.description,
        }

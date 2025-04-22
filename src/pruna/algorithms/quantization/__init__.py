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

from pruna.algorithms.pruna_base import PrunaAlgorithmBase
from pruna.config.smash_space import QUANTIZER
from pruna.engine.save import SAVE_FUNCTIONS


class PrunaQuantizer(PrunaAlgorithmBase):
    """Base class for quantization algorithms."""

    # most quantizers, in particular for LLMs, do not require a save_fn different from the original
    save_fn: None | SAVE_FUNCTIONS = None
    algorithm_group = QUANTIZER

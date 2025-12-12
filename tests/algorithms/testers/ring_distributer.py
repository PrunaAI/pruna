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

import pytest

from pruna.algorithms.ring_attn.ring import RingAttn
from pruna.engine.pruna_model import PrunaModel

from .base_tester import AlgorithmTesterBase


@pytest.mark.distributed
class TestRingAttn(AlgorithmTesterBase):
    """Test the RingAttn algorithm."""

    models = ["flux_tiny_random", "wan_tiny_random"]
    reject_models = ["opt_tiny_random"]
    allow_pickle_files = False
    algorithm_class = RingAttn
    metrics = ["psnr"]

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Post-smash hook."""
        assert hasattr(model, "pool")

    def execute_load(self):
        """Overwrite model loading as this is not supported for distributed models."""
        pass

    @classmethod
    def execute_save(cls, smashed_model: PrunaModel):
        """Overwrite model saving as this is not supported for distributed models."""
        pass

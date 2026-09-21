# Copyright 2026 - Pruna AI GmbH. All rights reserved.
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

import random

import pytest

from pruna.data.utils import stratify_dataset


class RecordingDataset:
    """Dataset selection double with observable indices and no remote data."""

    def __init__(self, rows):
        self.rows = list(rows)
        self.selections = []

    def __len__(self):
        """Return the number of rows in the test dataset."""
        return len(self.rows)

    def select(self, indices):
        """Return a test dataset containing the selected rows."""
        indices = list(indices)
        self.selections.append(indices)
        return RecordingDataset(self.rows[index] for index in indices)


@pytest.mark.cpu
@pytest.mark.parametrize("seed", [None, 0, 17])
@pytest.mark.parametrize("length", [0, 1, 12])
def test_explicit_zero_selects_no_rows(length, seed):
    """Verify that an explicit zero selects no rows for every dataset size."""
    dataset = RecordingDataset(range(length))
    result = stratify_dataset(dataset, sample_size=0, seed=seed)
    assert result.rows == []
    assert dataset.selections == [[]]
    assert dataset.rows == list(range(length))


@pytest.mark.cpu
@pytest.mark.parametrize("sample_size", [None, 1, 4, 8, 10])
def test_positive_and_unspecified_sample_sizes_keep_existing_behavior(sample_size):
    """Verify that omitted and positive sample sizes retain existing behavior."""
    dataset = RecordingDataset(range(8))
    result = stratify_dataset(dataset, sample_size=sample_size)
    expected = 8 if sample_size is None else min(sample_size, 8)
    assert result.rows == list(range(expected))
    if sample_size == 10:
        assert result is dataset


@pytest.mark.cpu
@pytest.mark.parametrize("fraction", [0.0, 0.25, 0.75, 1.0])
def test_fraction_sampling_is_unchanged(fraction):
    """Verify that fraction-based sampling remains unchanged."""
    dataset = RecordingDataset(range(8))
    result = stratify_dataset(dataset, fraction=fraction)
    assert result.rows == list(range(int(8 * fraction)))


@pytest.mark.cpu
def test_zero_does_not_bypass_fraction_conflict():
    """Verify that zero still conflicts with a fractional sample request."""
    with pytest.raises(ValueError, match="Fraction and sample_size"):
        stratify_dataset(RecordingDataset(range(8)), sample_size=0, fraction=0.5)


@pytest.mark.cpu
def test_seeded_sampling_is_deterministic_without_changing_global_rng():
    """Verify seeded sampling remains deterministic and isolated from global RNG."""
    state = random.getstate()
    expected = list(range(12))
    random.Random(42).shuffle(expected)
    dataset = RecordingDataset(range(12))
    assert stratify_dataset(dataset, sample_size=4, seed=42).rows == expected[:4]
    assert stratify_dataset(dataset, sample_size=0, seed=42).rows == []
    assert random.getstate() == state

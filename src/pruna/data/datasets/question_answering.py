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

from typing import Tuple

from datasets import Dataset, load_dataset

from pruna.data.utils import split_train_into_train_val_test


def setup_polyglot_dataset(seed: int) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the Polyglot dataset.

    License: Apache 2.0

    Parameters
    ----------
    seed : int
        The seed to use.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The Polyglot dataset.
    """
    dataset = load_dataset("Polyglot-or-Not/Fact-Completion", split="english".capitalize())
    dataset = dataset.rename_column("true", "answer")
    dataset = dataset.rename_column("stem", "question")
    return split_train_into_train_val_test(dataset, seed)

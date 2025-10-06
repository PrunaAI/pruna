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

from importlib.resources import as_file, files
from typing import List, Tuple

from datasets import Dataset, load_dataset


def setup_vbench_dataset(category: str | List[str] | None = None) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the VBench dataset from the VBennch full info json file.

    Parameters
    ----------
    category: str | List[str] | None
        The dimension(s) of the dataset to load.
    """
    # Loading the info json file from the package
    full_info_path = files("vbench").joinpath("VBench_full_info.json")
    with as_file(full_info_path) as real_path:
        dataset = load_dataset("json", data_files=str(real_path), split="train")

    # We change the column names to ensure consistency with other prompt datasets.
    dataset = dataset.rename_column("dimension", "category")
    dataset = dataset.rename_column("prompt_en", "text")

    if category is not None:
        dims = {category} if isinstance(category, str) else set(category)
        dataset = dataset.filter(lambda x: bool(dims.intersection(set(x["category"]))))

    # We put a single item in the train and validation sets, as the benchmark is only intended for testing
    return dataset.select([0]), dataset.select([0]), dataset

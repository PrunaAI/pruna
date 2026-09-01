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

import torch

FLOAT8_DTYPE_FROM_NAME: dict[str, torch.dtype] = {
    "torch.float8_e4m3fn": torch.float8_e4m3fn,
    "torch.float8_e5m2": torch.float8_e5m2,
}


def parse_float8_dtype(name: str) -> torch.dtype:
    """
    Resolve a float8 dtype from its ConfigSpace string name.

    Parameters
    ----------
    name : str
        The dtype name, e.g. ``"torch.float8_e4m3fn"``.

    Returns
    -------
    torch.dtype
        The corresponding float8 dtype.

    Raises
    ------
    ValueError
        If ``name`` is not a supported float8 dtype.
    """
    try:
        return FLOAT8_DTYPE_FROM_NAME[name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported float8 dtype {name!r}. Expected one of {sorted(FLOAT8_DTYPE_FROM_NAME)}."
        ) from exc

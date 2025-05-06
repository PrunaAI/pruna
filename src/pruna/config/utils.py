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
import torch

from pruna.config.smash_config import SmashConfig
from pruna.logging.logger import pruna_logger


def is_empty_config(config: SmashConfig) -> bool:
    """
    Check if the SmashConfig is empty.

    Parameters
    ----------
    config : SmashConfig
        The SmashConfig to check.

    Returns
    -------
    bool
        True if the SmashConfig is empty, False otherwise.
    """
    empty_config = SmashConfig()
    return config == empty_config


def validate_device(device: str | None) -> str:
    """
    Validate if the specified device is available on the current system.
    Supports 'cuda', 'mps', 'cpu' and other PyTorch devices.
    If device is None, the best available device will be returned.

    Parameters
    ----------
    device : str | None, optional
        The device to validate (e.g. 'cuda', 'mps', 'cpu')

    Returns
    -------
    str
        The best available device.
    """  # noqa: D205
    if device is None:
        if torch.cuda.is_available():
            pruna_logger.warning("No device specified. Using best available device: CUDA.")
            return "cuda"
        elif torch.backends.mps.is_available():
            pruna_logger.warning("No device specified. Using best available device: MPS.")
            return "mps"
        else:
            pruna_logger.warning("No device specified. Using best available device: CPU.")
            return "cpu"
    elif device == "cuda":
        if not torch.cuda.is_available():
            pruna_logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
    elif device == "mps":
        if not torch.backends.mps.is_available():
            pruna_logger.warning("MPS requested but not available. Falling back to CPU.")
            return "cpu"
    elif device != "cpu":
        pruna_logger.warning(f"Unknown device '{device}' requested. Falling back to CPU.")
        return "cpu"

    return device

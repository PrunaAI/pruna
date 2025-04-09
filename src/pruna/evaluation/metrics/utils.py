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

from typing import Any, List

import torch


def metric_data_processor(
    x: List[Any] | torch.Tensor, gt: List[Any] | torch.Tensor, outputs: Any, call_type: str
) -> List[Any]:
    """
    Arrange metric inputs based on the specified configuration call type.

    This function determines the order and selection of inputs to be passed to various metrics.

    The function supports different input arrangements through the 'call_type' configuration:
    - 'x_y': Uses input data (x) and model outputs
    - 'gt_y': Uses ground truth (gt) and model outputs
    - 'y_x': Uses model outputs and input data (x)
    - 'y_gt': Uses model outputs and ground truth (gt)
    - 'pairwise_gt_y': Uses cached base model outputs (gt) and smashed model outputs (y).
    - 'pairwise_y_gt': Uses smashed model outputs (y) and cached base model outputs (gt).
    The evaluation agent is expected to pass the cached base model outputs as gt.

    Parameters
    ----------
    x : Any
        The input data (e.g., input images, text prompts).
    gt : Any
        The ground truth data (e.g., correct labels, target images, cached model outputs).
    outputs : Any
        The model outputs or predictions.
    call_type : str
        The type of call to be made to the metric.

    Returns
    -------
    List[Any]
        A list containing the arranged inputs in the order specified by call_type.

    Raises
    ------
    ValueError
        If the specified call_type is not one of: 'x_y', 'gt_y', 'y_x', 'y_gt', 'pairwise'.

    Examples
    --------
    >>> call_type = "gt_y"
    >>> inputs = metric_data_processor(x_data, ground_truth, model_outputs, call_type)
    >>> # Returns [ground_truth, model_outputs]
    """
    if call_type == "x_y":
        return [x, outputs]
    elif call_type == "gt_y":
        return [gt, outputs]
    elif call_type == "y_x":
        return [outputs, x]
    elif call_type == "y_gt":
        return [outputs, gt]
    elif call_type == "pairwise_gt_y":
        return [gt, outputs]
    elif call_type == "pairwise_y_gt":
        return [outputs, gt]
    else:
        raise ValueError(f"Invalid call type: {call_type}")

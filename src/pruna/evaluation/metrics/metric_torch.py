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

from contextlib import suppress
from enum import Enum
from functools import partial
from typing import Any, Callable, List, Optional, Union

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.classification import Accuracy, Precision, Recall
from torchmetrics.image import (
    FrechetInceptionDistance,
    LearnedPerceptualImagePatchSimilarity,
    MultiScaleStructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchmetrics.image.arniqa import ARNIQA
from torchmetrics.multimodal.clip_iqa import CLIPImageQualityAssessment
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.text import Perplexity
from torchvision import transforms

from pruna.engine.utils import device_to_string
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import (
    CALL_TYPES,
    PAIRWISE,
    SINGLE,
    get_pairwise_pairing,
    get_single_pairing,
    metric_data_processor,
)
from pruna.logging.logger import pruna_logger

_PRUNA_TASK_ROUTING_KWARGS: tuple[str, ...] = (
    "vlm_type",
    "model_name",
    "structured_output",
    "vlm_kwargs",
    "api_key",
)


def _strip_task_routing_kwargs(kwargs: dict[str, Any]) -> None:
    """
    Drop kwargs :class:`~pruna.evaluation.task.Task` passes when building mixed metric lists.

    Torchmetrics classes often end with ``**kwargs`` and would otherwise accept bogus keys
    until a lower layer raises. Stripping here keeps :class:`TorchMetricWrapper` the single
    choke point between Pruna routing and torchmetrics constructors.
    """
    for key in _PRUNA_TASK_ROUTING_KWARGS:
        kwargs.pop(key, None)


def default_update(metric: Metric, *args, **kwargs) -> None:
    """
    Default update function for metrics that don't require special handling.

    Parameters
    ----------
    metric : Metric
        The metric instance.
    *args : Any
        The arguments to pass to the metric update method.
    **kwargs : Any
        The keyword arguments to pass to the metric update method.
    """
    metric.update(*args, **kwargs)


# Update functions for metrics that require special handling.
def fid_update(metric: FrechetInceptionDistance, reals: Any, fakes: Any) -> None:
    """
    Update handler for FID metric.

    Parameters
    ----------
    metric : FrechetInceptionDistance instance
        The FID metric instance.
    reals : Any
        The ground truth images tensor.
    fakes : Any
        The generated images tensor.
    """
    metric.update(reals, real=True)
    metric.update(fakes, real=False)


def lpips_update(metric: LearnedPerceptualImagePatchSimilarity, preds: Any, target: Any) -> None:
    """
    Update handler for LPIPS metric.

    Parameters
    ----------
    metric : LearnedPerceptualImagePatchSimilarity instance
        The LPIPS metric instance.
    preds : Any
        The generated images tensor.
    target : Any
        The ground truth images tensor.
    """
    transform = transforms.Compose([transforms.Normalize(mean=[0.5], std=[0.5])])  # converts to [-1, 1]
    preds = preds.float() / 255.0
    target = target.float() / 255.0

    preds = transform(preds)
    target = transform(target)

    metric.update(preds, target)


def arniqa_update(metric: ARNIQA, preds: Any) -> None:
    """
    Update handler for ARNIQA metric.

    Parameters
    ----------
    metric : ARNIQA instance
        The ARNIQA metric instance.
    preds : Any
        The generated images tensor.
    """
    preds = preds.float() / 255.0
    metric.update(preds)


def ssim_update(
    metric: StructuralSimilarityIndexMeasure | MultiScaleStructuralSimilarityIndexMeasure, preds: Any, target: Any
) -> None:
    """
    Update handler for SSIM or MS-SSIM metric.

    Parameters
    ----------
    metric : StructuralSimilarityIndexMeasure | MultiScaleStructuralSimilarityIndexMeasure instance
        The SSIM or MS-SSIM metric instance.
    preds : Any
        The generated images tensor.
    target : Any
        The ground truth images tensor.
    """
    if preds.dtype != torch.float32:
        preds = preds.float()
    if target.dtype != torch.float32:
        target = target.float()
    metric.update(preds, target)


# Available metrics
class TorchMetrics(Enum):
    """
    Enumeration of torchmetrics metrics for evaluation.

    This enum provides a tuple per member (metric_factory, update_fn, call_type):
    metric_factory builds the metric (typically a torchmetrics class, or
    functools.partial when some constructor arguments are fixed); update_fn is
    an optional custom update handler; call_type describes how inputs are paired
    for the metric.

    Parameters
    ----------
    value : tuple
        Tuple holding metric_factory, update_fn, and call_type as described above.
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

    fid = (FrechetInceptionDistance, fid_update, "gt_y")
    accuracy = (Accuracy, None, "y_gt")
    perplexity = (Perplexity, None, "y_gt")
    clip_score = (CLIPScore, None, "y_x")
    precision = (Precision, None, "y_gt")
    recall = (Recall, None, "y_gt")
    psnr = (partial(PeakSignalNoiseRatio, data_range=255.0), None, "pairwise_y_gt")
    ssim = (StructuralSimilarityIndexMeasure, ssim_update, "pairwise_y_gt")
    msssim = (MultiScaleStructuralSimilarityIndexMeasure, ssim_update, "pairwise_y_gt")
    lpips = (LearnedPerceptualImagePatchSimilarity, lpips_update, "pairwise_y_gt")
    arniqa = (ARNIQA, arniqa_update, "y")
    clipiqa = (CLIPImageQualityAssessment, None, "y")

    def __init__(self, *args, **kwargs) -> None:
        self.tm: Callable[..., Metric] = self.value[0]
        self.update_fn = self.value[1] or default_update
        self.call_type = self.value[2]

    def __call__(self, **kwargs) -> Metric:
        """
        Get an instance of the metric.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments for the metric constructor.

        Returns
        -------
        Metric
            An instance of the torchmetrics metric.
        """
        return self.tm(**kwargs)


@MetricRegistry.register_wrapper(available_metrics=TorchMetrics.__members__.keys())
class TorchMetricWrapper(StatefulMetric):
    """
    Wrapper for torchmetrics.

    Provides a consistent interface for different metrics from torchmetrics.

    Parameters
    ----------
    metric_name : str
        Name of the metric.
    call_type : str
        Specifies the order and type of inputs to use for metric calculation.
        This parameter helps determine how the inputs should be arranged
        when calculating the metric.
    **kwargs :
        Additional arguments for the metric constructor.
    """

    def __new__(cls, metric_name: str, call_type: str = "", **kwargs) -> StatefulMetric:  # type: ignore
        """
        Creating a new instance of the class.

        Parameters
        ----------
        metric_name : str
            Name of the metric.
        call_type : str
            Specifies the order and type of inputs to use for metric calculation.
        """
        # Special case for clip_score until new torchmetrics version.
        if metric_name == "clip_score" and call_type.startswith(PAIRWISE):
            from pruna.evaluation.metrics.metric_pairwise_clip import PairwiseClipScore

            _strip_task_routing_kwargs(kwargs)
            return PairwiseClipScore(**kwargs)
        return super().__new__(cls)

    def __init__(self, metric_name: str, call_type: str = "", **kwargs) -> None:
        """
        Initialize the torchmetrics metric wrapper.

        Raises
        ------
        ValueError
            If the metric name is not supported.
        """
        self.metric_name = metric_name
        _strip_task_routing_kwargs(kwargs)
        super().__init__(kwargs.pop("device", None))
        try:
            self.metric = TorchMetrics[metric_name](**kwargs)
            with suppress(AttributeError, TypeError):
                self.metric = self.metric.to(self.device)

            # Get the specific update function for the metric, or use the default if not found.
            self.update_fn = TorchMetrics[metric_name].update_fn
        except KeyError:
            raise ValueError(f"Metric {metric_name} is not supported.")

        self.call_type = get_call_type(call_type, metric_name)

        pruna_logger.info(f"Using call_type: {self.call_type} for metric {metric_name}")
        self.higher_is_better = self.metric.higher_is_better if self.metric.higher_is_better is not None else True

    def update(self, x: List[Any] | Tensor, gt: List[Any] | Tensor, outputs: Any) -> None:
        """
        Update the wrapped metric's state with new batch data.

        This method processes the input data through metric_data_processor to arrange inputs
        in the correct order based on the metric's configuration. The arranged inputs are then
        passed to the metric's update function.

        The metric_data_processor supports different input arrangements through 'call_type':

        Parameters
        ----------
        x : List[Any] | Tensor
            The input data.
        gt : List[Any] | Tensor
            The ground truth data.
        outputs : Any
            The output data.
        """
        metric_inputs = metric_data_processor(x, gt, outputs, self.call_type, self.metric.device)
        self.update_fn(self.metric, *metric_inputs)

    def add_state(
        self,
        name: str,
        default: Union[list, Tensor],
        dist_reduce_fx: Optional[Union[str, Callable]] = None,
        persistent: bool = False,
    ) -> None:
        """
        Add metric state variables.

        Parameters
        ----------
        name : str
            Name of the state variable.
        default : Union[list, Tensor]
            Default value of the state variable.
        dist_reduce_fx : Optional[Union[str, Callable]], optional
            Function to reduce the state variable in distributed mode.
        persistent : bool, optional
            Whether the state variable should be saved to the ``state_dict`` of the module.
        """
        self.metric.add_state(name, default, dist_reduce_fx, persistent)

    def forward(self, *args, **kwargs) -> None:
        """
        Aggregate and evaluate the batch input directly.

        Parameters
        ----------
        *args : Any
            The arguments to pass to the metric.
        **kwargs : Any
            The keyword arguments to pass to the metric.
        """
        self.metric.forward(*args, **kwargs)

    def reset(self) -> None:
        """Reset the wrapped metric's state."""
        self.metric.reset()

    def compute(self) -> Any:
        """
        Compute the metric value.

        Returns
        -------
        Any
            The computed metric value.
        """
        result = self.metric.compute()

        # Normally we have a single score for each metric for the entire dataset.
        # For IQA metrics we have a single score per image, so we need to convert the tensor to a list.
        if isinstance(result, Tensor):
            result_value = result.item() if result.numel() == 1 else result.mean().item()
        else:
            result_value = result
        return MetricResult(
            self.metric_name,
            self.__dict__.copy(),
            result_value,
        )

    def move_to_device(self, device: str | torch.device) -> None:
        """
        Move the metric to a specific device.

        Parameters
        ----------
        device : str | torch.device
            The device to move the metric to.
        """
        if not self.is_device_supported(device):
            raise ValueError(
                f"Metric {self.metric_name} does not support device {device}. Must be one of {self.runs_on}."
            )
        self.device = device_to_string(device)
        self.metric = self.metric.to(device)


def get_call_type(call_type: str, metric_name: str) -> str:
    """
    Get the correct call type for the metric.

    Parameters
    ----------
    call_type : str
        The call type to set.
    metric_name : str
        The name of the metric.

    Returns
    -------
    str
        The call type.
    """
    if not call_type:
        return TorchMetrics[metric_name].call_type
    elif call_type == PAIRWISE:
        # If the call type and default call type match, we use the default call type.
        if TorchMetrics[metric_name].call_type.startswith(PAIRWISE):
            return TorchMetrics[metric_name].call_type
        else:
            return get_pairwise_pairing(TorchMetrics[metric_name].call_type)
    elif call_type == SINGLE:
        # If the call type and default call type match, we use the default call type.
        if not (TorchMetrics[metric_name].call_type.startswith(PAIRWISE)):
            return TorchMetrics[metric_name].call_type
        else:
            return get_single_pairing(TorchMetrics[metric_name].call_type)
    else:
        pruna_logger.error(f"Invalid call type: {call_type}. Must be one of {CALL_TYPES}.")
        raise ValueError(f"Invalid call type: {call_type}. Must be one of {CALL_TYPES}.")

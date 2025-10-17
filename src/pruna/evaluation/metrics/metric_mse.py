from __future__ import annotations

from typing import Any, List, cast

import cv2
import numpy as np
import torch
from torch import Tensor

from pruna.engine.utils import set_to_best_available_device
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import SINGLE, get_call_type_for_single_metric, metric_data_processor
from pruna.logging.logger import pruna_logger

METRIC_MSE = "mse"


@MetricRegistry.register(METRIC_MSE)
class MSEMetric(StatefulMetric):
    """
    Mean Squared Error metric. Accumulates sum of squared errors and sample count across batches.
    """
    default_call_type: str = "gt_y"  # usually ground truths  VS preds in MSE
    higher_is_better: bool = False #Because Lower MSE means better performance
    metric_name: str = METRIC_MSE
    
    def __init__(self, *args, call_type: str = SINGLE, **kwargs) -> None:
        super().__init__() # calls the parent class’s constructor (__init__)
        self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type) # unless call type specified by user , it uses the default call type defined above (for now preds V/S ground truths)
        self.add_state("sum_sq_error", 0.0)
        self.add_state("count", 0)

    
    def update(self, inputs, ground_truths, predictions):
        preds, targets = metric_data_processor(inputs, ground_truths, predictions, self.call_type)
        # Compute squared errors
        squared_errors = (preds - targets) ** 2
        # Sum squared errors and count
        self.sum_sq_error += squared_errors.sum() # sum of squared errors 
        self.count += targets.numel() # total number of elements


    def compute(self) -> MetricResult:
        # Calculate Mean Squared Error
        mse = self.sum_sq_error / self.count  # Actual MSE calculation
        return MetricResult(self.metric_name, self.__dict__.copy(), mse.item())
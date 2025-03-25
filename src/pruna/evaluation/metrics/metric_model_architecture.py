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

import inspect
from typing import Any, Dict, List, Tuple

import thop
import torch
from torch.utils.data import DataLoader

from pruna.engine.call_sequence_tracker import CallSequenceTracker
from pruna.engine.pruna_model import PrunaModel
from pruna.evaluation.metrics.metric_base import BaseMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.logging.logger import pruna_logger


@MetricRegistry.register("model_architecture")
class ModelArchitectureMetric(BaseMetric):
    """
    Evaluate the model architecture.

    1. Evaluate the MACs of the model at inference.
    2. Evaluate the number of parameters of the model.

    Parameters
    ----------
    device : str | torch.device, default="cuda"
        The device to evaluate the model on.
    """

    def __init__(self, device: str | torch.device = "cuda") -> None:
        super().__init__()
        self.device = device
        self.module_macs: Dict[str, Any] = {}
        self.module_params: Dict[str, Any] = {}
        self.call_tracker = CallSequenceTracker()

    def compute(self, model: PrunaModel, dataloader: DataLoader) -> Dict[str, Any]:
        """
        Compute the MACs and number of parameters of the model during inference.

        Parameters
        ----------
        model : PrunaModel
            The model to evaluate.
        dataloader : DataLoader
            The dataloader to evaluate the model on.

        Returns
        -------
        Dict[str, Any]
            The MACs and number of parameters of the model during inference.
        """
        model.set_to_eval()
        model.move_to_device(self.device)
        # wrap the forward method of the model.
        # the wrapper will track the call sequence and input stats for the nn.Modules of the model.
        self.call_tracker.wrap(model)
        self.call_tracker.clean_call_sequence()

        batch = next(iter(dataloader))
        batch = model.inference_handler.move_inputs_to_device(batch, self.device)
        inputs = model.inference_handler.prepare_inputs(batch)

        model(inputs)

        total_macs = 0
        self.module_macs = {}
        total_params = 0
        self.module_params = {}

        # iterate over the call sequence and calculate the MACs and parameters for each nn.Module in the model.
        for call in self.call_tracker.call_sequence:
            module = call["module"]
            module = self.call_tracker.unwrap(module)

            try:
                sig = inspect.signature(module.forward)
                dummy_inputs = self.generate_dummy_inputs(call["inputs"], call["kwargs"], sig)
                macs, params = thop.profile(module, inputs=dummy_inputs, verbose=False)

                self.module_macs[module.__class__.__name__] = self.module_macs.get(module.__class__.__name__, 0) + macs
                total_macs += macs

                self.module_params[module.__class__.__name__] = (
                    self.module_params.get(module.__class__.__name__, 0) + params
                )
                total_params += params

            except Exception as e:
                pruna_logger.error(f"Could not calculate MACs for {module.__class__.__name__}: {e}")

        return {"total_macs": total_macs, "total_params": total_params}

    def generate_dummy_inputs(
        self, input_info: List[Dict[str, Any]], kwargs_info: Dict[str, Any], sig: inspect.Signature
    ) -> Tuple[Any, ...]:
        """
        Generate dummy tensors matching the metadata of intermediate activations.

        This method creates temporary tensors that match the shape, dtype, and device
        of the actual intermediate activations that flow between PyTorch modules
        during model execution. These dummy tensors are used solely for MAC counting
        and parameter counting.

        Parameters
        ----------
        input_info : List[Dict[str, Any]]
            The information about the inputs.
        kwargs_info : Dict[str, Any]
            The information about the keyword arguments.
        sig : inspect.Signature
            The signature of the model.

        Returns
        -------
        Tuple[Any, ...]
            The dummy inputs.
        """
        dummy_args: list[torch.Tensor | None] = []

        for info in input_info:
            if info["shape"] is not None:
                if info["dtype"] in [torch.int32, torch.int64]:
                    dummy_args.append(torch.randint(0, 10, info["shape"], dtype=info["dtype"], device=info["device"]))
                else:
                    dummy_args.append(torch.randn(*info["shape"], dtype=info["dtype"], device=info["device"]))
            else:  # non-tensor input
                dummy_args.append(None)

        dummy_kwargs: Dict[str, torch.Tensor | None] = {}

        for k, info in kwargs_info.items():
            if info["shape"] is not None:
                if info["dtype"] in [torch.int32, torch.int64]:
                    dummy_kwargs[k] = torch.randint(0, 10, info["shape"], dtype=info["dtype"], device=info["device"])
                else:
                    dummy_kwargs[k] = torch.randn(*info["shape"], dtype=info["dtype"], device=info["device"])
            else:  # non-tensor kwarg
                dummy_kwargs[k] = None

        ordered_args = []
        # Reorder arguments to match the module's forward method signature
        # Example: If forward(self, x, y, z=None) and we have:
        #   dummy_args = [tensor1, tensor2] # ruff: noqa: ERA001
        #   dummy_kwargs = {'z': tensor3} # ruff: noqa: ERA001
        # This ensures we return (tensor1, tensor2, tensor3) in the correct order
        # rather than passing z as a keyword argument
        for param in sig.parameters.values():
            if param.name in kwargs_info:
                ordered_args.append(dummy_kwargs[param.name])
            elif dummy_args:
                if param.name == "args":
                    for arg in dummy_args:
                        ordered_args.append(arg)
                else:
                    ordered_args.append(dummy_args.pop(0))
            elif param.name == "kwargs":
                for k, v in dummy_kwargs.items():
                    ordered_args.append(v)
                break
        return tuple(ordered_args)

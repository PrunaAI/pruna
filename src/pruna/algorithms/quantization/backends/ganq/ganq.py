# ruff: noqa: N806, N803, N802
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
import torch.nn as nn
import transformers

from pruna.algorithms.quantization.backends.ganq.lut_quant import LUTQuant

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GANQ:
    """GANQ class for quantizing neural network layers."""

    def __init__(self, layer, model_type):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.model_type = model_type
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.XXt = torch.zeros((self.columns, self.columns), device=self.dev)

    def add_batch(self, inp, out):
        """Accumulate input statistics for quantization."""
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)

        if isinstance(self.layer, (nn.Linear, transformers.Conv1D)):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        inp = inp.float()
        self.XXt += inp @ inp.T

    def fasterquant(self, sparsity=0.0, bits=4, max_epoch=10, pre_process=True, full_rows=0):
        """Main function to perform GANQ quantization."""
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        quant = LUTQuant(
            bits=bits,
            W=W,
            XXt=self.XXt,
            max_epoch=max_epoch,
            sparsity=sparsity,
            model_type=self.model_type,
            pre_process=pre_process,
            full_rows=full_rows,
        )
        W = quant.quantization()

        torch.cuda.synchronize()

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        """Free up memory."""
        self.XXt = None
        torch.cuda.empty_cache()

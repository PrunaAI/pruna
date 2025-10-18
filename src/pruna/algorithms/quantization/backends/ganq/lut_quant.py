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

from pruna.algorithms.quantization.backends.ganq.utils import (
    denormalize,
    init_t_3bit,
    init_t_4bit,
    norm_params,
    normalize,
)


class LUTQuant:
    """LUTQuant class for quantization using Look-Up Tables (LUTs)."""

    def __init__(
        self,
        bits,
        W,
        XXt,
        max_epoch=10,
        sparsity=0.0,
        full_rows=0,
        model_type="opt",
        pre_process=True,
    ):
        if bits not in [3, 4]:
            raise NotImplementedError("Only 3 and 4 bits are supported")

        # Set quantization parameters
        self.bits = bits
        self.num_levels = 2**self.bits
        self.sparsity = sparsity
        self.full_rows = full_rows
        self.max_epoch = max_epoch
        self.model_type = model_type
        self.pre_process = pre_process

        # Store input tensors
        self.W = W
        self.m, self.n = self.W.shape
        self.device = W.device
        self.XXt = XXt

        # Handle outliers if sparsity is enabled
        if self.sparsity > 0:
            ratio = 1 - 0.5 * self.sparsity
            self.W, self.S, self.row_mask = self.outlier_split(ratio=ratio, full_rows=self.full_rows)

        # Normalize weights if pre-processing is enabled
        if self.pre_process:
            self.median, self.iqr = norm_params(self.W)
            self.W = normalize(self.W, self.median, self.iqr)

        # Compute Cholesky decomposition with numerical stability
        if self.model_type == "opt":
            dead = torch.diag(self.XXt) == 0
            self.XXt[dead, dead] = 1
            self.W[:, dead] = torch.mean(self.W[:, ~dead], dim=1, keepdim=True)
            offset = (torch.sum(torch.abs(self.XXt), dim=1) - 2 * torch.diag(self.XXt)).clamp(min=1e-8)
            self.L = torch.linalg.cholesky(self.XXt + torch.diag(offset))
        else:
            try:
                self.L = torch.linalg.cholesky(self.XXt)
            except RuntimeError:
                offset = (torch.sum(torch.abs(self.XXt), dim=1) - 2 * torch.diag(self.XXt)).clamp(min=1e-8)
                self.L = torch.linalg.cholesky(self.XXt + torch.diag(offset))

        # Precompute matrices
        self.LLt = self.L @ self.L.T
        self.WLLt = self.W @ self.LLt
        self.WL = self.W @ self.L
        self.WXXt = self.W @ self.XXt

    def dequantization(self, T, S):
        """Dequantize using the selection matrix S and table T."""
        return torch.einsum("mil, mln -> min", T.unsqueeze(1), S).squeeze(1)

    def initialize_T(self, W, seg_levels=None, seg_boundaries=None):
        """Initialize the table T using distribution-based splits."""
        if self.bits == 4:
            seg_levels = seg_levels or [2, 6, 6, 2] if self.sparsity > 0 else seg_levels or [3, 5, 5, 3]
            seg_boundaries = seg_boundaries or [0.0, 0.25, 0.5, 0.75, 1.0]
            assert sum(seg_levels) == self.num_levels, "Number of levels must sum up to total num_levels."
            return init_t_4bit(W, seg_levels, seg_boundaries)
        elif self.bits == 3:
            seg_levels = seg_levels or [2, 4, 2]
            seg_boundaries = seg_boundaries or [0.0, 0.20, 0.80, 1.0]
            assert sum(seg_levels) == self.num_levels, "Number of levels must sum up to total num_levels."
            return init_t_3bit(W, seg_levels, seg_boundaries)
        else:
            raise NotImplementedError("Only 3 and 4 bits are supported")

    def outlier_split(self, ratio=0.9975, full_rows=0):
        """Outlier detection version 3 (more sophisticated)."""
        outlier_mask = torch.zeros_like(self.W)
        if full_rows > 0:
            row_variances = torch.var(self.W, dim=1, unbiased=False)
            _, largest_variance_indices = torch.topk(row_variances, full_rows, largest=True, sorted=False)
            outlier_mask[largest_variance_indices] = 1.0
            ratio = 2 * (1 - ratio) * self.m * self.n / ((self.m - full_rows) * self.n)
            ratio = 1 - 2 * ratio

        row_mask = 1.0 - outlier_mask
        cutoff_idx = int(self.n * ratio) - 1

        sorted_W, sorted_indices = torch.sort(self.W, dim=1, stable=True)
        cutoff_values = sorted_W[:, cutoff_idx].unsqueeze(1)

        lower_cutoff_idx = round(self.n * (1 - ratio) + 0.5) + 1
        lower_cutoff_values = sorted_W[:, lower_cutoff_idx + 1].unsqueeze(1)

        outliers = (cutoff_values <= self.W) | (lower_cutoff_values >= self.W)
        outlier_mask[outliers] = 1.0

        S_prime = self.W * outlier_mask
        W_prime = self.W - S_prime

        return W_prime, S_prime, row_mask

    def update_S(self, T):
        """Update the selection matrix S."""
        W_q = torch.zeros_like(self.W, device=self.device)
        S = torch.zeros(self.m, self.num_levels, self.n, device=self.device)

        for i in range(self.n - 1, -1, -1):
            residuals = self.WL[:, i] - torch.sum(W_q[:, i + 1 :] * self.L[i + 1 :, i], dim=1)
            candidates = T * self.L[i, i]

            differences = torch.abs(candidates - residuals.unsqueeze(1))
            closest_indices = torch.argmin(differences, dim=1)
            W_q[:, i] = T.gather(1, closest_indices.unsqueeze(1)).squeeze(1)
            S[torch.arange(self.m), closest_indices, i] = 1

        return S

    def update_T(self, S):
        """Update the table T."""
        St = S.transpose(-1, -2)
        SLLt = S @ self.LLt
        SLLtSt = torch.matmul(SLLt, St)

        numerator = torch.matmul(self.WLLt.unsqueeze(1), St)
        denominator = torch.linalg.pinv(SLLtSt.to(torch.float64)).to(torch.float32)

        T = torch.matmul(numerator, denominator)

        return T.squeeze(1)

    def quantization(self):
        """Main quantization loop."""
        T = self.initialize_T(self.W)
        best_diff = float("inf")
        best_S, best_T = None, None

        for _ in range(self.max_epoch):
            S = self.update_S(T)
            T = self.update_T(S)

            residual = self.W - self.dequantization(T, S)
            current_diff = torch.trace(residual.T @ residual @ self.XXt).item()

            if current_diff < best_diff:
                best_S, best_T = S.detach(), T.detach()
                best_diff = current_diff

        if self.max_epoch > 0:
            if self.pre_process:
                best_T = denormalize(best_T, self.median, self.iqr)
            self.W = self.dequantization(best_T, best_S)
            if self.sparsity > 0:
                self.W = self.W * self.row_mask + self.S

        return self.W

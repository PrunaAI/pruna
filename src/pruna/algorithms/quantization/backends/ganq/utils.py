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


def init_t_4bit(W, seg_levels=[3, 5, 5, 3], seg_boundaries=[0.0, 0.25, 0.5, 0.75, 1.0]):
    """4-bit quantization with non-uniform levels."""
    W_min = W.min(dim=1).values
    W_max = W.max(dim=1).values
    range_vals = W_max - W_min

    T_segments = []

    for i, levels in enumerate(seg_levels):
        seg_start_ratio = seg_boundaries[i]
        seg_end_ratio = seg_boundaries[i + 1]
        seg_start = W_min + range_vals * seg_start_ratio
        seg_end = W_min + range_vals * seg_end_ratio
        seg_lin = torch.linspace(0, 1, steps=levels, device=W.device)
        seg_T = seg_start.unsqueeze(1) + (seg_end - seg_start).unsqueeze(1) * seg_lin.unsqueeze(0)
        T_segments.append(seg_T)

    return torch.cat(T_segments, dim=1)


def init_t_3bit(W, sub_counts=[2, 4, 2], fraction_boundaries=[0.0, 0.20, 0.80, 1.0]):
    """3-bit quantization with distribution-based splits."""
    device = W.device
    B, N = W.shape
    sorted_data, _ = W.sort(dim=1)

    frac_t = torch.tensor(fraction_boundaries, device=device, dtype=sorted_data.dtype)
    M = len(sub_counts)

    splitted_list = []

    for j in range(M):
        c_j = sub_counts[j]
        if c_j == 0:
            continue

        f0 = frac_t[j]
        f1 = frac_t[j + 1]

        if j == 0:
            base = torch.linspace(0.0, 1.0, steps=c_j, device=device, dtype=sorted_data.dtype)
            frac_grid = f0 + base * (f1 - f0)
        else:
            base_full = torch.linspace(0.0, 1.0, steps=c_j + 1, device=device, dtype=sorted_data.dtype)
            base_segment = base_full[1:]
            frac_grid = f0 + base_segment * (f1 - f0)

        frac_grid = frac_grid.unsqueeze(0).expand(B, -1)
        indexf = frac_grid * (N - 1)
        left_idx = torch.floor(indexf).long()
        right_idx = torch.clamp(left_idx + 1, max=N - 1)
        alpha = indexf - left_idx

        left_vals = torch.gather(sorted_data, 1, left_idx)
        right_vals = torch.gather(sorted_data, 1, right_idx)
        sub_points = (1.0 - alpha) * left_vals + alpha * right_vals

        splitted_list.append(sub_points)

    return (
        torch.cat(splitted_list, dim=1) if splitted_list else torch.empty((B, 0), device=device, dtype=sorted_data.dtype)
    )


def norm_params(W):
    """Compute the median and IQR for normalization."""
    median = W.median(dim=1, keepdim=True).values
    q75, q25 = torch.quantile(W, 0.75, dim=1, keepdim=True), torch.quantile(W, 0.25, dim=1, keepdim=True)
    iqr = q75 - q25 + 1e-8
    return median, iqr


def normalize(tensor, median, iqr):
    """Normalize the tensor using median and IQR."""
    return (tensor - median) / iqr


def denormalize(tensor, median, iqr):
    """Denormalize the tensor using median and IQR."""
    return tensor * iqr + median

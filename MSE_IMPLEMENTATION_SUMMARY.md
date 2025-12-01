# MSE Metric Implementation Summary

## Completion Status

All tasks have been successfully completed for the MSE (Mean Squared Error) metric implementation in Pruna AI.

---
## Is this for merging it to Pruna or is it more for giving information?

I hope this gets merged to Pruna AI. But if it doesn't, I will still use it for my own purposes and this was a good learning experience.
Even if evaluation metrics is not merged, do use the screenshot in the CONTRIBUTING.md file (at the end of the file), I hope it helps others in their work.
---

## What Was Done

### 1. Metric Implementation (`src/pruna/evaluation/metrics/metric_mse.py`)

**Features:**
- Inherits from `StatefulMetric` for proper batch accumulation
- Implements `update()` method with proper signature: `(x, gt, outputs)`
- Uses `metric_data_processor()` for consistent input handling
- Accumulates squared errors in a list of tensors (proper state management)
- Implements `compute()` method returning `MetricResult`
- Handles edge cases: None inputs, shape mismatches, empty state
- Device-aware: automatically moves tensors to correct device
- Properly documented with NumPy-style docstrings

**Key Implementation Details:**
```python
@MetricRegistry.register(METRIC_MSE)
class MSEMetric(StatefulMetric):
    default_call_type: str = "gt_y"
    higher_is_better: bool = False
    metric_name: str = METRIC_MSE
```

---

### 2. Registry Integration

**File:** `src/pruna/evaluation/metrics/__init__.py`

Added:
- Import: `from pruna.evaluation.metrics.metric_mse import MSEMetric`
- Export in `__all__`: `"MSEMetric"`

The metric is now discoverable by Pruna's evaluation framework and can be used with:
```python
task = Task(metrics=["mse"])
```

---

### 3. Comprehensive Tests (`tests/evaluation/test_mse.py`)

**15 Tests Created:**
1. `test_mse_perfect_match` - Zero MSE for identical tensors
2. `test_mse_known_value` - Verify calculation correctness
3. `test_mse_multiple_batches` - Batch accumulation
4. `test_mse_empty_state` - Returns NaN when no data
5. `test_mse_multidimensional` - 2D tensor support
6. `test_mse_3d_tensors` - Image-like data support
7. `test_mse_reset` - State reset functionality
8. `test_mse_mixed_values` - Positive and negative errors
9. `test_mse_large_errors` - Large magnitude handling
10. `test_mse_none_handling` - Graceful None handling
11. `test_mse_cuda` - GPU support (skipped if CUDA unavailable)
12. `test_mse_device_mismatch` - Device compatibility
13. `test_mse_single_value` - Scalar input
14. `test_mse_fractional_values` - Floating point precision
15. `test_mse_batch_independence` - Consistent results regardless of batching

**Test Results:**
```
14 passed, 1 skipped, 5 warnings
Coverage: 89% for metric_mse.py
```

---

### 4. Style Compliance

**Checks Passed:**
- `ty check` - No type errors
- `ruff check` - All linting checks passed
- Follows project code style (NumPy docstrings, PascalCase class names)
- Proper type hints throughout
- No unused imports

---

### 5. Documentation (`docs/user_manual/metrics/mse.md`)

**Documentation Includes:**
- **Overview** - What MSE is and when to use it
- **Mathematical Formula** - LaTeX equation
- **Properties** - Metric attributes and behavior
- **Usage Examples**:
  - Basic standalone usage
  - Integration with Pruna's evaluation framework
  - Model comparison example
- **Use Cases** - When MSE is appropriate and considerations
- **Example Results** - Concrete examples with expected outputs
- **Technical Details** - State accumulation, device handling, shape flexibility
- **Related Metrics** - RMSE, MAE, PSNR, SSIM
- **References** - External resources and contribution guide

---

## Acceptance Criteria Met

from pruna.evaluation.metrics.metric_mse import MSEMetric

# Initialize the metric
metric = MSEMetric()

# During evaluation loop:
for x, y in dataloader:
    preds = model(x)
    metric.update(x, y, preds)

# Get final result
result = metric.compute()
print(f"MSE: {result.result}")

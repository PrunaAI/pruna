# Mean Squared Error (MSE) Metric

## Overview

The MSE (Mean Squared Error) metric computes the mean squared error between model predictions and ground truth values. It's a fundamental metric for evaluating regression models and image quality assessment.

## Formula

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Where:
- $y_i$ is the ground truth value
- $\hat{y}_i$ is the predicted value
- $n$ is the total number of samples

## Properties

- **Metric Name**: `"mse"`
- **Higher is Better**: `False` (lower MSE indicates better performance)
- **Type**: `StatefulMetric` (accumulates across batches)
- **Default Call Type**: `"gt_y"` (compares ground truth vs outputs)

## Usage

### Basic Usage

```python
from pruna.evaluation.metrics.metric_mse import MSEMetric

# Initialize the metric
mse_metric = MSEMetric()

# Update with batches (automatic during evaluation)
for x, gt in dataloader:
    outputs = model(x)
    mse_metric.update(x, gt, outputs)

# Compute final MSE
result = mse_metric.compute()
print(f"MSE: {result.result:.6f}")
```

### With Pruna's Evaluation Framework

```python
from pruna import smash
from pruna.evaluation.task import Task

# Create evaluation task
task = Task(
    metrics=["mse"],  # Simply include "mse" in your metrics list
    # ... other task parameters
)

# Smash and evaluate
smashed_model = smash(
    model=your_model,
    eval_task=task,
    # ... other smash config
)
```

### Integration Example

```python
import torch
from pruna.evaluation.metrics.metric_mse import MSEMetric

# Example: Compare two models
def evaluate_model(model, test_data):
    metric = MSEMetric()
    
    for batch in test_data:
        x, ground_truth = batch
        predictions = model(x)
        metric.update(x, ground_truth, predictions)
    
    result = metric.compute()
    return result.result

# Usage
model_mse = evaluate_model(my_model, test_loader)
print(f"Model MSE: {model_mse:.4f}")
```

## When to Use MSE

### ✅ Good Use Cases

- **Regression Tasks**: Comparing continuous predictions to ground truth
- **Image Quality**: Measuring pixel-wise differences between images
- **Model Comparison**: Evaluating compression or quantization impact
- **Signal Processing**: Comparing reconstructed vs original signals

### ⚠️ Considerations

- MSE is sensitive to outliers (large errors are squared)
- Assumes errors are normally distributed
- Same MSE can represent different error distributions
- Consider using RMSE (√MSE) for interpretability in original units

## Example Results

### Perfect Match
```python
gt = torch.tensor([1.0, 2.0, 3.0, 4.0])
outputs = torch.tensor([1.0, 2.0, 3.0, 4.0])
# MSE = 0.0
```

### Constant Error
```python
gt = torch.tensor([1.0, 2.0, 3.0, 4.0])
outputs = torch.tensor([2.0, 3.0, 4.0, 5.0])  # +1 error each
# MSE = 1.0
```

### With Images
```python
# Comparing two 64x64 RGB images
gt_image = torch.randn(1, 3, 64, 64)
pred_image = gt_image + torch.randn_like(gt_image) * 0.1
# MSE ≈ 0.01 (depends on noise)
```

## Technical Details

### State Accumulation

The MSE metric accumulates squared errors across all batches in a list of tensors:

```python
self.add_state("squared_errors", [])  # List of tensors
```

### Computation

The final MSE is computed by:
1. Concatenating all squared error tensors
2. Computing the mean across all elements

```python
all_squared_errors = torch.cat(self.squared_errors)
mse_value = float(all_squared_errors.mean().item())
```

### Device Handling

The metric automatically handles device placement:
- Moves outputs to match ground truth device
- Works seamlessly with CPU and CUDA tensors

### Shape Flexibility

The metric flattens tensors for computation, supporting:
- 1D tensors (scalars)
- 2D tensors (batches)
- 3D tensors (sequences)
- 4D tensors (images: batch × channels × height × width)

## Related Metrics

- **RMSE** (Root Mean Squared Error): `√MSE` - same scale as original data
- **MAE** (Mean Absolute Error): Less sensitive to outliers
- **PSNR** (Peak Signal-to-Noise Ratio): `10 * log10(MAX²/MSE)` for images
- **SSIM** (Structural Similarity): Perceptual image quality metric

## References

- [Mean Squared Error - Wikipedia](https://en.wikipedia.org/wiki/Mean_squared_error)
- Pruna AI Documentation: [Customize Metrics](https://docs.pruna.ai/en/stable/docs_pruna/user_manual/customize_metric.html)

## Contributing

Found a bug or want to improve the MSE metric? See our [Contributing Guide](../../CONTRIBUTING.md).

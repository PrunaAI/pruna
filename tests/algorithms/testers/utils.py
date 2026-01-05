from typing import Any

from pruna.config.smash_config import SmashConfig


def restrict_recovery_time(smash_config: SmashConfig, algorithm_name: str) -> None:
    """Restrict the recovery time to a few batches to test iteration multiple time but as few as possible."""
    smash_config[f"{algorithm_name}_training_batch_size"] = 1
    smash_config[f"{algorithm_name}_num_epochs"] = 1
    # restrict the number of train and validation samples in the dataset
    smash_config.data.limit_datasets((2, 1, 1))  # 2 train, 1 val, 1 test


def get_model_sparsity(model: Any) -> float:
    """Get the sparsity of the model."""
    total_params = 0
    zero_params = 0
    for module in model.modules():
        if hasattr(module, "weight"):
            # Use the effective weight if pruning has been applied.
            weight = module.weight_orig * module.weight_mask if hasattr(module, "weight_mask") else module.weight
            total_params += weight.numel()
            zero_params += (weight == 0).sum().item()
    return zero_params / total_params if total_params > 0 else 0.0

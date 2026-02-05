from pathlib import Path
from typing import Any

from pruna.config.smash_config import SmashConfig
from pruna.data.diffuser_distillation_data_module import DiffusionDistillationDataModule


def restrict_recovery_time(smash_config: SmashConfig, algorithm_name: str) -> None:
    """Restrict the recovery time to a few batches to test iteration multiple time but as few as possible."""
    smash_config[f"{algorithm_name}_training_batch_size"] = 1
    smash_config[f"{algorithm_name}_num_epochs"] = 1
    # restrict the number of train and validation samples in the dataset
    smash_config.data.limit_datasets((2, 1, 1))  # 2 train, 1 val, 1 test


def replace_datamodule_with_distillation_datamodule(smash_config: SmashConfig, model: Any) -> None:
    """Create a distillation datamodule from the model and replace the datamodule in the smash config."""
    cache_dir = Path(smash_config.cache_dir) / f"{model.__class__.__name__.lower()}_distillation"
    distillation_data = DiffusionDistillationDataModule(
        pipeline=model,
        caption_datamodule=smash_config.data,
        save_path=cache_dir,
        seed=0,
    )
    smash_config.add_data(distillation_data)


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

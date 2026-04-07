from typing import Any

import torch
from PIL import Image

from pruna import PrunaModel
from pruna.algorithms.token_merging import TokenMerging
from pruna.engine.utils import get_device

from .base_tester import AlgorithmTesterBase


class TestTokenMerging(AlgorithmTesterBase):
    """Test the token merging algorithm."""

    models = ["vit_base", "vit_large"]
    reject_models = []
    hyperparameters = {"token_merging_r": 8}
    allow_pickle_files = True
    algorithm_class = TokenMerging
    metrics = []

    def pre_smash_hook(self, model: Any) -> None:
        """Hook to modify the model before smashing."""
        self.input_image = Image.open("husky.png")
        # Necessary to set the device to the same device as the model
        model.device = torch.device(get_device(model))
        self.original_pred = model(self.input_image)
        self.original_pred = [p["label"] for p in self.original_pred]

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Hook to modify the model after smashing."""
        # The _tome_info lives on the ToMeModelWrapper inside the pipeline's .model
        inner_model = model.model.model if hasattr(model.model, "model") else model.model
        assert hasattr(inner_model, "_tome_info"), "Inner model should have _tome_info attribute"

        # Pass the PIL image so the pipeline preprocesses it
        output = model(self.input_image)
        pred_labels = [p["label"] for p in output]
        assert inner_model._tome_info["size"] is not None, "Size should be set"
        # Check that the original top-1 is still in the top-5 after merging
        assert self.original_pred[0] in pred_labels[:5], "Original top-1 should remain in top-5"

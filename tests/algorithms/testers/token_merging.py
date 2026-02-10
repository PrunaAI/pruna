from typing import Any

import torch
from PIL import Image
from transformers import ImageClassificationPipeline

from pruna import PrunaModel
from pruna.algorithms.token_merging import TokenMerging
from pruna.engine.utils import get_device

from .base_tester import AlgorithmTesterBase


class TestTokenMerging(AlgorithmTesterBase):
    """Test the token merging algorithm."""

    models = ["vit_base", "vit_large"]
    reject_models = []
    hyperparameters = {"token_merging_r": 16}
    allow_pickle_files = True
    algorithm_class = TokenMerging
    metrics = ["total_macs", "latency"]

    def pre_smash_hook(self, model: Any) -> None:
        """Hook to modify the model before smashing."""
        # Store original model info
        self.input_image = Image.open("husky.png")
        # Necessary to set the device to the same device as the model
        model.device = torch.device(get_device(model))
        self.original_pred = model(self.input_image)
        self.original_pred = [p["label"] for p in self.original_pred]
        if isinstance(model, ImageClassificationPipeline):
            self.input_image = model.preprocess(self.input_image)["pixel_values"]
            self.input_image = self.input_image.to(model.device)

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Hook to modify the model after smashing."""
        # Verify that token merging was applied
        print(model.__class__)
        print(model.model.__class__)
        assert hasattr(model, "_tome_info"), "Model should have _tome_info attribute"

        output = model(self.input_image)
        pred_labels = [model.config.id2label[p] for p in output[0].topk(5).indices[0].tolist()]
        print("Output: ", pred_labels)
        print("Original: ", self.original_pred)
        assert model._tome_info["size"] is not None, "Size should be set"
        assert pred_labels[0] == self.original_pred[0], "Most likely class should remain the same"

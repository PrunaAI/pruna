from pruna.algorithms.padding_pruning import PaddingPruner
from pruna.engine.pruna_model import PrunaModel

from .base_tester import AlgorithmTesterBase


class TestPaddingPruning(AlgorithmTesterBase):
    """Test the padding pruning algorithm."""

    models = ["flux_tiny_random_with_tokenizer"]
    reject_models = ["opt_tiny_random"]
    allow_pickle_files = False
    algorithm_class = PaddingPruner
    metrics = ["cmmd"]

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Hook to modify the model after smashing."""
        assert hasattr(model, "padding_pruning_helper")
        model.text_encoder.resize_token_embeddings(model.smash_config.tokenizer.vocab_size)

        if hasattr(model, "text_encoder_2"):
            model.text_encoder_2.resize_token_embeddings(model.smash_config.tokenizer.vocab_size)

from copy import deepcopy

import pytest

from pruna.smash import check_model_compatibility

from ..algorithms import testers
from ..common import get_negative_examples_from_module


# marking compatibility tests as slow because we load a lot of models and they should hence not run on GitHub actions
@pytest.mark.slow
@pytest.mark.cpu
@pytest.mark.parametrize(
    "algorithm, model_fixture", get_negative_examples_from_module(testers), indirect=["model_fixture"]
)
def test_compatibility_reject_cuda(algorithm: str, model_fixture: tuple) -> None:
    """Test the compatibility check failure of a model with a given algorithm."""
    model, smash_config = model_fixture
    # add dummy processor and tokenizer to smash config reach model-compatibility checks
    smash_config.add_tokenizer("bert-base-uncased")
    smash_config.add_processor("bert-base-uncased")
    smash_config = deepcopy(smash_config)
    smash_config.add(algorithm)
    with pytest.raises(ValueError):
        check_model_compatibility(model, smash_config)

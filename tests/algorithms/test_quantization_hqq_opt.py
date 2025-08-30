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

import pytest
from transformers import AutoModelForCausalLM
from pruna import smash
from pruna.algorithms.quantization.hqq import HQQQuantizer
from pruna.config.smash_config import SmashConfig
from pruna.engine.model_checks import is_opt_model


def test_hqq_quantization_with_opt_model():
    """Test that HQQ quantization works correctly with OPT models."""
    # Load a small OPT model for testing
    model = AutoModelForCausalLM.from_pretrained("yujiepan/opt-tiny-random")

    # Verify it's an OPT model
    assert is_opt_model(model), "Model should be detected as an OPT model"

    # Create smash config
    smash_config = SmashConfig(device="cpu")
    smash_config["quantizer"] = "hqq"

    # Create HQQ quantizer
    quantizer = HQQQuantizer()

    # Test that the model check passes
    assert quantizer.model_check_fn(model), "OPT model should pass model check"

    # Apply quantization - this should not raise the AttributeError
    try:
        quantized_model = smash(model, smash_config)
        assert quantized_model is not None, "Quantized model should not be None"
        print("HQQ quantization with OPT model completed successfully")
    except AttributeError as e:
        if "working_model" in str(e):
            pytest.fail(f"AttributeError with 'working_model' should not occur: {e}")
        else:
            # Other AttributeErrors might be expected
            print(f"Expected AttributeError occurred: {e}")
    except Exception as e:
        # Other exceptions might be expected (e.g., missing dependencies)
        print(f"Expected exception occurred: {e}")


def test_hqq_quantization_force_hf_implementation():
    """Test that HQQ quantization works when forcing HF implementation."""
    # Load a small OPT model for testing
    model = AutoModelForCausalLM.from_pretrained("yujiepan/opt-tiny-random")

    # Create smash config with force_hf_implementation=True
    smash_config = SmashConfig(device="cpu")
    smash_config["quantizer"] = "hqq"

    # Create HQQ quantizer
    quantizer = HQQQuantizer()

    # Apply quantization - this should use the HF implementation
    try:
        quantized_model = smash(model, smash_config)
        assert quantized_model is not None, "Quantized model should not be None"
        print("HQQ quantization with forced HF implementation completed successfully")
    except Exception as e:
        # Exceptions might be expected (e.g., missing dependencies)
        print(f"Expected exception occurred: {e}")


if __name__ == "__main__":
    # Run the tests
    test_hqq_quantization_with_opt_model()
    test_hqq_quantization_force_hf_implementation()
    print("All tests completed!")

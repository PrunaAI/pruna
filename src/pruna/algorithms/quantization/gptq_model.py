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

import tempfile
from typing import Any, Dict

from ConfigSpace import OrdinalHyperparameter

from pruna.algorithms.quantization import PrunaQuantizer
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.config.smash_space import Boolean
from pruna.data.utils import recover_text_from_dataloader
from pruna.engine.model_checks import is_causal_lm
from pruna.engine.utils import safe_memory_cleanup


class GPTQQuantizer(PrunaQuantizer):
    """
    Implement GPTQ using GPTQModel.

    GPTQ is a post-training quantization technique that independently quantizes each row of the weight matrix to
    minimize error. The weights are quantized to int4, stored as int32, and then dequantized on the fly to fp16
    during inference, resulting in nearly 4x memory savings and faster performance due to custom kernels that take
    advantage of the lower precision.
    """

    algorithm_name: str = "gptq"
    references: dict[str, str] = {"GitHub": "https://github.com/ModelCloud/GPTQModel"}
    tokenizer_required: bool = True
    processor_required: bool = False
    runs_on: list[str] = ["cuda"]
    dataset_required: bool = True
    compatible_algorithms: dict[str, list[str]] = dict(compiler=["torch_compile"])
    required_install: str = (
        "You must first install the base package with ``pip install pruna`` "
        "before installing the GPTQ extension with ``pip install pruna[gptq] --extra-index-url https://prunaai.pythonanywhere.com/``"
    )

    def get_hyperparameters(self) -> list:
        """
        Configure all algorithm-specific hyperparameters with ConfigSpace.

        Returns
        -------
        list
            The hyperparameters.
        """
        return [
            OrdinalHyperparameter(
                "weight_bits",
                sequence=[2, 3, 4, 8],
                default_value=4,
                meta=dict(desc="Sets the number of bits to use for weight quantization."),
            ),
            Boolean(
                "use_exllama",
                default=False,
                meta=dict(desc="Whether to use exllama for quantization."),
            ),
            OrdinalHyperparameter(
                "group_size",
                sequence=[64, 128, 256],
                default_value=128,
                meta=dict(desc="Group size for quantization."),
            ),
        ]

    def model_check_fn(self, model: Any) -> bool:
        """
        Check if the model is a causal language model.

        Parameters
        ----------
        model : Any
            The model to check.

        Returns
        -------
        bool
            True if the model is a causal language model, False otherwise.
        """
        return is_causal_lm(model)

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Quantize the model.

        Parameters
        ----------
        model : Any
            The model to quantize.
        smash_config : SmashConfigPrefixWrapper
            The configuration for the quantization.

        Returns
        -------
        Any
            The quantized model.
        """
        imported_modules = self.import_algorithm_packages()
        with tempfile.TemporaryDirectory(prefix=smash_config["cache_dir"]) as temp_dir:
            # cast original model to CPU to free memory for smashed model
            if hasattr(model, "to"):
                model.to("cpu")
                safe_memory_cleanup()
            model.save_pretrained(temp_dir)
            # The tokenizer is saved because it is needed when loading a GPTQModel
            smash_config.tokenizer.save_pretrained(temp_dir)

            # dataset and tokenizer have been ensured to be set in the config
            val_dl = smash_config.val_dataloader()
            calib_data = recover_text_from_dataloader(val_dl, smash_config.tokenizer)  # type: ignore[arg-type]
            gptq_config = imported_modules["QuantizeConfig"](
                bits=smash_config["weight_bits"], group_size=smash_config["group_size"]
            )

            model = imported_modules["GPTQModel"].load(temp_dir, gptq_config)
            model.quantize(calib_data, batch_size=smash_config.batch_size)
            model.save(temp_dir)
            model = imported_modules["GPTQModel"].load(temp_dir)

        return model

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Provide a algorithm packages for the algorithm.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        try:
            from gptqmodel import GPTQModel, QuantizeConfig
        except ImportError:
            raise ImportError(f"gptqmodel is not installed. Please install it using {self.required_install}.")

        return dict(GPTQModel=GPTQModel, QuantizeConfig=QuantizeConfig)

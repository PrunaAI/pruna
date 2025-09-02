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

from typing import Any, Dict

from ConfigSpace import CategoricalHyperparameter

from pruna.algorithms.quantization import PrunaQuantizer
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.engine.model_checks import is_causal_lm, is_transformers_pipeline_with_causal_lm


class LLMCompressorQuantizer(PrunaQuantizer):
    """
    Implement AWQ using llmcompressor.

    Activation Aware Quantization (AWQ) is a state-of-the-art technique to quantize the weights of
    large language models which involves using a small calibration dataset to calibrate the model.
    The AWQ algorithm utilizes calibration data to derive scaling factors which reduce the dynamic
    range of weights while minimizing accuracy loss to the most salient weight values.
    """

    algorithm_name: str = "awq"
    references: dict[str, str] = {"GitHub": "https://github.com/vllm-project/llm-compressor"}
    tokenizer_required: bool = True
    processor_required: bool = False
    runs_on: list[str] = ["cuda"]
    dataset_required: bool = True
    compatible_algorithms: dict[str, list[str]] = dict()

    def get_hyperparameters(self) -> list:
        """
        Configure all algorithm-specific hyperparameters with ConfigSpace.

        Returns
        -------
        list
            The hyperparameters.
        """
        return [
            CategoricalHyperparameter(
                "quant_scheme",
                choices=["W4A16", "W4A16_ASYM"],
                default_value="W4A16",
                meta=dict(desc="Quantization scheme to use. Use symmetric quantization to avoid decompression issues."),
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
        return is_causal_lm(model) or is_transformers_pipeline_with_causal_lm(model)

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
        if is_transformers_pipeline_with_causal_lm(model):
            return self._apply_to_model_within_transformers_pipeline(model, smash_config)

        imported = self.import_algorithm_packages()
        recipe = [
            imported["AWQModifier"](
                ignore=["lm_head"],
                scheme=smash_config["quant_scheme"],
                targets=["Linear"],
            )
        ]

        dataset = smash_config.data.val_dataset

        # For text generation models, provide the tokenizer as processor to avoid AutoProcessor errors
        processor = smash_config.tokenizer if smash_config.tokenizer is not None else "bert-base-uncased"
        imported["oneshot"](model=model, recipe=recipe, dataset=dataset, processor=processor)
        return model

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Provide a algorithm packages for the algorithm.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        from llmcompressor import oneshot
        from llmcompressor.modifiers.awq import AWQModifier

        return {"oneshot": oneshot, "AWQModifier": AWQModifier}

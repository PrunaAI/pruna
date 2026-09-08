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

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import torch
from ConfigSpace import CategoricalHyperparameter, OrdinalHyperparameter

from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag
from pruna.algorithms.global_utils.quantization.dtypes import FLOAT8_DTYPE_FROM_NAME, parse_float8_dtype
from pruna.algorithms.global_utils.quantization.swap_linear import swap_linear
from pruna.algorithms.time_aware_fp8_diffusers.utils import (
    TimeAwareFp8Linear,
    TimeAwareScaleHelper,
    quantize_linear_layer_time_aware_fp8,
)
from pruna.config.hyperparameters import UnconstrainedHyperparameter
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.config.target_modules import (
    TARGET_MODULES_TYPE,
    TargetModules,
    filter_targeted_modules,
    map_targeted_nn_roots,
    target_backbone,
)
from pruna.engine.load_artifacts import TIME_AWARE_FP8_DIFFUSERS_ARTIFACTS_FUNCTION_NAME
from pruna.engine.model_checks import is_diffusers_model
from pruna.engine.pruna_model import PrunaModel
from pruna.engine.save import SAVE_FUNCTIONS
from pruna.engine.utils import safe_memory_cleanup
from pruna.logging.logger import pruna_logger


class TimeAwareFp8Diffusers(PrunaAlgorithmBase):
    """
    Time-aware fp8 quantization for diffusion models with calibration over full generations.

    Weights are statically quantized. The input (activation) scales are calibrated by
    running a few complete noise-to-image generations while recording per-layer input
    maxima keyed by the exact denoising timestep value. The observed timesteps define a
    lookup table (bin edges at midpoints between observed values).

    During inference, each layer reads the scale for the current timestep from the ``TimeAwareScaleHelper``.

    The time-aware quantization is motivated by the variance in the input distribution across timesteps. \
    It is a simplified version of \
    [Temporal Dynamic Quantization for Diffusion Models](https://arxiv.org/abs/2306.02316), \
    using a single scale per timestep (numerical) rather than a neural network to predict the scale.
    """

    algorithm_name = "time_aware_fp8_diffusers"
    group_tags: list[AlgorithmTag] = [AlgorithmTag.QUANTIZER]
    references: dict[str, str] = {
        "Temporal Dynamic Quantization for Diffusion Models": "https://arxiv.org/abs/2306.02316"
    }
    save_fn = SAVE_FUNCTIONS.save_before_apply
    tokenizer_required = False
    processor_required = False
    runs_on: list[str] = ["cpu", "cuda", "accelerate"]
    # dataset_required=True is not compatible with the save_before_apply hook,
    # as the dataset is not stored on save, and thus not available on load.
    dataset_required = False
    compatible_before: Iterable[str | AlgorithmTag] = ["padding_pruning", "qkv_diffusers"]
    compatible_after: Iterable[str | AlgorithmTag] = [
        "fastercache",
        "flash_attn3",
        "fora",
        "pab",
        "realesrgan_upscale",
        "ring_attn",
        "sage_attn",
        "torch_compile",
    ]
    disjointly_compatible_before: Iterable[str | AlgorithmTag] = [
        "hqq",
        "hqq_diffusers",
        "torchao",
        "static_fp8_diffusers",
    ]
    disjointly_compatible_after: Iterable[str | AlgorithmTag] = []

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
                "weight_float8_dtype",
                choices=list(FLOAT8_DTYPE_FROM_NAME),
                default_value="torch.float8_e4m3fn",
                meta={"desc": "The float8 dtype to use for weight quantization."},
            ),
            CategoricalHyperparameter(
                "input_float8_dtype",
                choices=list(FLOAT8_DTYPE_FROM_NAME),
                default_value="torch.float8_e4m3fn",
                meta={"desc": "The float8 dtype to use for input quantization."},
            ),
            OrdinalHyperparameter(
                "calibration_batches",
                sequence=list(range(1, 65)),
                default_value=4,
                meta={
                    "desc": (
                        "How many batches to use for calibrating the static input scales. Default is 4."
                        "The smash config `batch_size` is used. Default is 1."
                        "Samples are drawn from the dataset attached to the SmashConfig via `add_data`."
                    )
                },
            ),
            UnconstrainedHyperparameter(
                "calibration_num_inference_steps",
                default_value=None,
                meta={
                    "desc": (
                        "Number of denoising steps to use for each calibration generation. "
                        "When None (default), the pipeline's default `num_inference_steps` is used."
                    )
                },
            ),
            TargetModules(
                "target_modules",
                default_value=None,
                meta={
                    "desc": (
                        "Precise choices of which modules to quantize, "
                        "e.g. {include: ['transformer.*']} to quantize only the transformer in a diffusion pipeline. "
                        f"See the {TargetModules.documentation_name_with_link} documentation for more details."
                    )
                },
            ),
        ]

    def model_check_fn(self, model: Any) -> bool:
        """
        Check whether the algorithm is compatible with a model.

        Requires a diffusers model whose denoiser ``forward`` takes a parameter whose name
        is in :attr:`TimeAwareScaleHelper.timestep_arg_names`, so the shared scale helper can
        capture the denoising step.

        Parameters
        ----------
        model : Any
            The model to check.

        Returns
        -------
        bool
            Whether the algorithm is compatible with the model.
        """
        if not is_diffusers_model(model):
            return False
        denoiser = getattr(model, "transformer", None) or getattr(model, "unet", None)
        if denoiser is None:
            return False
        return TimeAwareScaleHelper._resolve_denoiser_timestep_arg(denoiser.forward) is not None

    def get_model_dependent_hyperparameter_defaults(
        self, model: Any, smash_config: SmashConfigPrefixWrapper
    ) -> dict[str, Any]:
        """
        Provide default `target_modules` using `target_backbone`, excluding sensitive modules.

        Extends the base backbone targets by excluding layers which are sensitive to FP8 quantization,
        including embedding, norm, lm_head, and proj_out layers. The matching pattern excludes
        only layers which match the patterns "*embed*", "*norm*", "*lm_head", and "proj_out".
        For a more precise control, the user should specify a custom matching pattern in the smash config.

        Parameters
        ----------
        model : Any
            The model to derive defaults from.
        smash_config : SmashConfigPrefixWrapper
            The algorithm-prefixed configuration.

        Returns
        -------
        dict[str, Any]
            A dictionary with a "target_modules" key defining which modules should be targeted by default.
        """
        target_modules = target_backbone(model)
        proj_out_patterns = ["unet.proj_out", "transformer.proj_out", "proj_out"]
        extra_exclude = ["*embed*", "*norm*", "*lm_head"] + proj_out_patterns
        target_modules["exclude"].extend(extra_exclude)
        return {"target_modules": target_modules}

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Quantize the model and calibrate the per-timestep activation scales before compilation.

        Parameters
        ----------
        model : Any
            The model to quantize.
        smash_config : SmashConfigPrefixWrapper
            The configuration for the quantization.

        Returns
        -------
        Any
            The quantized, calibrated model.
        """
        weight_float8_dtype = parse_float8_dtype(smash_config["weight_float8_dtype"])
        input_float8_dtype = parse_float8_dtype(smash_config["input_float8_dtype"])

        scale_helper = TimeAwareScaleHelper()
        quantized_layers: dict[int, TimeAwareFp8Linear] = {}

        target_modules: None | TARGET_MODULES_TYPE = smash_config["target_modules"]
        if target_modules is None:
            target_modules = self.get_model_dependent_hyperparameter_defaults(model, smash_config)["target_modules"]
            target_modules = cast(TARGET_MODULES_TYPE, target_modules)
        target_linear_modules = filter_targeted_modules(
            keep_targeted_fn=lambda module, path: isinstance(module, torch.nn.Linear),
            model=model,
            target_modules=target_modules,
        )

        def quantize_nn_module(attr_name: str | None, module: torch.nn.Module, subpaths: list[str]) -> Any:
            """
            Apply time-aware fp8 quantization to a nn.Module.

            Parameters
            ----------
            attr_name : str | None
                The name of the attribute in the model pointing to the nn.Module to quantize.
            module : torch.nn.Module
                The nn.Module to quantize.
            subpaths : list[str]
                The subpaths of the module to quantize.

            Returns
            -------
            torch.nn.Module
                The quantized nn.Module.
            """
            for subpath in subpaths:
                swap_linear(
                    module,
                    quantize_linear_layer_fn=quantize_linear_layer_time_aware_fp8,
                    path=subpath,
                    kwargs={
                        "weight_float8_dtype": weight_float8_dtype,
                        "input_float8_dtype": input_float8_dtype,
                        "scale_helper": scale_helper,
                    },
                )
            for submodule in module.modules():
                if isinstance(submodule, TimeAwareFp8Linear):
                    quantized_layers[id(submodule)] = submodule
            return module

        model = map_targeted_nn_roots(quantize_nn_module, model, target_linear_modules)

        denoiser = getattr(model, "transformer", None) or getattr(model, "unet", None)
        if denoiser is None:
            raise ValueError("No supported model backbone found.")
        scale_helper.register_on_denoiser(denoiser)

        # Persist the calibrated per-timestep scales such that loading
        # restores them instead of re-running calibration generations.
        if TIME_AWARE_FP8_DIFFUSERS_ARTIFACTS_FUNCTION_NAME not in smash_config.save_artifacts_fns:
            smash_config.save_artifacts_fns.append(TIME_AWARE_FP8_DIFFUSERS_ARTIFACTS_FUNCTION_NAME)

        # On load, the sidecar (loaded after this reapply) restores calibration timesteps/amaxes,
        # freezes scales for the current input dtype, and rebuilds the scale helper's scale table.
        is_loading = TIME_AWARE_FP8_DIFFUSERS_ARTIFACTS_FUNCTION_NAME in smash_config.load_artifacts_fns
        if is_loading:
            pruna_logger.info(
                "Loading time_aware_fp8_diffusers. Skipping calibration. Scales restored from artifacts."
            )
        elif smash_config.data is None:
            raise ValueError(
                "time_aware_fp8_diffusers requires a calibration dataset to fix the per-timestep input scales, "
                "but no data is attached to the SmashConfig (use `smash_config.add_data(...)`)."
            )
        else:
            self._calibrate(
                model,
                list(quantized_layers.values()),
                scale_helper,
                smash_config,
            )

        safe_memory_cleanup()
        return model

    @staticmethod
    def _calibrate(
        model: Any,
        layers: list[TimeAwareFp8Linear],
        scale_helper: TimeAwareScaleHelper,
        smash_config: SmashConfigPrefixWrapper,
    ) -> None:
        """
        Calibrate the per-timestep activation scales over complete noise-to-image generations.

        Runs up to ``smash_config["calibration_batches"]`` noise-to-image batch generations using samples drawn from
        the validation dataloader of the dataset attached to the SmashConfig via ``add_data``.

        Parameters
        ----------
        model : Any
            The quantized pipeline to run calibration generations on.
        layers : list[TimeAwareFp8Linear]
            The quantized layers to finalize once calibration completes.
        scale_helper : TimeAwareScaleHelper
            The scale helper to use for the calibration.
        smash_config : SmashConfigPrefixWrapper
            The configuration providing the calibration data and metadata.
        """
        pruna_model = PrunaModel(model)
        calibration_batches = int(smash_config["calibration_batches"])
        calibration_num_inference_steps = smash_config["calibration_num_inference_steps"]
        if calibration_num_inference_steps is not None:
            pruna_model.inference_handler.model_args["num_inference_steps"] = int(calibration_num_inference_steps)

        batch_count = 0
        with torch.no_grad():
            for batch in smash_config.val_dataloader():
                pruna_model.run_inference(batch)
                batch_count += 1
                if batch_count == calibration_batches:
                    break
            else:
                if batch_count == 0:
                    raise ValueError("Calibration dataset does not contain any batches.")
                pruna_logger.warning(
                    "Calibration dataset does not contain as many batches as requested. "
                    f"Only {batch_count} batches were used."
                )

        for layer in layers:
            layer.freeze_input_scales()
        scale_helper.prepare_for_inference(layers)

        pruna_logger.info(f"Bin edges: {scale_helper.bin_edges}")
        pruna_logger.info(
            f"time_aware_fp8_diffusers calibrated over {batch_count} batch(es), "
            f"{int(scale_helper.all_scales.shape[1])} timestep bin(s)."
        )

        del pruna_model
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

from typing import Iterable

from pruna.algorithms.base.tags import AlgorithmTag
from pruna.algorithms.global_utils.recovery.perp_recoverer import PERPRecoverer


class TextToImagePERP(PERPRecoverer):
    """
    PERP recoverer for text-to-image models.

    This recoverer is a general purpose PERP recoverer for text-to-image models using norm and bias finetuning
    as well as LoRA layers.

    Parameters
    ----------
    use_lora : bool
        Whether to use LoRA adapters.
    use_in_place : bool
        Whether to use norm and bias finetuning which will modify the model in place.
    """

    algorithm_name: str = "text_to_image_perp"
    tokenizer_required: bool = False
    compatible_before: Iterable[str | AlgorithmTag] = ["quanto", "torch_dynamic", "deepcache"]
    compatible_after: Iterable[str | AlgorithmTag] = ["torch_compile"]
    runs_on: list[str] = ["cuda"]

    def __init__(self, use_lora: bool = True, use_in_place: bool = True) -> None:
        super().__init__(task_name="text_to_image", use_lora=use_lora, use_in_place=use_in_place, is_distillation=False)


class TextToImageInPlacePERP(TextToImagePERP):
    """
    PERP recoverer for text-to-image models without LoRA adapters.

    This is the same as ``text_to_image_perp``, but without LoRA layers which add extra computations and thus slow down
    the inference of the final model.
    """

    algorithm_name = "text_to_image_inplace_perp"

    def __init__(self) -> None:
        super().__init__(use_lora=False)


class TextToImageLoRA(TextToImagePERP):
    """
    LoRA recoverer for text-to-image models.

    This recoverer attaches LoRA adapters to the model and finetunes them.
    """

    algorithm_name: str = "text_to_image_lora"

    def __init__(self) -> None:
        super().__init__(use_in_place=False)


class TextToTextPERP(PERPRecoverer):
    """
    PERP recoverer for text-to-text models.

    This recoverer is a general purpose PERP recoverer for text-to-text models using norm, head and bias finetuning
    as well as LoRA layers.

    Parameters
    ----------
    use_lora : bool
        Whether to use LoRA adapters.
    use_in_place : bool
        Whether to use norm, bias and head finetuning which will modify the model in place.
    """

    algorithm_name: str = "text_to_text_perp"
    tokenizer_required: bool = True
    compatible_before: Iterable[str | AlgorithmTag] = ["half", "quanto", "torch_dynamic"]
    compatible_after: Iterable[str | AlgorithmTag] = ["torch_compile"]

    def __init__(self, use_lora: bool = True, use_in_place: bool = True) -> None:
        super().__init__(task_name="text_to_text", use_lora=use_lora, use_in_place=use_in_place, is_distillation=False)


class TextToTextInPlacePERP(TextToTextPERP):
    """
    PERP recoverer for text-to-text models without LoRA adapters.

    This is the same as ``text_to_text_perp``, but without LoRA layers which add extra computations and thus slow down
    the inference of the final model.
    """

    algorithm_name: str = "text_to_text_inplace_perp"

    def __init__(self) -> None:
        super().__init__(use_lora=False)


class TextToTextLoRA(TextToTextPERP):
    """
    LoRA recoverer for text-to-text models.

    This recoverer attaches LoRA adapters to the model and finetunes them.
    """

    algorithm_name: str = "text_to_text_lora"

    def __init__(self) -> None:
        super().__init__(use_in_place=False)

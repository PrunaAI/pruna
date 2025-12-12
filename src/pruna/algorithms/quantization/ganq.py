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


from typing import Any

import torch
from ConfigSpace import Constant, OrdinalHyperparameter

from pruna.algorithms.quantization import PrunaQuantizer
from pruna.config.hyperparameters import Boolean
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.engine.model_checks import (
    is_causal_lm,
)

from pruna.engine.utils import safe_memory_cleanup
from pruna.logging.filter import SuppressOutput
from pruna.logging.logger import pruna_logger


class GANQQuantizer(PrunaQuantizer):
    """GPU-Adaptive Non-Uniform Quantization (GANQ).

    GANQ performs layer-wise LUT-based non-uniform quantization by
    alternating optimization of codebook (T) and selection matrix (S).
    It adapts to the distribution of each layer’s weights and supports
    optional normalization and outlier handling."""

    algorithm_name: str = "ganq"
    references: dict[str, str] = {
        "GitHub": "https://github.com/Evans-Z/GANQ",
        "Article": "https://arxiv.org/pdf/2501.12956",
    }
    runs_on: list[str] = ["cuda"]
    dataset_required: bool = True
    compatible_algorithms = dict(compiler=["torch_compile"])
    processor_required: bool = False
    tokenizer_required: bool = False

    def get_hyperparameters(self):
        return [
            OrdinalHyperparameter(
                "weight_bits",
                [3, 4],
                default_value=4,
                meta=dict(desc="Bit width for weight quantization."),
            ),
            OrdinalHyperparameter(
                "max_epoch",
                [5, 10, 20],
                default_value=10,
                meta=dict(desc="Number of GANQ alternating steps."),
            ),
            Boolean(
                "pre_process",
                default=True,
                meta=dict(
                    desc="Normalize weights with median/IQR before quantization."
                ),
            ),
            Constant("sparsity", value=0.0),
            Constant("full_rows", value=0),
        ]

    def model_check_fn(self, model: Any) -> bool:
        return is_causal_lm(model)

    def _apply(self, model, smash_config: SmashConfigPrefixWrapper):
        imported_packages = self.import_algorithm_packages()
        GANQ, find_layers = imported_packages["GANQ"], imported_packages["find_layers"]

        pruna_logger.info("Running GANQ layer-wise quantization...")
        model.eval()
        device = smash_config["device"]

        val_dl = smash_config.val_dataloader()

        # TODO: Align on whether to use batch, or use calib data and use tokenizer
        # calib_data = recover_text_from_dataloader(val_dl, smash_config.tokenizer)  # type: ignore[arg-type]
        # tokenizer = smash_config.tokenizer
        # max_len = getattr(model.config, "max_position_embeddings")
        safe_memory_cleanup()

        layers = model.model.layers
        print(layers)
        inps = []
        cache = {"i": 0, "attention_mask": None, "position_ids": None}

        # Note: This is only used to capture inputs to layer 0, hence the value error
        class Catcher(torch.nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                if cache["i"] < len(inps):
                    inps[cache["i"]] = inp
                else:
                    inps.append(inp)
                cache["i"] += 1
                cache["attention_mask"] = kwargs.get("attention_mask", None)
                cache["position_ids"] = kwargs.get("position_ids", None)
                raise ValueError

        layers[0] = Catcher(layers[0])

        for batch in val_dl:
            try:
                model(batch[0].to(device))
            except ValueError:
                pass

        layers[0] = layers[0].module
        pruna_logger.info(f"Captured {len(inps)} input samples for layer 0.")

        outs = torch.zeros_like(inps[0])
        attention_mask = cache["attention_mask"]
        position_ids = cache["position_ids"]

        for i, layer in enumerate(layers):
            layer = layer.to(device)
            layer_dict = find_layers(layer)

            pruna_logger.info(f"Quantizing layer {i} ({len(layer_dict)} submodules)...")

            # Group submodules like the official repo
            sequential_groups = [
                ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]

            for group in sequential_groups:
                subset = {n: layer_dict[n] for n in group if n in layer_dict}
                if not subset:
                    continue

                gpts = {
                    name: GANQ(mod, model_type="hf") for name, mod in subset.items()
                }

                def make_hook(name):
                    def hook_fn(_, inp, out):
                        gpts[name].add_batch(inp[0].detach(), out.detach())

                    return hook_fn

                handles = [
                    mod.register_forward_hook(make_hook(name))
                    for name, mod in subset.items()
                ]

                with torch.no_grad():
                    for j in range(len(inps)):

                        # Note: This is different from author's implementation, because of change in RoPE handling in transformers version 4.42 and 4.53
                        # refer original implementation here - https://github.com/Evans-Z/GANQ/blob/176a87701fd0e07aea1ccd4f3faff84871d79f44/llama.py#L127
                        hs = inps[j].to(device)  # [1, seq, hidden]
                        seq_len = hs.shape[1]
                        pos_ids = position_ids[:1, :seq_len].to(device)
                        cos, sin = model.model.rotary_emb(hs, pos_ids)
                        cache_pos = torch.arange(seq_len, device=device)

                        outs = layer(
                            hs,
                            attention_mask=attention_mask,
                            position_ids=pos_ids,
                            position_embeddings=(cos, sin),
                        )[0]

                for h in handles:
                    h.remove()

                # Run quantization per submodule
                for name, gpt in gpts.items():
                    pruna_logger.info(f"Quantizing {name}...")
                    gpt.fasterquant(
                        # TODO: Figure out a way to pass these parameters
                        # sparsity=smash_config["sparsity"],
                        # bits=smash_config["weight_bits"],
                        # max_epoch=smash_config["max_epoch"],
                        # pre_process=smash_config["pre_process"],
                        # full_rows=smash_config["full_rows"],
                    )
                    gpt.free()

            inps = [outs]
            layer = layer.cpu()
            torch.cuda.empty_cache()

        pruna_logger.info("✅ GANQ quantization complete.")
        return model

    def import_algorithm_packages(self):

        with SuppressOutput():
            from pruna.algorithms.quantization.backends.ganq import (
                GANQ,
                LUTQuant,
                find_layers,
            )

        return dict(
            GANQ=GANQ,
            LUTQuant=LUTQuant,
            find_layers=find_layers,
        )

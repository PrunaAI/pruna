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
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, List

from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag as tags
from pruna.config.hyperparameters import Boolean
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.engine.save import SAVE_FUNCTIONS
from pruna.logging.logger import pruna_logger

class ShortGPT(PrunaAlgorithmBase):
    """
    ShortGPT algorithm for pruning transformer layers using a block influence metric.

    ShortGPT identifies and prunes less important blocks in transformer models based on their 
    BI scores, which uses the similarity between a layers input and output to measure its importance.
    """

    algorithm_name: str = "shortgpt"
    group_tags: list[str] = [tags.PRUNER]
    references: dict[str, str] = {
        "Paper": "https://arxiv.org/pdf/2403.03853",
    }
    save_fn = SAVE_FUNCTIONS.pickled
    tokenizer_required: bool = True
    dataset_required: bool = True
    processor_required: bool = False
    runs_on: list[str] = ["cuda", "cpu"]

    def get_hyperparameters(self) -> list:
        from ConfigSpace import CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter
        return [
            CategoricalHyperparameter(
                "metric_type", ["BI"], default_value="BI",
                meta=dict(desc="Metric type for layer importance: Block Influence")
            ),
            UniformFloatHyperparameter(
                "prune_ratio", lower=0.0, upper=0.8, default_value=0.25,
                meta=dict(desc="Fraction of layers to prune")
            ),
            Boolean("angular", meta=dict(desc="Use angular distance for BI computation")),
            UniformIntegerHyperparameter(
                "calibration_samples", lower=8, upper=512, default_value=64,
                meta=dict(desc="Number of calibration samples to compute metrics")
            ),
        ]
    

    @staticmethod
    @torch.inference_mode()
    def compute_block_influence(model, tokenizer, texts, angular=False, device="cuda", max_samples=64):
        model.eval().to(device)
        num_layers = len(model.model.layers)
        bis = torch.zeros(num_layers + 1, device=device)
        counts = 0

        for text in tqdm(texts[:max_samples], desc="Computing Block Influence"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            input_ids = inputs["input_ids"]
            hiddens = []

            def hook_fn(_, __, out):
                if isinstance(out, tuple): out = out[0]
                hiddens.append(out)

            handles = [layer.register_forward_hook(hook_fn) for layer in model.model.layers]
            _ = model(input_ids=input_ids)
            for h in handles: h.remove()

            hiddens.insert(0, model.model.embed_tokens(input_ids))
            hiddens.append(model.model.norm(hiddens[-1]))

            for i in range(len(hiddens) - 1):
                in_h, out_h = hiddens[i].float(), hiddens[i + 1].float()
                cos = F.cosine_similarity(
                    in_h.view(-1, in_h.shape[-1]),
                    out_h.view(-1, out_h.shape[-1]),
                    dim=-1
                )
                if angular:
                    cos = cos.clamp(-1 + 1e-7, 1 - 1e-7)
                    bi = torch.acos(cos).mean() / np.pi
                else:
                    bi = (1 - cos).mean()
                bis[i] += bi
            counts += 1

        bis /= counts
        return bis.tolist()
    
    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        device = smash_config["device"]
        model = model.to(device)
        model.eval()

        pruna_logger.info(f"[ShortGPT] Starting layer pruning for model on device: {device}")
        pruna_logger.info(f"[ShortGPT] Model depth: {len(model.model.layers)}")
        pruna_logger.info(f"[ShortGPT] Model parameters: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
        tokenizer = smash_config["tokenizer"]
        
        texts = smash_config["texts"]

        metric_type = smash_config["metric_type"]
        prune_ratio = smash_config["prune_ratio"]
        angular = smash_config["angular"]

        pruna_logger.info(f"[ShortGPT] Running {metric_type}-based layer pruning (ratio={prune_ratio:.2f})")

        scores = self.compute_block_influence(model, tokenizer, texts, angular=angular, device=device)

        num_layers = len(model.model.layers)
        n_prune = int(prune_ratio * num_layers)
        layer_scores = np.array(scores[1:num_layers+1])  # skip embedding span

        prune_indices = np.argsort(layer_scores)[:n_prune].tolist()
        keep_indices = [i for i in range(num_layers) if i not in prune_indices]

        pruna_logger.info(f"[ShortGPT] Pruning {n_prune}/{num_layers} layers: {prune_indices}")

        kept_layers = torch.nn.ModuleList([layer for i, layer in enumerate(model.model.layers) if i in keep_indices])
        model.model.layers = kept_layers

        pruna_logger.info(f"[ShortGPT] Pruned model depth: {len(model.model.layers)}")
        pruna_logger.info(f"[ShortGPT] Pruned model parameters: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")

        return model


    def model_check_fn(self, model):
        return isinstance(model, torch.nn.Module)
    

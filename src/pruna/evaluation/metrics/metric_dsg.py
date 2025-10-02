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

from typing import Any, Dict, List, Optional, Union
import warnings

import torch

from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import SINGLE, get_call_type_for_single_metric, metric_data_processor
from pruna.logging.logger import pruna_logger

METRIC_DSG = "dsg"


@MetricRegistry.register(METRIC_DSG)
class DSGMetric(StatefulMetric):
    """
    Davidsonian Scene Graph (DSG) metric for text-to-image evaluation.
    
    DSG is an automatic evaluation framework for text-to-image models inspired by 
    formal semantics to assess faithfulness. It generates atomic semantic tuples,
    dependency graphs, and questions from text prompts, then evaluates generated
    images based on VQA consistency.
    
    The metric follows the DSG evaluation pipeline:
    1. Generate atomic semantic tuples from text prompts
    2. Create dependency graphs between tuples  
    3. Convert tuples to natural language questions
    4. Answer questions using VQA on generated images
    5. Apply dependency constraints to get final score
    
    Parameters
    ----------
    llm_model : str, default="gpt-3.5-turbo"
        LLM model for question generation. Supported: gpt-3.5-turbo, gpt-4.
    vqa_model : str, default="mplug-large"
        VQA model for answering questions. Supported: mplug-large, instructblip.
    api_key : Optional[str], default=None
        API key for LLM (required for OpenAI models).
    call_type : str, default=SINGLE
        Call type for the metric.
    max_questions : int, default=20
        Maximum number of questions to generate per prompt.
    temperature : float, default=0.0
        Temperature for LLM generation.
    device : str | torch.device | None, optional
        The device to be used for VQA model. Default is None.
    
    References
    ----------
    Davidsonian Scene Graph: Improving Reliability in Fine-grained Evaluation 
    for Text-to-Image Generation (ICLR 2024)
    https://github.com/j-min/DSG
    """

    default_call_type: str = "x_y"  # Input prompts (x) and generated images (y)
    higher_is_better: bool = True
    metric_name: str = METRIC_DSG

    def __init__(
        self,
        llm_model: str = "gpt-3.5-turbo", 
        vqa_model: str = "mplug-large",
        api_key: Optional[str] = None,
        call_type: str = SINGLE,
        max_questions: int = 20,
        temperature: float = 0.0,
        device: str | torch.device | None = None,
        **kwargs
    ) -> None:
        """Initialize DSG metric."""
        super().__init__(**kwargs)
        self.llm_model = llm_model
        self.vqa_model = vqa_model
        self.api_key = api_key
        self.max_questions = max_questions
        self.temperature = temperature
        self.device = device
        self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type)

        # Initialize state variables for accumulating scores
        self.add_state("total_score", torch.zeros(1))
        self.add_state("count", torch.zeros(1))

        # Initialize models lazily to avoid import errors if DSG not installed
        self._llm = None
        self._vqa = None
        self._dsg_generator = None
        self._models_initialized = False

    def _initialize_models(self) -> None:
        """Initialize LLM and VQA models lazily."""
        if self._models_initialized:
            return
            
        try:
            # Try to import DSG dependencies
            import os
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # Avoid tokenizer warnings
            
            from dsg.openai_utils import OpenAILLM
            from dsg.query_utils import DSGGenerator
            
            # Initialize LLM
            if "gpt" in self.llm_model.lower():
                self._llm = OpenAILLM(model=self.llm_model, api_key=self.api_key)
            else:
                raise ValueError(f"Unsupported LLM model: {self.llm_model}. Supported: gpt-3.5-turbo, gpt-4")
            
            # Initialize VQA model
            if "mplug" in self.vqa_model.lower():
                from dsg.vqa_utils import mPLUGVQA
                self._vqa = mPLUGVQA(device=self.device)
            elif "instructblip" in self.vqa_model.lower():
                from dsg.vqa_utils import InstructBLIPVQA
                self._vqa = InstructBLIPVQA(device=self.device)
            else:
                raise ValueError(f"Unsupported VQA model: {self.vqa_model}. Supported: mplug-large, instructblip")
                
            # Initialize DSG generator
            self._dsg_generator = DSGGenerator()
            self._models_initialized = True
            
            pruna_logger.info(f"DSG metric initialized with LLM: {self.llm_model}, VQA: {self.vqa_model}")
            
        except ImportError as e:
            error_msg = (
                "DSG dependencies not found. Please install with: "
                "'pip install git+https://github.com/j-min/DSG.git' "
                "and ensure you have the required dependencies (transformers, torch, etc.)"
            )
            pruna_logger.error(error_msg)
            raise ImportError(error_msg) from e
        except Exception as e:
            pruna_logger.error(f"Failed to initialize DSG models: {e}")
            raise

    def _generate_dsg(self, text: str) -> tuple:
        """Generate DSG (tuples, dependency, and questions) from text."""
        try:
            # Generate DSG components using the DSG library
            id2tuples, id2dependency, id2questions = self._dsg_generator.generate_dsg(
                text, self._llm, max_questions=self.max_questions
            )
            return id2tuples, id2dependency, id2questions
        except Exception as e:
            pruna_logger.warning(f"DSG generation failed for text '{text[:50]}...': {e}")
            # Return empty results on failure
            return {}, {}, {}

    def _evaluate_image_dsg(self, text: str, image: Any) -> float:
        """Evaluate a generated image with DSG."""
        try:
            # Generate DSG from text
            id2tuples, id2dependency, id2questions = self._generate_dsg(text)
            
            if not id2questions:
                pruna_logger.warning(f"No questions generated for text: {text[:50]}...")
                return 0.0
            
            # Answer questions with the generated image
            id2scores = {}
            for question_id, question in id2questions.items():
                try:
                    answer = self._vqa.answer(image, question)
                    # DSG expects yes/no answers, convert to binary score
                    score = 1.0 if answer.lower().strip() in ['yes', 'true', '1'] else 0.0
                    id2scores[question_id] = score
                except Exception as e:
                    pruna_logger.warning(f"VQA failed for question '{question}': {e}")
                    # If VQA fails, assign score of 0
                    id2scores[question_id] = 0.0
            
            # Apply dependency constraints: zero-out scores from invalid questions
            for question_id, parent_ids in id2dependency.items():
                if question_id in id2scores and parent_ids:
                    # Check if any parent questions were answered 'no'
                    any_parent_answered_no = False
                    for parent_id in parent_ids:
                        if parent_id in id2scores and id2scores[parent_id] == 0:
                            any_parent_answered_no = True
                            break
                    if any_parent_answered_no:
                        id2scores[question_id] = 0.0
            
            # Calculate the final score by averaging
            if len(id2scores) == 0:
                return 0.0
            
            average_score = sum(id2scores.values()) / len(id2scores)
            return float(average_score)
            
        except Exception as e:
            pruna_logger.warning(f"DSG evaluation failed for text '{text[:50]}...': {e}")
            return 0.0

    def update(self, x: List[Any] | torch.Tensor, gt: List[Any] | torch.Tensor, outputs: Any) -> None:
        """
        Update metric state with current batch.
        
        Parameters
        ----------
        x : List[Any] | torch.Tensor
            Input prompts (text).
        gt : List[Any] | torch.Tensor  
            Ground truth data (not used for DSG).
        outputs : Any
            Generated images from the model.
        """
        # Initialize models on first use
        self._initialize_models()
        
        # Process data according to call_type
        metric_data = metric_data_processor(x, gt, outputs, self.call_type)
        
        if self.call_type in ["x_y", "y_x"]:
            # For text-to-image evaluation: x are prompts, outputs are images
            if self.call_type == "x_y":
                prompts, images = metric_data
            else:  # y_x
                images, prompts = metric_data
        else:
            pruna_logger.warning(f"Call type {self.call_type} not fully supported for DSG, using first two elements")
            prompts, images = metric_data[0], metric_data[1]
        
        # Ensure we have sequences
        if not isinstance(prompts, (list, tuple)):
            prompts = [prompts]
        if not isinstance(images, (list, tuple)):
            images = [images]
            
        # Evaluate each prompt-image pair
        batch_size = min(len(prompts), len(images))
        
        for i in range(batch_size):
            prompt = prompts[i]
            image = images[i]
            
            # Convert prompt to string if needed
            if not isinstance(prompt, str):
                prompt = str(prompt)
            
            try:
                score = self._evaluate_image_dsg(prompt, image)
                self.total_score += score
                self.count += 1
            except Exception as e:
                pruna_logger.warning(f"Failed to evaluate prompt-image pair {i}: {e}")
                # Count failed evaluations as 0 score
                self.total_score += 0.0
                self.count += 1

    def compute(self) -> MetricResult:
        """
        Compute final DSG score.
        
        Returns
        -------
        MetricResult
            The computed DSG metric result with average score across all samples.
        """
        if self.count == 0:
            final_score = 0.0
            pruna_logger.warning("No samples processed for DSG metric")
        else:
            final_score = float(self.total_score / self.count)
        
        # Prepare metadata
        params = {
            "llm_model": self.llm_model,
            "vqa_model": self.vqa_model,
            "max_questions": self.max_questions,
            "temperature": self.temperature,
            "call_type": self.call_type,
            "total_samples": int(self.count),
        }
        
        pruna_logger.info(f"DSG metric computed: {final_score:.4f} (from {self.count} samples)")
        
        return MetricResult(self.metric_name, params, final_score)
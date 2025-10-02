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
import torch
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

from pruna.evaluation.metrics.metric_dsg import DSGMetric, METRIC_DSG
from pruna.evaluation.metrics.registry import MetricRegistry


class TestDSGMetric:
    """Test suite for DSG metric."""

    def test_metric_registration(self):
        """Test that DSG metric is properly registered."""
        assert METRIC_DSG in MetricRegistry._registry
        metric = MetricRegistry.get(METRIC_DSG)
        assert isinstance(metric, DSGMetric)

    def test_metric_initialization_default(self):
        """Test DSG metric initialization with default parameters."""
        metric = DSGMetric()
        assert metric.metric_name == METRIC_DSG
        assert metric.higher_is_better is True
        assert metric.llm_model == "gpt-3.5-turbo"
        assert metric.vqa_model == "mplug-large"
        assert metric.max_questions == 20
        assert metric.temperature == 0.0
        assert metric.call_type == "x_y"
        assert metric._models_initialized is False

    def test_metric_initialization_custom(self):
        """Test DSG metric initialization with custom parameters."""
        metric = DSGMetric(
            llm_model="gpt-4",
            vqa_model="instructblip",
            api_key="test-key",
            max_questions=15,
            temperature=0.5,
            call_type="y_x"
        )
        assert metric.llm_model == "gpt-4"
        assert metric.vqa_model == "instructblip"
        assert metric.api_key == "test-key"
        assert metric.max_questions == 15
        assert metric.temperature == 0.5
        assert metric.call_type == "y_x"

    def test_state_initialization(self):
        """Test that state variables are properly initialized."""
        metric = DSGMetric()
        assert hasattr(metric, 'total_score')
        assert hasattr(metric, 'count')
        assert torch.equal(metric.total_score, torch.zeros(1))
        assert torch.equal(metric.count, torch.zeros(1))

    @patch('pruna.evaluation.metrics.metric_dsg.OpenAILLM')
    @patch('pruna.evaluation.metrics.metric_dsg.mPLUGVQA')
    @patch('pruna.evaluation.metrics.metric_dsg.DSGGenerator')
    def test_model_initialization_mplug(self, mock_dsg_gen, mock_vqa, mock_llm):
        """Test successful model initialization with mPLUG VQA."""
        # Setup mocks
        mock_llm_instance = Mock()
        mock_vqa_instance = Mock()
        mock_dsg_instance = Mock()
        
        mock_llm.return_value = mock_llm_instance
        mock_vqa.return_value = mock_vqa_instance
        mock_dsg_gen.return_value = mock_dsg_instance
        
        metric = DSGMetric(llm_model="gpt-3.5-turbo", vqa_model="mplug-large")
        
        # Models should not be initialized yet
        assert not metric._models_initialized
        
        # Initialize models
        metric._initialize_models()
        
        # Check models are initialized
        assert metric._models_initialized
        assert metric._llm == mock_llm_instance
        assert metric._vqa == mock_vqa_instance
        assert metric._dsg_generator == mock_dsg_instance
        
        # Verify calls
        mock_llm.assert_called_once_with(model="gpt-3.5-turbo", api_key=None)
        mock_vqa.assert_called_once_with(device=None)
        mock_dsg_gen.assert_called_once()

    @patch('pruna.evaluation.metrics.metric_dsg.OpenAILLM')
    @patch('pruna.evaluation.metrics.metric_dsg.InstructBLIPVQA')
    @patch('pruna.evaluation.metrics.metric_dsg.DSGGenerator')
    def test_model_initialization_instructblip(self, mock_dsg_gen, mock_vqa, mock_llm):
        """Test successful model initialization with InstructBLIP VQA."""
        # Setup mocks
        mock_llm_instance = Mock()
        mock_vqa_instance = Mock()
        mock_dsg_instance = Mock()
        
        mock_llm.return_value = mock_llm_instance
        mock_vqa.return_value = mock_vqa_instance
        mock_dsg_gen.return_value = mock_dsg_instance
        
        metric = DSGMetric(llm_model="gpt-4", vqa_model="instructblip", device="cuda")
        metric._initialize_models()
        
        # Verify calls
        mock_llm.assert_called_once_with(model="gpt-4", api_key=None)
        mock_vqa.assert_called_once_with(device="cuda")

    def test_model_initialization_import_error(self):
        """Test handling of missing DSG dependencies."""
        with patch('pruna.evaluation.metrics.metric_dsg.OpenAILLM', side_effect=ImportError("DSG not installed")):
            metric = DSGMetric()
            
            with pytest.raises(ImportError, match="DSG dependencies not found"):
                metric._initialize_models()

    def test_unsupported_llm_model(self):
        """Test error handling for unsupported LLM model."""
        with patch('pruna.evaluation.metrics.metric_dsg.OpenAILLM'):
            metric = DSGMetric(llm_model="unsupported-model")
            
            with pytest.raises(ValueError, match="Unsupported LLM model"):
                metric._initialize_models()

    def test_unsupported_vqa_model(self):
        """Test error handling for unsupported VQA model."""
        with patch('pruna.evaluation.metrics.metric_dsg.OpenAILLM'):
            with patch('pruna.evaluation.metrics.metric_dsg.DSGGenerator'):
                metric = DSGMetric(vqa_model="unsupported-vqa")
                
                with pytest.raises(ValueError, match="Unsupported VQA model"):
                    metric._initialize_models()

    @patch('pruna.evaluation.metrics.metric_dsg.OpenAILLM')
    @patch('pruna.evaluation.metrics.metric_dsg.mPLUGVQA')
    @patch('pruna.evaluation.metrics.metric_dsg.DSGGenerator')
    def test_generate_dsg_success(self, mock_dsg_gen, mock_vqa, mock_llm):
        """Test successful DSG generation."""
        # Setup mocks
        mock_llm_instance = Mock()
        mock_vqa_instance = Mock()
        mock_dsg_instance = Mock()
        
        mock_llm.return_value = mock_llm_instance
        mock_vqa.return_value = mock_vqa_instance
        mock_dsg_gen.return_value = mock_dsg_instance
        
        # Mock DSG generation
        expected_tuples = {"q1": "Is there a cat?", "q2": "Is the cat orange?"}
        expected_dependency = {"q2": ["q1"]}
        expected_questions = {"q1": "Is there a cat in the image?", "q2": "Is the cat orange in color?"}
        
        mock_dsg_instance.generate_dsg.return_value = (expected_tuples, expected_dependency, expected_questions)
        
        metric = DSGMetric()
        metric._initialize_models()
        
        # Test DSG generation
        tuples, dependency, questions = metric._generate_dsg("A orange cat sitting on a chair")
        
        assert tuples == expected_tuples
        assert dependency == expected_dependency
        assert questions == expected_questions
        
        mock_dsg_instance.generate_dsg.assert_called_once_with(
            "A orange cat sitting on a chair", mock_llm_instance, max_questions=20
        )

    @patch('pruna.evaluation.metrics.metric_dsg.OpenAILLM')
    @patch('pruna.evaluation.metrics.metric_dsg.mPLUGVQA')
    @patch('pruna.evaluation.metrics.metric_dsg.DSGGenerator')
    def test_generate_dsg_failure(self, mock_dsg_gen, mock_vqa, mock_llm):
        """Test DSG generation failure handling."""
        # Setup mocks
        mock_llm_instance = Mock()
        mock_vqa_instance = Mock()
        mock_dsg_instance = Mock()
        
        mock_llm.return_value = mock_llm_instance
        mock_vqa.return_value = mock_vqa_instance
        mock_dsg_gen.return_value = mock_dsg_instance
        
        # Mock DSG generation failure
        mock_dsg_instance.generate_dsg.side_effect = Exception("DSG generation failed")
        
        metric = DSGMetric()
        metric._initialize_models()
        
        # Test DSG generation failure
        tuples, dependency, questions = metric._generate_dsg("Invalid prompt")
        
        # Should return empty dicts on failure
        assert tuples == {}
        assert dependency == {}
        assert questions == {}

    @patch('pruna.evaluation.metrics.metric_dsg.OpenAILLM')
    @patch('pruna.evaluation.metrics.metric_dsg.mPLUGVQA')
    @patch('pruna.evaluation.metrics.metric_dsg.DSGGenerator')
    def test_evaluate_image_dsg_success(self, mock_dsg_gen, mock_vqa, mock_llm):
        """Test successful image evaluation with DSG."""
        # Setup mocks
        mock_llm_instance = Mock()
        mock_vqa_instance = Mock()
        mock_dsg_instance = Mock()
        
        mock_llm.return_value = mock_llm_instance
        mock_vqa.return_value = mock_vqa_instance
        mock_dsg_gen.return_value = mock_dsg_instance
        
        # Mock DSG generation
        mock_dsg_instance.generate_dsg.return_value = (
            {"q1": "Is there a cat?", "q2": "Is the cat orange?"},
            {"q2": ["q1"]},  # q2 depends on q1
            {"q1": "Is there a cat in the image?", "q2": "Is the cat orange in color?"}
        )
        
        # Mock VQA answers: cat exists (yes) but not orange (no)
        mock_vqa_instance.answer.side_effect = ["yes", "no"]
        
        metric = DSGMetric()
        metric._initialize_models()
        
        # Create test image
        test_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        
        # Test evaluation
        score = metric._evaluate_image_dsg("A orange cat sitting on a chair", test_image)
        
        # q1="yes" (1.0), q2="no" (0.0), final score = (1.0 + 0.0) / 2 = 0.5
        assert score == 0.5
        assert mock_vqa_instance.answer.call_count == 2

    @patch('pruna.evaluation.metrics.metric_dsg.OpenAILLM')
    @patch('pruna.evaluation.metrics.metric_dsg.mPLUGVQA')
    @patch('pruna.evaluation.metrics.metric_dsg.DSGGenerator')
    def test_evaluate_image_dsg_dependency_constraints(self, mock_dsg_gen, mock_vqa, mock_llm):
        """Test dependency constraint application in DSG evaluation."""
        # Setup mocks
        mock_llm_instance = Mock()
        mock_vqa_instance = Mock()
        mock_dsg_instance = Mock()
        
        mock_llm.return_value = mock_llm_instance
        mock_vqa.return_value = mock_vqa_instance
        mock_dsg_gen.return_value = mock_dsg_instance
        
        # Mock DSG generation with dependency: q2 depends on q1
        mock_dsg_instance.generate_dsg.return_value = (
            {"q1": "Is there a cat?", "q2": "Is the cat orange?"},
            {"q2": ["q1"]},  # q2 depends on q1
            {"q1": "Is there a cat in the image?", "q2": "Is the cat orange in color?"}
        )
        
        # Mock VQA answers: no cat (no), but orange (yes) - dependency should zero out q2
        mock_vqa_instance.answer.side_effect = ["no", "yes"]
        
        metric = DSGMetric()
        metric._initialize_models()
        
        test_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        score = metric._evaluate_image_dsg("A orange cat", test_image)
        
        # q1="no" (0.0), q2 should be zeroed out due to dependency, final score = (0.0 + 0.0) / 2 = 0.0
        assert score == 0.0

    @patch('pruna.evaluation.metrics.metric_dsg.OpenAILLM')
    @patch('pruna.evaluation.metrics.metric_dsg.mPLUGVQA')
    @patch('pruna.evaluation.metrics.metric_dsg.DSGGenerator')
    def test_evaluate_image_dsg_no_questions(self, mock_dsg_gen, mock_vqa, mock_llm):
        """Test handling when no questions are generated."""
        # Setup mocks
        mock_llm_instance = Mock()
        mock_vqa_instance = Mock()
        mock_dsg_instance = Mock()
        
        mock_llm.return_value = mock_llm_instance
        mock_vqa.return_value = mock_vqa_instance
        mock_dsg_gen.return_value = mock_dsg_instance
        
        # Mock DSG generation with no questions
        mock_dsg_instance.generate_dsg.return_value = ({}, {}, {})
        
        metric = DSGMetric()
        metric._initialize_models()
        
        test_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        score = metric._evaluate_image_dsg("Invalid prompt", test_image)
        
        assert score == 0.0

    @patch('pruna.evaluation.metrics.metric_dsg.OpenAILLM')
    @patch('pruna.evaluation.metrics.metric_dsg.mPLUGVQA')
    @patch('pruna.evaluation.metrics.metric_dsg.DSGGenerator')
    def test_evaluate_image_dsg_vqa_failure(self, mock_dsg_gen, mock_vqa, mock_llm):
        """Test handling of VQA failures."""
        # Setup mocks
        mock_llm_instance = Mock()
        mock_vqa_instance = Mock()
        mock_dsg_instance = Mock()
        
        mock_llm.return_value = mock_llm_instance
        mock_vqa.return_value = mock_vqa_instance
        mock_dsg_gen.return_value = mock_dsg_instance
        
        # Mock DSG generation
        mock_dsg_instance.generate_dsg.return_value = (
            {"q1": "Is there a cat?"},
            {},
            {"q1": "Is there a cat in the image?"}
        )
        
        # Mock VQA to raise an exception
        mock_vqa_instance.answer.side_effect = Exception("VQA failed")
        
        metric = DSGMetric()
        metric._initialize_models()
        
        test_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        score = metric._evaluate_image_dsg("A cat", test_image)
        
        # Should handle exception gracefully and return 0.0
        assert score == 0.0

    @patch('pruna.evaluation.metrics.metric_dsg.OpenAILLM')
    @patch('pruna.evaluation.metrics.metric_dsg.mPLUGVQA')
    @patch('pruna.evaluation.metrics.metric_dsg.DSGGenerator')
    def test_update_and_compute(self, mock_dsg_gen, mock_vqa, mock_llm):
        """Test metric update and compute methods."""
        # Setup mocks
        mock_llm_instance = Mock()
        mock_vqa_instance = Mock()
        mock_dsg_instance = Mock()
        
        mock_llm.return_value = mock_llm_instance
        mock_vqa.return_value = mock_vqa_instance
        mock_dsg_gen.return_value = mock_dsg_instance
        
        # Mock DSG generation to return simple case
        mock_dsg_instance.generate_dsg.return_value = (
            {"q1": "Is there an object?"},
            {},  # no dependencies
            {"q1": "Is there an object in the image?"}
        )
        
        # Mock VQA to always return "yes"
        mock_vqa_instance.answer.return_value = "yes"
        
        metric = DSGMetric()
        
        # Create test data
        prompts = ["A cat sitting", "A dog playing"]
        images = [
            Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)),
            Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        ]
        
        # Update metric
        metric.update(prompts, None, images)
        
        # Check state
        assert metric.count == 2
        assert metric.total_score == 2.0  # Both should score 1.0
        
        # Compute result
        result = metric.compute()
        assert result.metric_name == METRIC_DSG
        assert result.result == 1.0  # Average score
        assert result.params["total_samples"] == 2

    @patch('pruna.evaluation.metrics.metric_dsg.OpenAILLM')
    @patch('pruna.evaluation.metrics.metric_dsg.mPLUGVQA')
    @patch('pruna.evaluation.metrics.metric_dsg.DSGGenerator')
    def test_update_single_samples(self, mock_dsg_gen, mock_vqa, mock_llm):
        """Test update with single prompt and image."""
        # Setup mocks
        mock_llm_instance = Mock()
        mock_vqa_instance = Mock()
        mock_dsg_instance = Mock()
        
        mock_llm.return_value = mock_llm_instance
        mock_vqa.return_value = mock_vqa_instance
        mock_dsg_gen.return_value = mock_dsg_instance
        
        # Mock DSG generation
        mock_dsg_instance.generate_dsg.return_value = (
            {"q1": "Is there a cat?"},
            {},
            {"q1": "Is there a cat in the image?"}
        )
        
        mock_vqa_instance.answer.return_value = "yes"
        
        metric = DSGMetric()
        
        # Test with single samples (not lists)
        prompt = "A cat"
        image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        
        metric.update(prompt, None, image)
        
        assert metric.count == 1
        assert metric.total_score == 1.0

    def test_compute_empty_state(self):
        """Test compute with no samples processed."""
        metric = DSGMetric()
        
        result = metric.compute()
        assert result.metric_name == METRIC_DSG
        assert result.result == 0.0
        assert result.params["total_samples"] == 0

    @patch('pruna.evaluation.metrics.metric_dsg.OpenAILLM')
    @patch('pruna.evaluation.metrics.metric_dsg.mPLUGVQA')
    @patch('pruna.evaluation.metrics.metric_dsg.DSGGenerator')
    def test_update_failure_handling(self, mock_dsg_gen, mock_vqa, mock_llm):
        """Test handling of update failures."""
        # Setup mocks
        mock_llm_instance = Mock()
        mock_vqa_instance = Mock()
        mock_dsg_instance = Mock()
        
        mock_llm.return_value = mock_llm_instance
        mock_vqa.return_value = mock_vqa_instance
        mock_dsg_gen.return_value = mock_dsg_instance
        
        # Mock DSG generation failure
        mock_dsg_instance.generate_dsg.side_effect = Exception("Complete failure")
        
        metric = DSGMetric()
        
        # Test update with failure
        prompts = ["A cat"]
        images = [Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))]
        
        metric.update(prompts, None, images)
        
        # Should count the failed sample as 0
        assert metric.count == 1
        assert metric.total_score == 0.0

    def test_call_type_handling(self):
        """Test different call type configurations."""
        # Test default call type
        metric = DSGMetric()
        assert metric.call_type == "x_y"
        
        # Test custom call type
        metric = DSGMetric(call_type="y_x")
        assert metric.call_type == "y_x"

    def test_answer_parsing(self):
        """Test VQA answer parsing for different response formats."""
        # Test through the evaluation pipeline
        with patch('pruna.evaluation.metrics.metric_dsg.OpenAILLM') as mock_llm:
            with patch('pruna.evaluation.metrics.metric_dsg.mPLUGVQA') as mock_vqa:
                with patch('pruna.evaluation.metrics.metric_dsg.DSGGenerator') as mock_dsg_gen:
                    # Setup mocks
                    mock_llm_instance = Mock()
                    mock_vqa_instance = Mock()
                    mock_dsg_instance = Mock()
                    
                    mock_llm.return_value = mock_llm_instance
                    mock_vqa.return_value = mock_vqa_instance
                    mock_dsg_gen.return_value = mock_dsg_instance
                    
                    # Mock DSG generation
                    mock_dsg_instance.generate_dsg.return_value = (
                        {"q1": "Test question?"},
                        {},
                        {"q1": "Test question?"}
                    )
                    
                    metric = DSGMetric()
                    metric._initialize_models()
                    test_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
                    
                    # Test different positive answers
                    for answer in ["yes", "YES", "true", "TRUE", "1"]:
                        mock_vqa_instance.answer.return_value = answer
                        score = metric._evaluate_image_dsg("Test", test_image)
                        assert score == 1.0, f"Answer '{answer}' should give score 1.0"
                    
                    # Test different negative answers
                    for answer in ["no", "NO", "false", "FALSE", "0", "maybe"]:
                        mock_vqa_instance.answer.return_value = answer
                        score = metric._evaluate_image_dsg("Test", test_image)
                        assert score == 0.0, f"Answer '{answer}' should give score 0.0"


@pytest.mark.integration
class TestDSGMetricIntegration:
    """Integration tests for DSG metric."""

    def test_metric_from_registry(self):
        """Test DSG metric can be retrieved from registry."""
        # Should be able to get DSG metric by name
        dsg_metric = MetricRegistry.get(METRIC_DSG)
        assert isinstance(dsg_metric, DSGMetric)
        assert dsg_metric.metric_name == METRIC_DSG
        assert dsg_metric.higher_is_better is True

    def test_metric_with_custom_params_from_registry(self):
        """Test DSG metric with custom parameters from registry."""
        # Test with custom parameters
        dsg_metric = MetricRegistry.get(
            METRIC_DSG, 
            vqa_model="instructblip", 
            max_questions=10,
            temperature=0.5
        )
        assert dsg_metric.vqa_model == "instructblip"
        assert dsg_metric.max_questions == 10
        assert dsg_metric.temperature == 0.5
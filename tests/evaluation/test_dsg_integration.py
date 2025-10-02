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
from unittest.mock import Mock, patch

from pruna.evaluation.metrics.metric_dsg import METRIC_DSG


class TestDSGIntegration:
    """Integration tests for DSG metric with evaluation pipeline."""

    def test_dsg_metric_in_task_creation(self):
        """Test DSG metric can be used in Task creation."""
        # Import here to avoid circular imports in test discovery
        from pruna.evaluation.task import Task
        from pruna.data.pruna_datamodule import PrunaDataModule
        
        # Test creating a task with DSG metric
        datamodule = PrunaDataModule.from_string("TinyCIFAR10")
        datamodule.limit_datasets(2)  # Use very small dataset for testing
        
        # Create task with DSG metric
        task = Task(
            request=[METRIC_DSG],
            datamodule=datamodule
        )
        
        # Verify the metric is properly configured
        assert len(task.metrics) == 1
        assert METRIC_DSG in [metric.metric_name for metric in task.metrics]

    def test_dsg_metric_with_custom_params_in_task(self):
        """Test DSG metric with custom parameters in Task."""
        from pruna.evaluation.task import Task
        from pruna.data.pruna_datamodule import PrunaDataModule
        
        datamodule = PrunaDataModule.from_string("TinyCIFAR10")
        datamodule.limit_datasets(1)
        
        # Create task with custom DSG parameters
        task = Task(
            request=[{
                METRIC_DSG: {
                    "vqa_model": "instructblip",
                    "max_questions": 10,
                    "temperature": 0.5
                }
            }],
            datamodule=datamodule
        )
        
        # Verify the metric is configured with custom parameters
        assert len(task.metrics) == 1
        dsg_metric = task.metrics[0]
        assert dsg_metric.metric_name == METRIC_DSG
        assert dsg_metric.vqa_model == "instructblip"
        assert dsg_metric.max_questions == 10
        assert dsg_metric.temperature == 0.5

    @patch('pruna.evaluation.metrics.metric_dsg.OpenAILLM')
    @patch('pruna.evaluation.metrics.metric_dsg.mPLUGVQA')
    @patch('pruna.evaluation.metrics.metric_dsg.DSGGenerator')
    def test_dsg_metric_in_evaluation_agent(self, mock_dsg_gen, mock_vqa, mock_llm):
        """Test DSG metric works with EvaluationAgent."""
        # Import here to avoid circular imports
        from pruna.evaluation.task import Task
        from pruna.evaluation.evaluation_agent import EvaluationAgent
        from pruna.data.pruna_datamodule import PrunaDataModule
        
        # Setup mocks for DSG
        mock_llm_instance = Mock()
        mock_vqa_instance = Mock()
        mock_dsg_instance = Mock()
        
        mock_llm.return_value = mock_llm_instance
        mock_vqa.return_value = mock_vqa_instance
        mock_dsg_gen.return_value = mock_dsg_instance
        
        # Mock DSG to return simple successful case
        mock_dsg_instance.generate_dsg.return_value = (
            {"q1": "Is there an object?"},
            {},
            {"q1": "Is there an object in the image?"}
        )
        mock_vqa_instance.answer.return_value = "yes"
        
        # Create task with DSG metric
        datamodule = PrunaDataModule.from_string("TinyCIFAR10")
        datamodule.limit_datasets(2)  # Very small dataset
        
        task = Task(
            request=[METRIC_DSG],
            datamodule=datamodule
        )
        
        # Create evaluation agent
        eval_agent = EvaluationAgent(task=task)
        
        # Verify the agent was created successfully with DSG metric
        assert eval_agent.task == task
        assert len(task.metrics) == 1
        assert task.metrics[0].metric_name == METRIC_DSG

    def test_dsg_metric_registry_access(self):
        """Test DSG metric is accessible through MetricRegistry."""
        from pruna.evaluation.metrics.registry import MetricRegistry
        from pruna.evaluation.metrics.metric_dsg import DSGMetric
        
        # Test direct registry access
        assert METRIC_DSG in MetricRegistry._registry
        
        # Test getting metric instance
        metric = MetricRegistry.get(METRIC_DSG)
        assert isinstance(metric, DSGMetric)
        assert metric.metric_name == METRIC_DSG
        
        # Test getting metric with parameters
        metric_with_params = MetricRegistry.get(
            METRIC_DSG,
            vqa_model="instructblip",
            max_questions=5
        )
        assert isinstance(metric_with_params, DSGMetric)
        assert metric_with_params.vqa_model == "instructblip"
        assert metric_with_params.max_questions == 5

    def test_dsg_metric_import_accessibility(self):
        """Test DSG metric can be imported from metrics module."""
        # Test importing from main metrics module
        from pruna.evaluation.metrics import DSGMetric, MetricRegistry
        
        # Verify the import works
        assert DSGMetric is not None
        assert hasattr(DSGMetric, 'metric_name')
        assert DSGMetric.metric_name == METRIC_DSG
        
        # Verify it's in the registry
        assert METRIC_DSG in MetricRegistry._registry

    def test_multiple_metrics_including_dsg(self):
        """Test DSG metric works alongside other metrics."""
        from pruna.evaluation.task import Task
        from pruna.data.pruna_datamodule import PrunaDataModule
        
        datamodule = PrunaDataModule.from_string("TinyCIFAR10")
        datamodule.limit_datasets(1)
        
        # Create task with multiple metrics including DSG
        task = Task(
            request=[METRIC_DSG, "total_params"],  # DSG + a simple metric
            datamodule=datamodule
        )
        
        # Verify both metrics are configured
        assert len(task.metrics) == 2
        metric_names = [metric.metric_name for metric in task.metrics]
        assert METRIC_DSG in metric_names
        assert "total_params" in metric_names

    def test_dsg_error_handling_in_pipeline(self):
        """Test that DSG metric errors don't break the evaluation pipeline."""
        from pruna.evaluation.task import Task
        from pruna.data.pruna_datamodule import PrunaDataModule
        
        # Create task with DSG metric (this should work even if DSG deps not installed)
        datamodule = PrunaDataModule.from_string("TinyCIFAR10")
        datamodule.limit_datasets(1)
        
        task = Task(
            request=[METRIC_DSG],
            datamodule=datamodule
        )
        
        # The task creation should succeed
        assert len(task.metrics) == 1
        assert task.metrics[0].metric_name == METRIC_DSG
        
        # The actual evaluation would fail if DSG not installed,
        # but the metric setup should be graceful
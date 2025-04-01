Adding a Metric
===============================

This guide will walk you through the process of adding a new metric to Pruna's evaluation system.

If anything is unclear or you want to discuss your contribution before opening a PR, please reach out on `Discord <https://discord.gg/Tun8YgzxZ9>`_ anytime!

If this is your first time contributing to |pruna|, please refer to the :ref:`how-to-contribute` guide for more information.

Understanding Pruna's Metric System
-----------------------------------

|pruna| has two main types of metrics that live under ``pruna/evaluation/metrics``:

1. **Base Metrics** - Inherit from ``BaseMetric`` and compute values directly without maintaining state. These metrics usually require isolated inference computation. Examples: ``GPUMemoryMetric``, ``ElapsedTimeMetric``. 
2. **Stateful Metrics** - Inherit from ``StatefulMetric`` and maintain internal state across multiple computations. State here refers to the information that is accumulated across multiple batches. Examples: all metrics under ``TorchMetricWrapper`` like ``Accuracy``, ``CLIPScore``. 

When adding a new metric to |pruna|, you should place your implementation in ``pruna/evaluation/metrics`` directory to ensure it's properly integrated with the rest of the system. Use snake_case for the file name (e.g., ``your_new_metric.py``).

In |pruna|, we evaluate metrics by sharing inference runs across multiple metrics whenever possible. This means that |pruna| runs inference once for all compatible metrics.
 
- **Stateful metrics** are preferred for most use cases, especially quality metrics, as they can share inference results across multiple metrics
- **Base metrics** are primarily used when isolated inference is required (e.g., for GPU memory metrics where sharing inference would distort results)

.. note::
   If you are confused about which type of metric to implement, you will likely need to implement stateful metrics. Base metrics are typically only used for specialized performance measurements that require isolated inference.

We use PascalCase for the class names (e.g, ``YourNewMetric``) and NumPy style docstrings for documentation. 

Base Metrics
~~~~~~~~~~~~

Base metrics inherit from the ``BaseMetric`` class and implement the ``compute()`` method. These are used when a metric requires isolated inference or cannot share computation with other metrics.

|pruna| ``EvaluationAgent`` (`documentation <../user_manual/evaluation.html#evaluationagent>`_) requires all ``BaseMetric`` s to implement the ``compute`` method with two specific parameters: ``model`` and ``dataloader``. Please take note that the ``EvaluationAgent`` does not handle inference for base metrics. You will need to handle inference computations yourself.



.. code-block:: python

    from pruna.evaluation.metrics.metric_base import BaseMetric

    class YourNewMetric(BaseMetric):
        def __init__(self):
            super().__init__()
            # Initialize any parameters your metric needs
            
        def compute(self, model, dataloader):
            '''Run inference on the model and compute the metric value.'''
       
            outputs = run_inference(model, dataloader)
            result = some_calculation(outputs)
            return result

Stateful Metrics
~~~~~~~~~~~~~~~~

Stateful metrics inherit from the ``StatefulMetric`` class and are the preferred approach for most metrics in |pruna|. They maintain internal state variables that accumulate information across multiple batches, allowing for efficient sharing of inference across different metrics.

Every stateful metric must implement the following methods:

1. ``__init__(self, **kwargs)``: Initialize your metric and its parameters
    - Call ``super().__init__()``
    - Set ``self.metric_name``
    - Set ``self.default_call_type``. We also recommend passing ``call_type`` to the ``__init__`` method to allow for pairwise evaluation.
    - Initialize state variables using ``add_state()``
    - Define any additional parameters

2. ``update(self, inputs, ground_truths, predictions)``: Process each batch
    - Called automatically by the evaluation pipeline
    - Update your state variables based on the current batch. Your implementation can use any combination of these parameters as needed for its specific calculations.
    - No return value needed

3. ``compute(self)``: Calculate final metric value
    - Use accumulated state to compute final result
    - Called after all batches are processed
    - Must return the final metric value


Here's a complete example showing all required methods:

.. code-block:: python

    from pruna.evaluation.metrics.metric_stateful import StatefulMetric
    from pruna.evaluation.metrics.utils import metric_data_processor
    import torch

    class YourNewStatefulMetric(StatefulMetric):
        def __init__(self, param1='default1', param2='default2', call_type=""):
            super().__init__()
            self.param1 = param1
            self.param2 = param2
            self.metric_name = "your_metric_name"
            self.default_call_type = "y_gt"
            self.call_type = call_type if call_type else self.default_call_type
            
            # Initialize state variables
            self.add_state("total", torch.zeros(1))
            self.add_state("count", torch.zeros(1))
        
        def update(self, inputs, ground_truths, predictions):
            # Update the state variables based on the current batch
            # Pass the inputs, ground_truths and predictions and the call_type to the metric_data_processor to get the data in the correct format
            metric_data = metric_data_processor(inputs, ground_truths, predictions, self.call_type)
            batch_result = some_calculation(*metric_data)
            self.total += batch_result
            self.count += 1
            
        def compute(self):
            # Compute the final metric value using the accumulated state
            if self.count == 0:
                return 0
            return self.total / self.count
            

When to Use Each Type
~~~~~~~~~~~~~~~~~~~~~

- **Use Stateful Metrics when**: Your metric can share inference with other metrics without affecting results (most quality metrics fall into this category)
- **Use Basic Metrics when**: Your metric requires isolated inference or would produce incorrect results if inference were shared (e.g., performance metrics like GPU memory usage)

By using stateful metrics whenever possible, |pruna| can efficiently evaluate multiple metrics with just a single inference pass. 

Registering Your Metric
-----------------------

After implementing your metric, you need to register it with Pruna's ``MetricRegistry`` system. 

The simplest way to do this is with the ``@MetricRegistry.register`` decorator:

.. code-block:: python

    from pruna.evaluation.metrics.registry import MetricRegistry
    from pruna.evaluation.metrics.metric_stateful import StatefulMetric

    @MetricRegistry.register("your_new_metric_name")
    class YourNewMetric(StatefulMetric):
        def __init__(self, param1='default1', param2='default2'): # Don't forget to add default values for your parameters!
            super().__init__()
            self.param1 = param1
            self.param2 = param2
            self.metric_name = "your_metric_name"
            
Thanks to this registry system, everyone using |pruna| can now refer to your metric by name without having to create instances directly!

One important thing: the registration happens when your module is imported. To ensure your metric is always available, we suggest importing it in ``pruna/evaluation/metrics/__init__.py`` file.

Steps to Add a New Metric
-------------------------

1. **Decide on the metric type**: Determine whether your metric needs isolated inference (use ``BaseMetric``) or can share inference (use ``StatefulMetric``).

2. **Create a new file**: Create a new Python file in the ``pruna/evaluation/metrics/`` directory with a descriptive name for your metric.

3. **Implement your metric class**: Inherit from the appropriate class and implement the required methods.

4.  **Register your metric**: Use the ``MetricRegistry.register`` decorator to make your metric available throughout the system.

5. **Add tests**: Create tests in ``pruna/tests/evaluation`` for your metric to ensure it works correctly.

6. **Update documentation**: Add documentation for your new metric in the user manual ``docs/user_manual/evaluation.rst``, including examples of how to use it.

7. **Submit a pull request**: Follow the standard contribution process to submit your new metric for review.

By following these steps, you'll help expand Pruna's capabilities and contribute to the project's success.


Using your new metric
---------------------

Once you've implemented your metric, everyone can use it in Pruna's evaluation pipeline! Here's how:

.. container:: hidden_code

    .. code-block:: python

        # mock certain imports to make the code block runnable
        import sys
        import types
        from diffusers import StableDiffusionPipeline

        dummy_your_metric = types.ModuleType("pruna.evaluation.metrics.your_metric_file")
        dummy_your_metric.YourNewMetric = "dummy_your_metric"
        sys.modules["pruna.evaluation.metrics.your_metric_file"] = dummy_your_metric

        model_path = "CompVis/stable-diffusion-v1-4"
        model = StableDiffusionPipeline.from_pretrained(model_path)

.. code-block:: python
    :emphasize-lines: 2, 6

    from pruna.evaluation.metrics.metric_torch import TorchMetricWrapper
    from pruna.evaluation.metrics.your_metric_file import YourNewMetric

    metrics = [
        'clip_score',
        'your_new_metric_name' 
    ]

    data_module = PrunaDataModule.from_string('LAION256')
    test_dataloader = data_module.train_dataloader()

    task = Task(request=metrics, dataloader=test_dataloader)

    eval_agent = EvaluationAgent(task=task)

    results = eval_agent.evaluate(model)

    



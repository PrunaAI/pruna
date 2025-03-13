Adding a Metric
===============================

This guide will walk you through the process of adding a new metric to Pruna's evaluation system.

If anything is unclear or you want to discuss your contribution before opening a PR, please reach out on `Discord <https://discord.gg/Tun8YgzxZ9>`_ anytime!

If this is your first time contributing to |pruna|, please refer to the :ref:`how-to-contribute` guide for more information.

Understanding Pruna's Metric System
-----------------------------------

|pruna| has two main types of metrics that live under ``pruna/evaluation/metrics``:

1. **Base Metrics** - Inherit from ``BaseMetric`` and compute values directly without maintaining state. These metrics usually require isolated inference computation. Examples: ``GPUMemoryMetric``, ``ElapsedTimeMetric``. 
2. **Stateful Metrics** - Inherit from ``StatefulMetric`` and maintain internal state across multiple computations. State here refers to the information that is accumulated across multiiple batches. Examples: all metrics under ``TorchMetricWrapper`` like ``Accuracy``, ``CLIPScore``. 

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
    - Initialize state variables using ``add_state()``
    - Define any additional parameters

2. ``add_state(self, name, default_value)``: Define persistent state variables
    - Must be called in ``__init__``
    - Creates variables that persist and accumulate across batches
    - Example states: totals, counts, running sums

3. ``update(self, inputs, ground_truths, predictions)``: Process each batch
    - Called automatically by the evaluation pipeline
    - Update your state variables based on the current batch. Your implementation can use any combination of these parameters as needed for its specific calculations.
    - No return value needed

4. ``compute(self)``: Calculate final metric value
    - Use accumulated state to compute final result
    - Called after all batches are processed
    - Must return the final metric value

5. ``reset(self)``: Reset all state variables
    - Must reset all states to their initial values
    - Called automatically between evaluation runs

Here's a complete example showing all required methods:

.. code-block:: python

    from pruna.evaluation.metrics.metric_stateful import StatefulMetric

    class YourNewStatefulMetric(StatefulMetric):
        def __init__(self, param1='default1', param2='default2'):
            super().__init__()
            self.param1 = param1
            self.param2 = param2
            self.metric_name = "your_metric_name"
            
            # Initialize state variables
            self.add_state("total", 0)
            self.add_state("count", 0)
        
        def add_state(self, name, default_value):
            '''Add a state variable to the metric.'''
            self.state[name] = default_value
            
        def update(self, inputs, ground_truths, predictions):
            # Update the state variables based on the current batch
            # Choose the required combination of inputs, ground_truths and predictions
            batch_result = some_calculation(predictions, ground_truths)
            self.total += batch_result
            self.count += 1
            
        def compute(self):
            # Compute the final metric value using the accumulated state
            if self.count == 0:
                return 0
            return self.total / self.count
            
        def reset(self):
            # Reset state variables to initial values
            self.total = 0
            self.count = 0


When to Use Each Type
~~~~~~~~~~~~~~~~~~~~~

- **Use Stateful Metrics when**: Your metric can share inference with other metrics without affecting results (most quality metrics fall into this category)
- **Use Basic Metrics when**: Your metric requires isolated inference or would produce incorrect results if inference were shared (e.g., performance metrics like GPU memory usage)

By using stateful metrics whenever possible, |pruna| can efficiently evaluate multiple metrics with just a single inference pass. 

Steps to Add a New Metric
-------------------------

1. **Decide on the metric type**: Determine whether your metric needs isolated inference (use ``BaseMetric``) or can share inference (use ``StatefulMetric``).

2. **Create a new file**: Create a new Python file in the ``pruna/evaluation/metrics/`` directory with a descriptive name for your metric.

3. **Implement your metric class**: Inherit from the appropriate class and implement the required methods.

4. **Add tests**: Create tests in ``pruna/tests/evaluation`` for your metric to ensure it works correctly.

5. **Update documentation**: Add documentation for your new metric in the user manual ``docs/user_manual/evaluation.rst``, including examples of how to use it.

6. **Submit a pull request**: Follow the standard contribution process to submit your new metric for review.

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
        TorchMetricWrapper('clip_score'),
        YourNewMetric(param1='custom_value') 
    ]

    data_module = PrunaDataModule.from_string('LAION256')
    test_dataloader = data_module.train_dataloader()

    task = Task(request=metrics, dataloader=test_dataloader)

    eval_agent = EvaluationAgent(task=task)

    results = eval_agent.evaluate(model)

    



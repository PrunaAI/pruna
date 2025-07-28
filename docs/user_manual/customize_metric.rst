Customize a Metric
===============================

This guide will walk you through the process of adding a new metric to Pruna's evaluation system.

If anything is unclear or you want to discuss your contribution before opening a PR, please reach out on `Discord <https://discord.gg/JFQmtFKCjd>`_ anytime!

.. tip::

   If you want to contribute to |pruna|, please refer to the :doc:`/docs_pruna/contributions/how_to_contribute` guide for more information. You can also `find finished PRs <https://github.com/PrunaAI/pruna/pulls?q=is%3Apr+label%3Ametric+>`_ in the GitHub repository.

1. Choosing the right type of metric
------------------------------------

|pruna|'s evaluation system supports two types of metrics, located under ``pruna/evaluation/metrics``: ``BaseMetric`` and ``StatefulMetric``.

These two types are designed to accommodate different use cases.

- **BaseMetric**: Inherit from ``BaseMetric`` and compute values directly without maintaining state.
    - Used when isolated inference is required (e.g., ``latency``, ``disk_memory``, etc.)
- **StatefulMetric**: Inherit from ``StatefulMetric`` and accumulate state across multiple batches.
    - Best suited for quality evaluations (e.g, ``accuracy``, ``clip_score``, etc.)

.. note::
    In most cases, you should implement a ``StatefulMetric``. ``BaseMetric`` is reserved for specialized performance measurements where shared inference would distort results.

2. Implement the metric class
-----------------------------

Create a new file in ``pruna/evaluation/metrics`` with a descriptive name for your metric. (e.g, ``your_new_metric.py``)

We use snake_case for the file names (e.g., ``your_new_metric.py``), PascalCase for the class names (e.g, ``YourNewMetric``) and NumPy style docstrings for documentation.

Both  ``BaseMetric`` and ``StatefulMetric`` return a ``MetricResult`` object, which contains the metric name, result value and other metadata.

Implementing a ``BaseMetric``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a new class that inherits from ``BaseMetric`` and implements the ``compute()`` method.

Your metric should have a ``metric_name`` and a ``higher_is_better`` attribute. Higher is better is a boolean value that indicates if a higher metric value is better.

Every metric has a ``runs_on`` attribute that should be a list that contains the device(s) the metric can run on. For base metrics it is by default ``["cuda", "cpu", "mps"]``.

Please update the ``runs_on`` if you want to change the default devices. For instance, you can also include ``accelerate`` if your metric logic works on multiple devices. Inference with ``accelerate`` should work out of the box, and some of |pruna|'s metrics already use it!

``compute()`` takes two parameters: ``model`` and ``dataloader``.

Inside ``compute()``, you are responsible for running inference manually.

Your method should return a ``MetricResult`` object with the metric name, result value and other metadata. The result value should be a float or int.

.. code-block:: python

    from pruna.evaluation.metrics.metric_base import BaseMetric
    from pruna.evaluation.metrics.result import MetricResult

    class YourNewMetric(BaseMetric):
        '''Your metric description'''

        metric_name = "your_metric_name"
        higher_is_better = True # or False
        runs_on = ["cuda", "cpu"] # This is optional. Only change if your metric runs on different devices than the default ones.

        def __init__(self):
            super().__init__()
            # Initialize any parameters your metric needs

        def compute(self, model, dataloader):
            '''Run inference on the model and compute the metric value.'''

            outputs = run_inference(model, dataloader)
            result = some_calculation(outputs)
            params = self.__dict__.copy() # or any metadata you prefer
            return MetricResult(self.metric_name, params, result)

Implementing a ``StatefulMetric``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To implement a ``StatefulMetric``, create a class that inherits from ``StatefulMetric``. These metrics are designed to accumulate state across multiple batches and can share inference with other metrics.

Your metric should have a ``metric_name`` and a ``higher_is_better`` attribute. Higher is better is a boolean value that indicates if a higher metric value is better.

Every metric has a ``runs_on`` attribute that should be a list that contains the device(s) the metric can run on. For base metrics it is by default ``["cuda", "cpu", "mps"]``.

Please update the ``runs_on`` if you want to change the default devices.

Use ``add_state()`` method to define internal state variables that will accumulate data across batches. For example, you might track totals and counts to compute an average.

The ``update()`` method processes each batch of data, updating the state variables based on the current batch. It takes three parameters: ``inputs``, ``ground_truths`` and ``predictions``.

The ``compute()`` method is called after all batches are processed and returns a ``MetricResult`` object, which contains the final metric value calculated from the accumulated state.

The ``move_to_device()`` method is used to move the metric and necessary attributes and variables of the metric to the specified device.

Metrics can operate in both single-model and pairwise modes, determined by the ``call_type`` parameter. Common ``call_types`` include ``y_gt``, ``gt_y``, ``x_gt``, ``gt_x``, ``pairwise_y_gt``, and ``pairwise_gt_y``. For more details, see the :ref:`Understanding Call Types <understanding-call-types>` section.

Once you have implemented your metric, you can switch the mode of the metric despite your default ``call_type`` simply by passing ``single`` or ``pairwise`` to the ``call_type`` parameter of the ``StatefulMetric`` constructor.

Here's a complete example implementing a ``StatefulMetric`` with a single ``call_type`` showing all required methods:

.. code-block:: python

    from pruna.evaluation.metrics.metric_stateful import StatefulMetric
    from pruna.evaluation.metrics.result import MetricResult
    from pruna.evaluation.metrics.utils import SINGLE, get_call_type_for_single_metric, metric_data_processor # for pairwise metrics, you would need to change the imports to pairwise
    import torch

    class YourNewStatefulMetric(StatefulMetric):
        '''Your metric description'''

        default_call_type = "y_gt"
        metric_name = "your_metric_name"
        higher_is_better = True # or False
        runs_on = ["cuda", "cpu"]
        def __init__(self, param1='default1', param2='default2', call_type=SINGLE): # Since we picked a single call_type for default, we can use it as a default value
            super().__init__()
            self.param1 = param1
            self.param2 = param2
            self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type) # Call the correct helper function to get the correct call_type


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
            return MetricResult(self.metric_name, self.__dict__.copy(), self.total / self.count)

        def move_to_device(self, device: str | torch.device):
            # Move the metric and necessary attributes and variables of the metric to the specified device
            self.device = device
            self.total = self.total.to(device)
            self.count = self.count.to(device)

.. _understanding-call-types:

Understanding Call Types
~~~~~~~~~~~~~~~~~~~~~~~~~

|pruna| metrics can operate in both single-model and pairwise modes:

 - **Single-model mode**: Each evaluation produces independent scores for the model being evaluated.
 - **Pairwise mode**: Metrics compare a subsequent model against the first model evaluated by the agent and produce a single comparison score.

+--------------------+-------------------------------------------------------------+
| Call Type          | Description                                                 |
+====================+=============================================================+
| `y_gt`             | Model's output first, then ground truth                     |
+--------------------+-------------------------------------------------------------+
| `gt_y`             | Ground truth first, then model's output                     |
+--------------------+-------------------------------------------------------------+
| `x_gt`             | Input data first, then ground truth                         |
+--------------------+-------------------------------------------------------------+
| `gt_x`             | Ground truth first, then input data                         |
+--------------------+-------------------------------------------------------------+
| `pairwise_y_gt`    | Base model's output first, then subsequent model's output   |
+--------------------+-------------------------------------------------------------+
| `pairwise_gt_y`    | Subsequent model's output first, then base model's output   |
+--------------------+-------------------------------------------------------------+ 
| `y`                | Only the output is used, the metric has an internal dataset |
+--------------------+-------------------------------------------------------------+


You need to decide on the default ``call_type`` based on the metric you are implementing.

For example, if you are implementing a metric that compares two models, you should use the ``pairwise_y_gt`` call type. Examples from |pruna| include ``psnr``, ``ssim``, ``lpips``.

If you are implementing an alignment metric comparing model's output with the input, you should use the ``x_gt`` or ``gt_x`` call type. Examples from |pruna| include ``clip_score``.

If you are implementing a metric that compares the model's output with the ground truth, you should use the ``y_gt`` or ``gt_y`` call type. Examples from |pruna| include ``fid``, ``cmmd``, ``accuracy``, ``recall``, ``precision``.

If you are wrapping an Image Quality Assessment (IQA) metric, that has an internal dataset, you should use the ``y`` call type. Examples from |pruna| include ``arniqa``.

You may want to switch the mode of the metric despite your default ``call_type``. For instance you may want to use ``fid`` in pairwise mode to get a single comparison score for two models.

In this case, you can pass ``pairwise`` to the ``call_type`` parameter of the ``StatefulMetric`` constructor.

.. container:: hidden_code

    .. code-block:: python

        import sys, types

        mod_name = "pruna.evaluation.metrics.your_metric_file"
        dummy = types.ModuleType(mod_name)

        class YourNewStatefulMetric:
            def __init__(self, *args, **kwargs): pass
            def reset(self):  ...
            def update(self, *a, **k): ...
            def compute(self): return 0.0

        dummy.YourNewStatefulMetric = YourNewStatefulMetric
        sys.modules[mod_name] = dummy

.. code-block:: python

    from pruna.evaluation.metrics.your_metric_file import YourNewStatefulMetric


    # Initialize your metric from the instance
    YourNewStatefulMetric(param1='value1', param2='value2', call_type="pairwise")

If you have implemented your metric using the correct ``get_call_type_for_metric`` function and ``metric_data_processor`` function, this will work as expected.


3. Register the metric
----------------------

After implementing your metric, you need to register it with Pruna's ``MetricRegistry`` system.

The simplest way to do this is with the ``@MetricRegistry.register`` decorator:

.. code-block:: python

    from pruna.evaluation.metrics.registry import MetricRegistry
    from pruna.evaluation.metrics.metric_stateful import StatefulMetric

    @MetricRegistry.register("your_metric_name")
    class YourNewMetric(StatefulMetric):
        def __init__(self, param1='default1', param2='default2'): # Don't forget to add default values for your parameters!
            super().__init__()
            self.param1 = param1
            self.param2 = param2
            self.metric_name = "your_metric_name"

Thanks to this registry system, everyone using |pruna| can now refer to your metric by name without having to create instances directly!

.. container:: hidden_code

    .. code-block:: python

        # mock certain imports to make the code block runnable

        import sys, types
        from pruna.evaluation.metrics.registry import MetricRegistry

        mod_name = "pruna.evaluation.metrics.your_metric_file"
        dummy = types.ModuleType(mod_name)

        @MetricRegistry.register("your_new_metric_name")
        class YourNewMetric:
            def __init__(self, *args, **kwargs): pass
            def compute(self): return 0.0

        dummy.YourNewMetric = YourNewMetric
        sys.modules[mod_name] = dummy

.. code-block:: python

    from pruna.evaluation.metrics.your_metric_file import YourNewMetric

    # Classic way: Initialize your metric from the instance
    YourNewMetric(param1='value1', param2='value2')

.. code-block:: python

    from pruna.evaluation.task import Task
    from pruna.data.pruna_datamodule import PrunaDataModule

    metrics = [
        'your_new_metric_name'
    ]

    # Now you can create a task with your metric from the metric name.
    task = Task(request=metrics, datamodule=PrunaDataModule.from_string('LAION256'))


One important thing: the registration happens when your module is imported. To ensure your metric is always available, we suggest importing it in ``pruna/evaluation/metrics/__init__.py`` file.

4. Add tests and update the documentation
-----------------------------------------

Create tests in ``pruna/tests/evaluation`` for your metric to ensure it works correctly.

Add documentation for your new metric in the user manual ``docs/user_manual/evaluation.rst``, including examples of how to use it.


By following these steps, you'll help expand Pruna's capabilities and contribute to the project's success.


Using your new metric
---------------------

Once you've implemented your metric, everyone can use it in Pruna's evaluation pipeline! Here's how:

.. container:: hidden_code

    .. code-block:: python

        # mock certain imports to make the code block runnable
        import sys, types

        modname = "pruna.evaluation.metrics.your_metric_file"
        dummy = types.ModuleType(modname)

        class YourNewMetric:
            def __init__(self, *a, **k): ...
            def compute(self): return 0.0

        dummy.YourNewMetric = YourNewMetric
        sys.modules[modname] = dummy

.. code-block:: python
    :emphasize-lines: 2, 6

    from pruna.evaluation.metrics.metric_torch import TorchMetricWrapper
    from pruna.evaluation.metrics.your_metric_file import YourNewMetric


    metrics = [
        'clip_score',
        'your_new_metric_name'
    ]

    task = Task(request=metrics, data_module=pruna.data.pruna_datamodule.PrunaDataModule.from_string('LAION256'))

    eval_agent = EvaluationAgent(task=task)

    results = eval_agent.evaluate(model)

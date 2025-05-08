Evaluate quality with the Evaluation Agent
================================================

This guide provides an introduction to evaluating models with |pruna|.

Evaluation helps you understand how compression affects your models across different dimensions - from output quality to resource requirements.
This knowledge is essential for making informed decisions about which compression techniques work best for your specific needs.

Haven't smashed a model yet? Check out the :doc:`optimize guide </docs_pruna/user_manual/optimize>` to learn how to do that.

Basic Evaluation Workflow
-------------------------

|pruna| follows a simple workflow for evaluating model optimizations:

.. mermaid::
   :align: center

   graph LR
    User -->|creates| Task
    User -->|creates| EvaluationAgent
    Task -->|defines| PrunaDataModule
    Task -->|defines| Metrics
    Task -->|uses| EvaluationAgent
    Metrics -->|includes| StatefulMetric
    Metrics -->|includes| StatelessMetric
    PrunaModel -->|provides predictions| EvaluationAgent
    EvaluationAgent -->|evaluates| PrunaModel
    EvaluationAgent -->|returns| C["Evaluation Results"]
    User -->|configures| EvaluationAgent

    subgraph Metric_Types
    StatefulMetric
    StatelessMetric
    end

    subgraph Task_Definition
    Task
    PrunaDataModule
    Metrics
    Metric_Types
    end

    style User fill:#bbf,stroke:#333,stroke-width:2px
    style Task fill:#bbf,stroke:#333,stroke-width:2px
    style EvaluationAgent fill:#bbf,stroke:#333,stroke-width:2px
    style PrunaDataModule fill:#bbf,stroke:#333,stroke-width:2px
    style PrunaModel fill:#bbf,stroke:#333,stroke-width:2px
    style C fill:#bbf,stroke:#333,stroke-width:2px
    style Metrics fill:#bbf,stroke:#333,stroke-width:2px

Let's see what that looks like in code.

.. code-block:: python

    from pruna.evaluation.evaluation_agent import EvaluationAgent
    from pruna.evaluation.task import Task
    from pruna.data.pruna_datamodule import PrunaDataModule

    # Load the optimized model
    optimized_model = PrunaModel.from_pretrained("PrunaAI/opt-125m-smashed")

    # Create and configure Task
    task = Task(
        requests=["accuracy"],
        datamodule=PrunaDataModule.from_string('WikiText'),
        device="cpu"
    )

    # Create and configure EvaluationAgent
    eval_agent = EvaluationAgent(task)

    # Evaluate the model
    eval_agent.evaluate(optimized_model)

Evaluation Components
---------------------

The |pruna| package provides a variety of evaluation metrics to assess your models.
In this section, we’ll introduce the evaluation metrics you can use.

Task
^^^^

The ``Task`` is a class that defines the task you want to evaluate your model on and it requires a set of :doc:`Metrics </reference/evaluation>` and a :doc:`PrunaDataModule </reference/pruna_model>` to perform the evaluation.

Metrics
~~~~~~~

Metrics are the core components that calculate specific performance indicators. There are two main types of metrics:

- **Base Metrics**: These metrics compute values directly from inputs without maintaining state across batches.
- **Stateful Metrics**: Metrics that maintain internal state and accumulate information across multiple batches. These are typically used for quality assessment.

The ``Task`` accepts ``Metrics`` in three ways:

.. tabs::

    .. tab:: Predefined Options

        As a plain text request from predefined options (e.g., ``image_generation_quality``)

        .. code-block:: python

            from pruna.evaluation.task import Task
            from pruna.data.pruna_datamodule import PrunaDataModule

            task = Task(
                request="image_generation_quality",
                datamodule=PrunaDataModule.from_string('LAION256'),
                device="cpu"
            )

    .. tab:: List of Metric Names

        As a list of metric names (e.g., [``"clip_score"``, ``"psnr"``])

        .. code-block:: python

            from pruna.evaluation.task import Task
            from pruna.data.pruna_datamodule import PrunaDataModule

            task = Task(
                metrics=["clip_score", "psnr"],
                datamodule=PrunaDataModule.from_string('LAION256'),
                device="cpu"
            )

    .. tab:: List of Metric Instances

        As a list of metric instances (e.g., ``CMMD()``), which provides more flexibility in configuring the metrics.

        .. code-block:: python

            from pruna.evaluation.task import Task
            from pruna.data.pruna_datamodule import PrunaDataModule
            from pruna.evaluation.metrics import CMMD, TorchMetricWrapper

            task = Task(
                metrics=[CMMD(call_type="pairwise"), TorchMetricWrapper(metric_name="accuracy")],
                datamodule=PrunaDataModule.from_string('LAION256'),
                device="cpu"
            )

.. note::

    You can find the full list of available metrics in the :ref:`Metric Overview <metrics>` section.

Metric Call Types
^^^^^^^^^^^^^^^^

|pruna| metrics can operate in both single-model and pairwise modes.

- **Single-Model mode**: Each evaluation produces independent scores for the model being evaluated.
- **Pairwise mode**: Metrics compare a subsequent model against the first model evaluated by the agent and produce a single comparison score.

Underneath the hood, the ``StatefulMetric`` class uses the ``call_type`` parameter to determine the order of the inputs.

The following table shows the different call types supported by |pruna| metrics and the metrics that support each call type.

.. list-table::
   :widths: 10 60 10
   :header-rows: 1

   * - Call Type
     - Description
     - Example Metrics

   * - ``y_gt``
     - Model's output first, then ground truth
     - ``fid``, ``cmmd``, ``accuracy``, ``recall``, ``precision``

   * - ``gt_y``
     - Ground truth first, then model's output
     - ``fid``, ``cmmd``, ``accuracy``, ``recall``, ``precision``

   * - ``x_gt``
     - Input data first, then ground truth
     - ``clip_score``

   * - ``gt_x``
     - Ground truth first, then input data
     - ``clip_score``

   * - ``pairwise_y_gt``
     - Base model's output first, then subsequent model's output
     -  ``psnr``, ``ssim``, ``lpips``, ``cmmd``

   * - ``pairwise_gt_y``
     - Subsequent model's output first, then base model's output
     - ``psnr``, ``ssim``, ``lpips``, ``cmmd``

Each metric has a default ``call_type`` but you can switch the mode of the metric despite your default ``call_type``.

.. tabs::

    .. tab:: Single-Model mode

        .. code-block:: python

            from pruna.evaluation.metrics import CMMD

            metric = CMMD() # or ["cmmd"]

    .. tab:: Pairwise mode

        .. code-block:: python

            from pruna.evaluation.metrics import CMMD
            metric = CMMD(call_type="pairwise")

PrunaDataModule
~~~~~~~~~~~~~~~

The ``PrunaDataModule`` is a class that defines the data you want to evaluate your model on.
Data modules are a core component of the evaluation framework, providing standardized access to datasets for evaluating model performance before and after optimization.

A more detailed overview of the ``PrunaDataModule``, its datasets and their corresponding collate functions can be found in the :doc:`Data Module Overview </docs_pruna/user_manual/configure>` section.

The ``Task`` accepts ``PrunaDataModule`` in two different ways:

.. tabs::

    .. tab:: From String

        As a plain text request from predefined options (e.g., ``WikiText``)

        .. code-block:: python

            from pruna.data.pruna_datamodule import PrunaDataModule
            from transformers import AutoTokenizer

            # Load the tokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")

            # Create the data Module
            datamodule = PrunaDataModule.from_string(
                dataset_name='WikiText',
                tokenizer=tokenizer,
                collate_fn="text_generation_collate",
                collate_fn_args={"max_seq_len": 512},
                dataloader_args={"batch_size": 16, "num_workers": 4}
            )

    .. tab:: From Datasets

        As a list of datasets, which provides more flexibility in configuring the data module.

        .. code-block:: python

            from pruna.data.pruna_datamodule import prunadatamodule
            from transformers import AutoTokenizer
            from datasets import load_dataset

            # Load the tokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")

            # Load custom datasets
            train_ds = load_dataset("SamuelYang/bookcorpus")["train"]
            train_ds, val_ds, test_ds = split_train_into_train_val_test(train_ds, seed=42)

            # Create the data module
            datamodule = PrunaDataModule.from_datasets(
                datasets=(train_ds, val_ds, test_ds),
                collate_fn="text_generation_collate",
                tokenizer=tokenizer,
                collate_fn_args={"max_seq_len": 512},
                dataloader_args={"batch_size": 16, "num_workers": 4}
            )

.. tip::

    You can find the full list of available datasets in the :doc:`Dataset Overview </docs_pruna/user_manual/configure>` section.

Lastly, you can limit the number of samples in the dataset by using the ``PrunaDataModule.limit_samples`` method.

.. code-block:: python

    from pruna.data.pruna_datamodule import PrunaDataModule

    # Create the data module
    datamodule = PrunaDataModule.from_string('WikiText')

    # Limit all splits to 100 samples
    datamodule.limit_datasets(100)

    # Use different limits for each split
    datamodule.limit_datasets([500, 100, 200])  # train, val, test

EvaluationAgent
^^^^^^^^^^^^^^^

The ``EvaluationAgent`` is a class that evaluates the performance of your model.

To evaluate a model with the ``EvaluationAgent``, you need to create a ``Task`` with ``Metrics`` and a ``PrunaDataModule``.
Then, initialize an ``EvaluationAgent`` with that task and call the ``evaluate()`` method with your model.

We can then chose to evaluate a single model or a pair of models.

- **Single-Model mode**: each model is evaluated independently, producing metrics that only pertain to that model's performance. The metrics are computed from the model's outputs without reference to any other model.
- **Pairwise mode**: metrics compare the outputs of the current model against the first model evaluated by the agent. The first model's outputs are cached by the EvaluationAgent and used as a reference for subsequent evaluations.

Let's see how this works in code.

.. tabs::

    .. tab:: Single-Model Evaluation

        .. code-block:: python

            import copy

            from diffusers import StableDiffusionPipeline

            from pruna import smash, SmashConfig
            from pruna.data.pruna_datamodule import PrunaDataModule
            from pruna.evaluation.evaluation_agent import EvaluationAgent
            from pruna.evaluation.task import Task
            from pruna.evaluation.metrics import CMMD
            # Load data and set up smash config
            smash_config = SmashConfig()
            smash_config['cacher'] = 'deepcache'

            # Load the base model
            model_path = "CompVis/stable-diffusion-v1-4"
            pipe = StableDiffusionPipeline.from_pretrained(model_path)

            # Smash the model
            copy_pipe = copy.deepcopy(pipe)
            smashed_pipe = smash(copy_pipe, smash_config)

            # Define the task and the evaluation agent
            metrics = [CMMD()]
            task = Task(metrics, datamodule=PrunaDataModule.from_string('LAION256'))
            eval_agent = EvaluationAgent(task)

            # Evaluate base model, all models need to be wrapped in a PrunaModel before passing them to the EvaluationAgent
            first_results = eval_agent.evaluate(pipe)
            print(first_results)

    .. tab:: Pairwise Evaluation

        .. code-block:: python

            import copy

            from diffusers import StableDiffusionPipeline

            from pruna import smash, SmashConfig
            from pruna.data.pruna_datamodule import PrunaDataModule
            from pruna.evaluation.evaluation_agent import EvaluationAgent
            from pruna.evaluation.task import Task
            from pruna.evaluation.metrics import CMMD
            # Load data and set up smash config
            smash_config = SmashConfig()
            smash_config['cacher'] = 'deepcache'

            # Load the base model
            model_path = "CompVis/stable-diffusion-v1-4"
            pipe = StableDiffusionPipeline.from_pretrained(model_path)

            # Smash the model
            copy_pipe = copy.deepcopy(pipe)
            smashed_pipe = smash(copy_pipe, smash_config)

            # Define the task and the evaluation agent
            metrics = [CMMD(call_type="pairwise")]
            task = Task(metrics, datamodule=PrunaDataModule.from_string('LAION256'))
            eval_agent = EvaluationAgent(task)

            # Evaluate base model, all models need to be wrapped in a PrunaModel before passing them to the EvaluationAgent
            first_results = eval_agent.evaluate(pipe)
            print(first_results)

            # Evaluate smashed model
            smashed_results = eval_agent.evaluate(smashed_pipe)
            print(smashed_results)

Best Practices
--------------

Start with a small dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^

When first setting up evaluation, limit the dataset size with ``datamodule.limit_datasets(n)`` to make debugging faster.

Use pairwise metrics for comparison
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When comparing an optimized model against the baseline, use pairwise metrics to get direct comparison scores.
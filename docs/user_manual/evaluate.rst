Evaluate optimizations with the Evaluation Agent
================================================

This guide provides an introduction to evaluating model optimizations with |pruna|.

Evaluation helps you understand how compression affects your models across different dimensions - from output quality to resource requirements.
This knowledge is essential for making informed decisions about which compression techniques work best for your specific needs.

Basic Evaluation Workflow
-------------------------

|pruna| follows a simple workflow for evaluating model optimizations:

.. mermaid::
   :align: center

   graph LR
    User -->|creates| Task
    User -->|creates| EvaluationAgent
    Task -->|uses| PrunaDataModule
    Task -->|defines| Metrics
    Metrics -->|includes| StatefulMetric
    Metrics -->|includes| StatelessMetric
    PrunaDataModule -->|provides data| EvaluationAgent
    PrunaModel -->|provides predictions| EvaluationAgent
    EvaluationAgent -->|evaluates| PrunaModel
    EvaluationAgent -->|returns| Evaluation_Results
    User -->|configures| EvaluationAgent

    subgraph Metric_Types
        StatefulMetric
        StatelessMetric
    end

    style User fill:#bbf,stroke:#333,stroke-width:2px
    style Task fill:#bbf,stroke:#333,stroke-width:2px
    style EvaluationAgent fill:#bbf,stroke:#333,stroke-width:2px
    style PrunaDataModule fill:#bbf,stroke:#333,stroke-width:2px
    style PrunaModel fill:#bbf,stroke:#333,stroke-width:2px
    style Evaluation_Results fill:#bbf,stroke:#333,stroke-width:2px
    style Metrics fill:#bbf,stroke:#333,stroke-width:2px

Let's see what that looks like in code.

.. code-block:: python

    from pruna.evaluation.evaluation_agent import EvaluationAgent
    from pruna.evaluation.task import Task
    from pruna.data.pruna_datamodule import PrunaDataModule

    # Load the optimized model
    optimized_model = PrunaModel.from_pretrained("CompVis/stable-diffusion-v1-4")

    # Create and configure Task
    task = Task(
        requests=["clip_score", "psnr"],
        datamodule=PrunaDataModule.from_string('LAION256'),
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

The ``Task`` is a class that defines the task you want to evaluate your model on and it requires a set of metrics and a :ref:`PrunaDataModule <prunadatamodule>` to perform the evaluation.

.. code-block:: python

    from pruna.evaluation.task import Task
    from pruna.data.pruna_datamodule import PrunaDataModule

    task = Task(
        requests=["image_generation_quality"],
        datamodule=PrunaDataModule.from_string('LAION256'),
        device="cpu"
    )

Metrics
~~~~~~~

The ``Metrics`` is a class that defines the metrics you want to evaluate your model on.

Metrics are the core components that calculate specific performance indicators. There are two main types of metrics:

- **Stateful Metrics**: These metrics compute values directly from inputs without maintaining state across batches.
- **Stateless Metrics**: Metrics that maintain internal state and accumulate information across multiple batches. These are typically used for quality assessment.

The ``Task`` accepts ``Metrics`` in three ways:

.. tabs::

    .. tab:: Predefined Options

        As a plain text request from predefined options (e.g., ``image_generation_quality``)

        .. code-block:: python

            from pruna.evaluation.task import Task
            from pruna.data.pruna_datamodule import PrunaDataModule

            # Create the task
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

            # Create the task
            task = Task(
                metrics=["clip_score", "psnr"],
                datamodule=PrunaDataModule.from_string('LAION256'),
                device="cpu"
            )

    .. tab:: List of Metric Instances

        As a list of metric instances, which provides more flexibility in configuring the metrics.

        .. code-block:: python

            from pruna.evaluation.task import Task
            from pruna.data.pruna_datamodule import PrunaDataModule
            from pruna.evaluation.metrics.metric_psnr import PSNR

            # Initialize the metrics
            metrics = [
                PSNR()
            ]

            # Create the task
            task = Task(
                metrics=metrics,
                datamodule=PrunaDataModule.from_string('LAION256'),
                device="cpu"
            )

.. note::

    You can find the full list of available metrics in the :ref:`Metric Overview <metrics>` section.

PrunaDataModule
~~~~~~~~~~~~~~~

The ``PrunaDataModule`` is a class that defines the data you want to evaluate your model on.
Data modules are a core component of the evaluation framework, providing standardized access to datasets for evaluating model performance before and after optimization.

They offer the following functionality:

- Standard dataloaders for training, validation, and testing
- Integration with appropriate collate functions for different data types
- Support for dataset size limitations for faster evaluation
- Compatibility with tokenizers for text-based tasks

The ``Task`` accepts ``PrunaDataModule`` in three ways:

.. tabs::

    .. tab:: From String

        As a plain text request from predefined options (e.g., ``LAION256``)

        .. code-block:: python

            from pruna.data.pruna_datamodule import PrunaDataModule

            # Create the data Module
            datamodule = PrunaDataModule.from_string('LAION256')

    .. tab:: From Datasets

        As a list of datasets, which provides more flexibility in configuring the data module.

        .. code-block:: python

            from pruna.data.pruna_datamodule import prunadatamodule
            from transformers import AutoTokenizer
            from datasets import load_dataset

            # Load a built-in dataset
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







EvaluationAgent
^^^^^^^^^^^^^^^

The ``EvaluationAgent`` is a class that evaluates the performance of your model.
It is a subclass of ``pl.LightningModule`` and ``pruna.SmashConfig``.






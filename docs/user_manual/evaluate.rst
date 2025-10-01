:title: Evaluation Agent Guide - Model Quality Assessment with Pruna AI
:description: Learn to evaluate AI model quality with Pruna's Evaluation Agent. Comprehensive guide to efficiency metrics, quality metrics, and model assessment workflows.

Evaluate quality with the Evaluation Agent
==========================================

This guide provides a comprehensive introduction to the Evaluation Agent in |pruna|, your all-in-one suite for evaluating and optimizing AI models.

Evaluation helps you understand how compression affects your models across different dimensions - from output quality to resource requirements.
This knowledge is essential for making informed decisions about which compression techniques work best for your specific needs using two types of metrics:

- **Efficiency Metrics:** Measure speed (total time, latency, throughput), memory (disk, inference, training), and energy usage (consumption, CO2 emissions).
- **Quality Metrics:** Assess fidelity (FID, CMMD), alignment (Clip Score), diversity (PSNR, SSIM), accuracy (accuracy, precision, perplexity), and more. Custom metrics are supported.

.. image:: /_static/assets/images/evaluation_agent.png
    :alt: Evaluation Agent
    :align: center
    :width: 100%

Haven't smashed a model yet? Check out the :doc:`optimize guide </docs_pruna/user_manual/smash>` to learn how to do that.

Basic Evaluation Workflow
-------------------------

|pruna| follows a simple workflow for evaluating model optimizations. You can use either the direct parameters approach or the Task-based approach:

**Direct Parameters Workflow:**

.. mermaid::
   :align: center

   graph LR
    User -->|configures| Metrics
    User -->|configures| PrunaDataModule
    PrunaModel -->|provides predictions| EvaluationAgent
    EvaluationAgent -->|evaluates| PrunaModel
    EvaluationAgent -->|returns| D["Evaluation Results"]

    subgraph E["Evaluation Configuration"]
        PrunaDataModule
        Metrics
    end

    Metrics-->|is used by| EvaluationAgent
    PrunaDataModule -->|is used by| EvaluationAgent
    User -->|creates| EvaluationAgent

    style User fill:#bbf,stroke:#333,stroke-width:2px
    style EvaluationAgent fill:#bbf,stroke:#333,stroke-width:2px
    style PrunaDataModule fill:#bbf,stroke:#333,stroke-width:2px
    style PrunaModel fill:#bbf,stroke:#333,stroke-width:2px
    style D fill:#bbf,stroke:#333,stroke-width:2px
    style Metrics fill:#bbf,stroke:#333,stroke-width:2px

**Task-based Workflow:**

.. mermaid::
   :align: center

   flowchart LR
    User -->|creates| Task
    User -->|creates| EvaluationAgent
    Task -->|defines| PrunaDataModule
    Task -->|defines| Metrics
    Task -->|is used by| EvaluationAgent
    Metrics -->|includes| B["Base Metrics"]
    Metrics -->|includes| C["Stateful Metrics"]
    PrunaModel -->|provides predictions| EvaluationAgent
    EvaluationAgent -->|evaluates| PrunaModel
    EvaluationAgent -->|returns| D["Evaluation Results"]
    User -->|configures| EvaluationAgent

    subgraph A["Metric Types"]
        B
        C
    end

    subgraph E["Task Definition"]
        Task
        PrunaDataModule
        Metrics
        A
    end

    style User fill:#bbf,stroke:#333,stroke-width:2px
    style Task fill:#bbf,stroke:#333,stroke-width:2px
    style EvaluationAgent fill:#bbf,stroke:#333,stroke-width:2px
    style PrunaDataModule fill:#bbf,stroke:#333,stroke-width:2px
    style PrunaModel fill:#bbf,stroke:#333,stroke-width:2px
    style D fill:#bbf,stroke:#333,stroke-width:2px
    style Metrics fill:#bbf,stroke:#333,stroke-width:2px
    style B fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#f9f,stroke:#333,stroke-width:2px

The implementation details and initialization options are covered in the sections below.

Evaluation Components
---------------------

The |pruna| package provides a variety of evaluation metrics to assess your models.
In this section, we'll introduce the evaluation metrics you can use.

EvaluationAgent Initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``EvaluationAgent`` is the main class for evaluating model performance. It can be initialized using two approaches:

.. tabs::

    .. tab:: Direct Parameters

        Pass request, datamodule, and device directly to the constructor:

        .. code-block:: python

            from pruna.evaluation.evaluation_agent import EvaluationAgent
            from pruna.data.pruna_datamodule import PrunaDataModule

            eval_agent = EvaluationAgent(
                request=["cmmd", "ssim"],
                datamodule=PrunaDataModule.from_string('LAION256'),
                device="cpu"
            )

    .. tab:: Task-based

        Create a Task object that encapsulates the configuration:

        .. code-block:: python

            from pruna.evaluation.evaluation_agent import EvaluationAgent
            from pruna.evaluation.task import Task
            from pruna.data.pruna_datamodule import PrunaDataModule

            task = Task(
                request=["cmmd", "ssim"],
                datamodule=PrunaDataModule.from_string('LAION256'),
                device="cpu"
            )
            eval_agent = EvaluationAgent(task)

Parameters
~~~~~~~~~~

- **request**: ``str | List[str | BaseMetric | StatefulMetric]`` - The metrics to evaluate
- **datamodule**: ``PrunaDataModule`` - The data module containing the evaluation dataset
- **device**: ``str | torch.device | None`` - The device to use for evaluation (defaults to best available)

Task
^^^^

The ``Task`` class provides an alternative way to define evaluation configurations. It encapsulates the evaluation parameters and can be passed directly to the ``EvaluationAgent`` constructor.

.. code-block:: python

    from pruna.evaluation.task import Task
    from pruna.data.pruna_datamodule import PrunaDataModule

    task = Task(
        request=["cmmd", "ssim"],
        datamodule=PrunaDataModule.from_string('LAION256'),
        device="cpu"
    )

Metrics
~~~~~~~

Metrics are the core components that calculate specific performance indicators. There are two main types of metrics:

- **Base Metrics**: These metrics compute values directly from inputs without maintaining state across batches.
- **Stateful Metrics**: Metrics that maintain internal state and accumulate information across multiple batches. These are typically used for quality assessment.

The ``EvaluationAgent`` accepts ``Metrics`` in three ways:

.. tabs::

    .. tab:: Predefined Options

        As a plain text request from predefined options (e.g., ``image_generation_quality``)

        .. code-block:: python

            from pruna.evaluation.evaluation_agent import EvaluationAgent
            from pruna.data.pruna_datamodule import PrunaDataModule

            eval_agent = EvaluationAgent(
                request ="image_generation_quality",
                datamodule=PrunaDataModule.from_string('LAION256'),
                device="cpu"
            )

    .. tab:: List of Metric Names

        As a list of metric names (e.g., [``"clip_score"``, ``"psnr"``])

        .. code-block:: python

            from pruna.evaluation.evaluation_agent import EvaluationAgent
            from pruna.data.pruna_datamodule import PrunaDataModule

            task = Task(
                request=["clip_score", "psnr"],
                datamodule=PrunaDataModule.from_string('LAION256'),
                device="cpu"
            )

    .. tab:: List of Metric Instances

        As a list of metric instances (e.g., ``CMMD()``), which provides more flexibility in configuring the metrics.

        .. code-block:: python

            from pruna.evaluation.evaluation_agent import EvaluationAgent
            from pruna.data.pruna_datamodule import PrunaDataModule
            from pruna.evaluation.metrics import CMMD, TorchMetricWrapper

            task = Task(
                request=[CMMD(call_type="pairwise"), TorchMetricWrapper(metric_name="clip_score")],
                datamodule=PrunaDataModule.from_string('LAION256'),
                device="cpu"
            )

.. note::

    You can find the full list of available metrics in the :ref:`Metric Overview <metrics>` section.

Metric Call Types
~~~~~~~~~~~~~~~~~

Stateful metrics can generally be evaluated in single-model and pariwise modes.
Single-mode allows to compare a mode against ground-truth data, while pairwise mode allows to compare the fidelity of model against another model.

- **Single-Model mode**: Each evaluation produces independent scores for the model being evaluated. IQA metrics are only supported in single-model mode.
- **Pairwise mode**: Metrics compare a subsequent model against the first model evaluated by the agent and produce a single comparison score.

Underneath the hood, the ``StatefulMetric`` class uses the ``call_type`` parameter to determine the order of the inputs.

Each metric has a default ``call_type`` but you can switch the mode of the metric despite your default ``call_type``.

.. tabs::

    .. tab:: Single-Model mode

        .. code-block:: python

            from pruna.evaluation.metrics import CMMD

            metric = CMMD(call_type="single") # or [CMMD() since single is the default call type]

    .. tab:: Pairwise mode

        .. code-block:: python

            from pruna.evaluation.metrics import CMMD

            metric = CMMD(call_type="pairwise")

These high-level modes abstract away the underlying input ordering. Internally, each metric uses a more specific call_type to determine the exact order of inputs passed to the metric function.

Internal Call Types
~~~~~~~~~~~~~~~~~~~~

The following table lists the supported internal call types and examples of metrics using them.
The following table lists the supported internal call types and examples of metrics using them.

This is what's happening under the hood when you pass ``call_type="single"`` or ``call_type="pairwise"`` to a metric.

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

   * - ``pairwise``
     - Pairwise mode to default to ``pairwise_y_gt`` or ``pairwise_gt_y``
     - ``psnr``, ``ssim``, ``lpips``, ``cmmd``

   * - ``pairwise_y_gt``
     - Base model's output first, then subsequent model's output
     -  ``psnr``, ``ssim``, ``lpips``, ``cmmd``

   * - ``pairwise_gt_y``
     - Subsequent model's output first, then base model's output
     - ``psnr``, ``ssim``, ``lpips``, ``cmmd``

   * - ``y``
     - Only the output is used, the metric has an internal dataset
     - ``arniqa``

Metric Results
~~~~~~~~~~~~~~~

The ``MetricResult`` is a class that contains the result of a metric evaluation.

Each metric returns a ``MetricResult`` instance, which contains the outcome of a single evaluation.

The ``MetricResult`` class stores the metric's name, any associated parameters, and the computed result value:

.. container:: hidden_code

    .. code-block:: python

        from pruna.evaluation.metrics.result import MetricResult

.. code-block:: python

    # Example output
    MetricResult(
        name="clip_score",
        params={"param1": "value1", "param2": "value2"},
        result=28.0828
    )

PrunaDataModule
~~~~~~~~~~~~~~~

The ``PrunaDataModule`` is a class that defines the data you want to evaluate your model on.
Data modules are a core component of the evaluation framework, providing standardized access to datasets for evaluating model performance before and after optimization.

A more detailed overview of the ``PrunaDataModule``, its datasets and their corresponding collate functions can be found in the :doc:`Data Module Overview </docs_pruna/user_manual/configure>` section.

The ``EvaluationAgent`` accepts ``PrunaDataModule`` in two different ways:

.. tabs::

    .. tab:: From String

        As a plain text request from predefined options (e.g., ``WikiText``)

        .. code-block:: python

            from transformers import AutoTokenizer

            from pruna.data.pruna_datamodule import PrunaDataModule

            # Load the tokenizer
            tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-3.2-1B")
            tokenizer.pad_token = tokenizer.eos_token

            # Create the data Module
            datamodule = PrunaDataModule.from_string(
                dataset_name="WikiText",
                tokenizer=tokenizer,
                collate_fn_args={"max_seq_len": 512},
                dataloader_args={"batch_size": 16, "num_workers": 4},
            )

    .. tab:: From Datasets

        As a list of datasets, which provides more flexibility in configuring the data module.

        .. code-block:: python

            from datasets import load_dataset
            from transformers import AutoTokenizer

            from pruna.data.pruna_datamodule import PrunaDataModule
            from pruna.data.utils import split_train_into_train_val_test

            # Load the tokenizer
            tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-3.2-1B")
            tokenizer.pad_token = tokenizer.eos_token

            # Load custom datasets
            train_ds = load_dataset("SamuelYang/bookcorpus")["train"]
            train_ds, val_ds, test_ds = split_train_into_train_val_test(train_ds, seed=42)

            # Create the data module
            datamodule = PrunaDataModule.from_datasets(
                datasets=(train_ds, val_ds, test_ds),
                collate_fn="text_generation_collate",
                tokenizer=tokenizer,
                collate_fn_args={"max_seq_len": 512},
                dataloader_args={"batch_size": 16, "num_workers": 4},
            )

.. tip::

    You can find the full list of available datasets in the :doc:`Dataset Overview </docs_pruna/user_manual/configure>` section.

Lastly, you can limit the number of samples in the dataset by using the ``PrunaDataModule.limit_samples`` method.

.. code-block:: python

    from transformers import AutoTokenizer

    from pruna.data.pruna_datamodule import PrunaDataModule

    # Create the data module
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token
    datamodule = PrunaDataModule.from_string("WikiText", tokenizer=tokenizer)

    # Limit all splits to 100 samples
    datamodule.limit_datasets(100)

    # Use different limits for each split
    datamodule.limit_datasets([50, 10, 20])  # train, val, test

Evaluation Examples
-------------------

The ``EvaluationAgent`` evaluates model performance and can work in both single-model and pairwise modes.

- **Single-Model mode**: each model is evaluated independently, producing metrics that only pertain to that model's performance. The metrics are computed from the model's outputs without reference to any other model.
- **Pairwise mode**: metrics compare the outputs of the current model against the first model evaluated by the agent. The first model's outputs are cached by the EvaluationAgent and used as a reference for subsequent evaluations.

Let's see how this works in code.

.. tabs::

    .. tab:: Single-Model Evaluation

        .. code-block:: python

            from diffusers import DiffusionPipeline

            from pruna import SmashConfig, smash
            from pruna.data.pruna_datamodule import PrunaDataModule
            from pruna.evaluation.evaluation_agent import EvaluationAgent
            from pruna.evaluation.metrics import CMMD
            from pruna.evaluation.task import Task

            # Load data and set up smash config
            smash_config = SmashConfig()
            smash_config["quantizer"] = "hqq_diffusers"

            # Load the base model
            model_path = "segmind/Segmind-Vega"
            pipe = DiffusionPipeline.from_pretrained(model_path)

            # Smash the model
            smashed_pipe = smash(pipe, smash_config)

            # Define the task and the evaluation agent
            metrics = [CMMD()]
            datamodule = PrunaDataModule.from_string("LAION256")
            datamodule.limit_datasets(5)
            task = Task(metrics, datamodule=datamodule)
            eval_agent = EvaluationAgent(task)

            # Optional: tweak model generation parameters for benchmarking
            smashed_pipe.inference_handler.model_args.update(
                {"num_inference_steps": 1, "guidance_scale": 0.0}
            )

            # Evaluate base model, all models need to be wrapped in a PrunaModel before passing them to the EvaluationAgent
            first_results = eval_agent.evaluate(pipe)

    .. tab:: Pairwise Evaluation

        .. code-block:: python

            import copy

            from diffusers import DiffusionPipeline

            from pruna import SmashConfig, smash
            from pruna.data.pruna_datamodule import PrunaDataModule
            from pruna.evaluation.evaluation_agent import EvaluationAgent
            from pruna.evaluation.metrics import CMMD
            from pruna.evaluation.task import Task

            # Load data and set up smash config
            smash_config = SmashConfig()
            smash_config["quantizer"] = "hqq_diffusers"

            # Load the base model
            model_path = "segmind/Segmind-Vega"
            pipe = DiffusionPipeline.from_pretrained(model_path)

            # Smash the model
            copy_pipe = copy.deepcopy(pipe)
            smashed_pipe = smash(copy_pipe, smash_config)

            # Define the task and the evaluation agent
            metrics = [CMMD(call_type="pairwise")]
            datamodule = PrunaDataModule.from_string("LAION256")
            datamodule.limit_datasets(5)
            task = Task(metrics, datamodule=datamodule)
            eval_agent = EvaluationAgent(task)

            # wrap the model in a PrunaModel to use the EvaluationAgent
            wrapped_pipe = PrunaModel(pipe, None)

            # Optional: tweak model generation parameters for benchmarking
            inference_arguments = {"num_inference_steps": 1, "guidance_scale": 0.0}
            wrapped_pipe.inference_handler.model_args.update(inference_arguments)


            # Evaluate base model first (cached for comparison)
            first_results = eval_agent.evaluate(pipe)

            # Evaluate smashed model (compared against base model)
            smashed_results = eval_agent.evaluate(smashed_pipe)
            print(smashed_results)

EvaluationAgent Initialization Options
--------------------------------------

You can choose between the two initialization approaches shown above based on your preference and project requirements. Both approaches provide identical functionality and can be used interchangeably.

Best Practices
--------------

Start with a small dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^

When first setting up evaluation, limit the dataset size with ``datamodule.limit_datasets(n)`` to make debugging faster.

Use pairwise metrics for comparison
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When comparing an optimized model against the baseline, use pairwise metrics to get direct comparison scores.

Choose your initialization style
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Both direct parameters and Task-based initialization are valid approaches. Choose the one that best fits your project's coding patterns and requirements.

.. _face_score:

FaceScore Metric
~~~~~~~~~~~~~~~~

The ``face_score`` metric evaluates the quality of human faces in generated images using the FaceScore reward model.

**Dependencies:**
- Install with: ``pip install batch-face image-reward``

**Model Weights Required:**
- Download ``FS_model.pt`` and ``med_config.json`` from the official FaceScore repository: https://huggingface.co/OPPOer/FaceScore
- Place both files in ``C:\Users\<your-username>\.cache\FaceScore\`` (or as specified in the metric code).

**Example Usage:**

.. code-block:: python

    from pruna.evaluation.evaluation_agent import EvaluationAgent
    from pruna.data.pruna_datamodule import PrunaDataModule

    eval_agent = EvaluationAgent(
        request=["clip_score", "face_score"],
        datamodule=PrunaDataModule.from_string('LAION256'),
        device="cpu"
    )
    results = eval_agent.evaluate(model)

**Note:**  
If the required model files are not present, the metric will raise a clear error with instructions.

FaceScore is a stateful metric and can be used alongside other quality metrics in the Pruna evaluation pipeline.

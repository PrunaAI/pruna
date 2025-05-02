.. _evaluation:

Evaluation Metrics
===================

The |pruna| package provides helpful evaluation tools to assess your models. In this section, we'll introduce the evaluation metrics you can use with the package.

Evaluation helps you understand how compression affects your models across different dimensions - from output quality to resource requirements. This knowledge is essential for making informed decisions about which compression techniques work best for your specific needs.

.. _quicktutorial:

Quick Tutorial
--------------

Before we start, here's a simple example showing how to evaluate your models using |pruna|.

The rest of this guide provides more detailed explanations of each component and additional features available for model evaluation.

.. code-block:: python

  import copy

  from diffusers import StableDiffusionPipeline

  from pruna import smash, SmashConfig
  from pruna.data.pruna_datamodule import PrunaDataModule
  from pruna.evaluation.evaluation_agent import EvaluationAgent
  from pruna.evaluation.task import Task

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
  metrics = ['clip_score', 'psnr']
  task = Task(metrics, datamodule=PrunaDataModule.from_string('LAION256')) 
  eval_agent = EvaluationAgent(task)

  # Evaluate base model, all models need to be wrapped in a PrunaModel before passing them to the EvaluationAgent
  first_results = eval_agent.evaluate(pipe) 
  print(first_results)

  # Evaluate smashed model
  smashed_results = eval_agent.evaluate(smashed_pipe)
  print(smashed_results)


.. code-block:: python

  # Base model result output
  {'clip_score_y_x': 28.0828}

  # Smashed model result output
  {'clip_score_y_x': 28.4500, 'psnr_pairwise_y_gt': 18.7465}

Evaluation Framework
--------------------

The evaluation framework in |pruna| consists of several key components:

Task
^^^^
Processes user requests and converts them into a set of metrics. The ``Task`` accepts metrics in three ways:

- As a plain text request from predefined options (e.g., ``image_generation_quality``)
- As a list of metric names (e.g., [``"clip_score"``, ``"psnr"``])  (see :ref:`Available Metrics <metrics>` below)
- As a list of metric instances

In addition to metrics, ``Task`` requires a :ref:`PrunaDataModule <prunadatamodule>` to perform the evaluation.

.. autoclass:: pruna.evaluation.task.Task

Currently, ``Task`` supports the following plain text requests:

- ``image_generation_quality``: Creates metrics for evaluating image generation models (``clip_score``, ``pairwise_clip_score``, ``psnr``)


.. code-block:: python

  from pruna.evaluation.task import Task
  from pruna.data.pruna_datamodule import PrunaDataModule

  task = Task("image_generation_quality", datamodule=PrunaDataModule.from_string('LAION256')) 

EvaluationAgent
^^^^^^^^^^^^^^^
The main entry point for evaluating models. The ``EvaluationAgent``:

- Takes a ``Task`` object that defines what metrics to use
- Provides methods to evaluate any model
- Handles the evaluation process, including separating metrics by execution strategy
- Runs inference on the model to generate predictions
- Caches predictions to avoid redundant computations
- Passes ground truth data and predictions to the appropriate metrics
- Collects and returns results from all metrics

.. autoclass:: pruna.evaluation.evaluation_agent.EvaluationAgent
  :members: evaluate

.. container:: hidden_code

    .. code-block:: python

      from pruna.evaluation.task import Task
      from pruna.data.pruna_datamodule import PrunaDataModule

      data_module = PrunaDataModule.from_string('LAION256')
      data_module.limit_datasets(10)

      task = Task("image_generation_quality", datamodule=data_module)

.. code-block:: python

  from pruna.evaluation.evaluation_agent import EvaluationAgent

  eval_agent = EvaluationAgent(task)

For the full example running evaluation please see :ref:`Quick Tutorial <quicktutorial>` above.

.. _metrics:

Metrics
-------

Metrics help quantify different aspects of model performance, from output quality to resource requirements. The |pruna| package includes metrics for both quality assessment and resource utilization.

When using the ``EvaluationAgent``, all metrics are executed automatically as part of the evaluation pipeline. The agent handles model inference, data preparation, and passing the appropriate inputs to each metric, eliminating the need to run metrics individually.

total_time
^^^^^^^^^^^

Measures total time taken to run the model.

:Evaluation on CPU: Yes.
:Required:
  A PrunaModel object that defines the model to evaluate.
  A DataLoader object that defines the dataloader to evaluate the model on.
:Parameters:

  | ``n_iterations``: Number of inference iterations to measure (default 100).
  | ``n_warmup_iterations``: Number of warmup iterations before measurement (default 10).
  | ``device``: Device to run inference on (default "cuda").
  | ``timing_type``: Type of timing to use ("sync" or "async", default "sync").


latency
^^^^^^^

Measures the latency of the model.

:Evaluation on CPU: Yes.
:Required:
  A PrunaModel object that defines the model to evaluate.
  A DataLoader object that defines the dataloader to evaluate the model on.
:Parameters:

  | ``n_iterations``: Number of inference iterations to measure (default 100).
  | ``n_warmup_iterations``: Number of warmup iterations before measurement (default 10).
  | ``device``: Device to run inference on (default "cuda").
  | ``timing_type``: Type of timing to use ("sync" or "async", default "sync").


throughput
^^^^^^^^^^^

Measures the throughput of the model.

:Evaluation on CPU: Yes.
:Required:
  A PrunaModel object that defines the model to evaluate.
  A DataLoader object that defines the dataloader to evaluate the model on.
:Parameters:

  | ``n_iterations``: Number of inference iterations to measure (default 100).
  | ``n_warmup_iterations``: Number of warmup iterations before measurement (default 10).
  | ``device``: Device to run inference on (default "cuda").
  | ``timing_type``: Type of timing to use ("sync" or "async", default "sync").


`disk_memory <https://pypi.org/project/nvidia-ml-py/>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Measures peak GPU memory usage during model loading.

:Evaluation on CPU: No.
:Required:
  Path to the PrunaModel to evaluate.
  A DataLoader object that defines the dataloader to evaluate the model on.
  The model class to load the model from the path. 
:Parameters:
  | ``gpu_indices``: List of GPU indices to monitor (default all available GPUs).


`inference_memory <https://pypi.org/project/nvidia-ml-py/>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Measures peak GPU memory usage during model inference.

:Evaluation on CPU: No.
:Required:
  Path to the PrunaModel to evaluate.
  A DataLoader object that defines the dataloader to evaluate the model on.
  The model class to load the model from the path. 
:Parameters:
  | ``gpu_indices``: List of GPU indices to monitor (default all available GPUs).


`training_memory <https://pypi.org/project/nvidia-ml-py/l>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Measures peak GPU memory usage during model training.

:Evaluation on CPU: No.
:Required:
  Path to the PrunaModel to evaluate.
  A DataLoader object that defines the dataloader to evaluate the model on.
  The model class to load the model from the path. 
:Parameters:
  | ``gpu_indices``: List of GPU indices to monitor (default all available GPUs).

`energy_consumed <https://github.com/mlco2/codecarbon>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Measures energy consumption in kilowatt-hours (kWh)

:Evaluation on CPU: Yes.
:Description: Measures energy consumption in kilowatt-hours (kWh).
:Required: 
    A PrunaModel object that defines the model to evaluate.
    A DataLoader object that defines the dataloader to evaluate the model on.
:Parameters:

  | ``n_iterations``: Number of inference iterations to measure (default 100).
  | ``n_warmup_iterations``: Number of warmup iterations before measurement (default 10).
  | ``device``: Device to run inference on (default "cuda").


`co2_emissions <https://github.com/mlco2/codecarbon>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Measures CO2 emissions in kilograms (kg).

:Evaluation on CPU: Yes.
:Description: Measures CO2 emissions in kilograms (kg).
:Required: 
    A PrunaModel object that defines the model to evaluate.
    A DataLoader object that defines the dataloader to evaluate the model on.
:Parameters:

  | ``n_iterations``: Number of inference iterations to measure (default 100).
  | ``n_warmup_iterations``: Number of warmup iterations before measurement (default 10).
  | ``device``: Device to run inference on (default "cuda").

`total_params <https://github.com/Lyken17/pytorch-OpCounter>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Measures the number of parameters of the model.

:Evaluation on CPU: Yes.
:Required: 
    A PrunaModel object that defines the model to evaluate.
    A DataLoader object that defines the dataloader to evaluate the model on.
:Parameters:

  | ``device``: Device to evaluate the model on (default "cuda").


`total_macs <https://github.com/Lyken17/pytorch-OpCounter>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Measures the number of MACs (multiply-accumulate operations) in the model.

:Evaluation on CPU: Yes.
:Required: 
    A PrunaModel object that defines the model to evaluate.
    A DataLoader object that defines the dataloader to evaluate the model on.
:Parameters:

  | ``device``: Device to evaluate the model on (default "cuda").


`clip_score <https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/multimodal/clip_score.py>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Measures the similarity between images and text using the CLIP model.


:Evaluation on CPU: Yes.
:Required: Inputs, ground truth and predictions.
:Parameters: Accepts all parameters from the TorchMetrics CLIPScore implementation.

`pairwise_clip_score <https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/multimodal/clip_score.py>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Measures the similarity between images of first and subsequent models using the CLIP model.

:Evaluation on CPU: Yes.
:Required: Inputs, ground truth and predictions.
:Parameters: Accepts all parameters from the TorchMetrics CLIPScore implementation.

`cmmd <https://arxiv.org/abs/2401.09603>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CMMD measures the distributional discrepancy between two sets of images or text by computing Maximum Mean Discrepancy (MMD) in the CLIP embedding space. It captures both semantic and visual alignment.

Key Benefits:

- **Distribution-Free:** Does not rely on any assumptions about the underlying feature distribution.
- **Unbiased Estimation:** Provides a statistically unbiased measure of the discrepancy between two image sets.
- **Sample Efficiency:** Achieves reliable estimates even with smaller image samples, making it suitable for rapid evaluations.
- **Human-Aligned:** Demonstrates better agreement with human perceptual assessments of image quality compared to FID.

:Evaluation on CPU: Yes.
:Required: Inputs, ground truth and predictions.
:Parameters: 

  | ``clip_model_name``: Name of the CLIP model to use (default "openai/clip-vit-large-patch14-336").
  | ``call_type``: Call type to use for the metric (default "gt_y"). For pairwise evaluation pass "pairwise" or "pairwise_gt_y".    
  | ``device``: Device to run the metric on (default "cuda").



`accuracy <https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/classification/accuracy.py>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Measures the proportion of correct predictions in classification tasks.


:Evaluation on CPU: Yes.
:Required: Inputs, ground truth and predictions. TorchMetrics requires a 'task' parameter to be set to 'binary', 'multiclass', or 'multilabel'. Each task type may have additional specific requirements - please refer to the TorchMetrics documentation for details.
:Parameters: Accepts all parameters from the TorchMetrics Accuracy implementation (task, num_classes, threshold, etc.).

`precision <https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/classification/precision_recall.py>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Measures the proportion of positive identifications that were actually correct.


:Evaluation on CPU: Yes.
:Required: Inputs, ground truth and predictions. TorchMetrics requires a 'task' parameter to be set to 'binary', 'multiclass', or 'multilabel'. Each task type may have additional specific requirements - please refer to the TorchMetrics documentation for details.
:Parameters: Accepts all parameters from the TorchMetrics Precision implementation (task, num_classes, threshold, etc.).

`recall <https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/classification/precision_recall.py>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Measures the proportion of actual positives that were identified correctly.

 
:Evaluation on CPU: Yes.
:Required: Inputs, ground truth and predictions. TorchMetrics requires a 'task' parameter to be set to 'binary', 'multiclass', or 'multilabel'. Each task type may have additional specific requirements - please refer to the TorchMetrics documentation for details.
:Parameters: Accepts all parameters from the TorchMetrics Recall implementation (task, num_classes, threshold, etc.).

`perplexity <https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/text/perplexity.py>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Measures how well a probability model predicts a text sample.

:Evaluation on CPU: Yes.
:Required: Inputs, ground truth and predictions. 
:Parameters: Accepts all parameters from the TorchMetrics Perplexity implementation.

`fid <https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/image/fid.py>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Measures the similarity between generated and real image distributions using the Frechet Distance between Gaussian distributions fitted to the Inception embeddings of the generated and real images.

FID compares the **distribution** of real and generated images in a high-dimensional feature space. Since it estimates **mean and covariance statistics**, smaller sample sizes can introduce high variance, making the metric less stable. Large-scale evaluations often use **tens of thousands of images**, but for practical use, smaller sample sizes may still provide a reasonable approximation.

**Computation Considerations**  

When generating images and computing FID on **thousands to tens of thousands of samples**, the process can take **multiple hours to several days**, even on a high-end GPU like an **A100 or RTX 4090**. On mid-range GPUs like a **3060 or 4060**, it can take **significantly longer**. A rough approximation using **a few thousand images** may still take **several hours**, even with strong hardware.

:Evaluation on CPU: No (impractical due to the high computational cost)
:Required: Inputs, ground truth and predictions.
:Parameters: Accepts all parameters from the TorchMetrics FrechetInceptionDistance implementation (feature extraction parameters, etc.).

`psnr <https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/image/psnr.py>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Measures the peak signal-to-noise ratio (PSNR) between two images.

:Evaluation on CPU: Yes.
:Required: Inputs, ground truth and predictions.
:Parameters: Accepts all parameters from the TorchMetrics PSNR implementation.

`ssim <https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/image/ssim.py>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Measures the structural similarity index (SSIM) between two images.

:Evaluation on CPU: Yes.
:Required: Inputs, ground truth and predictions.
:Parameters: Accepts all parameters from the TorchMetrics SSIM implementation.

`lpips <https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/image/lpip.py>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Measures the Learned Perceptual Image Patch Similarity (LPIPS) between two images.

:Evaluation on CPU: Yes.
:Required: Inputs, ground truth and predictions.
:Parameters: Accepts all parameters from the TorchMetrics LPIPS implementation.
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

  # Load the base model and wrap it in a PrunaModel
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

  # Evaluate smashed model
  smashed_results = eval_agent.evaluate(smashed_pipe)
  

Evaluation Framework
--------------------

The evaluation framework in |pruna| consists of several key components:

Task
^^^^
Processes user requests and converts them into a set of metrics. The ``Task`` accepts metrics in three ways:

- As a plain text request from predefined options (e.g., ``image_generation_quality``)
- As a list of metric names (e.g., [``clip_score``, ``psnr``])  (see :ref:`Available Metrics <metrics>` below)
- As a list of metric instances

In addition to metrics, ``Task`` requires a |pruna| datamodule to perform the evaluation.

.. autoclass:: pruna.evaluation.task.Task

Currently, ``Task`` supports the following plain textrequests:

- ``image_generation_quality``: Creates metrics for evaluating image generation models (``clip_score``, ``pairwise_clip_score``, ``psnr``)


.. code-block:: python

  from pruna.evaluation.task import Task

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

Metrics can operate in both single-model and pairwise modes:

- In single-model mode, each evaluation produces independent scores for the model being evaluated.
- In pairwise mode, metrics compare a subsequent model against the first model evaluated by the agent. The first model's outputs are cached and used as a reference point for all following evaluations. The pairwise comparison produces a single score that quantifies the relationship (e.g., similarity or difference) between the two models.

Our metrics fall into two implementation categories that work differently under the hood:

Base Metrics
^^^^^^^^^^^^^
Simple metrics that compute values directly from inputs without maintaining state across batches. Examples include:
- Model Architecture metrics
- Energy consumption metrics
- Memory usage metrics

`elapsed_time <https://github.com/Lightning-AI/torchmetrics>`_
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Measures inference time, latency, and throughput.

:Evaluation on CPU: Yes.
:Required:
  A PrunaModel object that defines the model to evaluate.
  A DataLoader object that defines the dataloader to evaluate the model on.
:Parameters:

  | ``n_iterations``: Number of inference iterations to measure (default 100).
  | ``n_warmup_iterations``: Number of warmup iterations before measurement (default 10).
  | ``device``: Device to run inference on (default "cuda").
  | ``timing_type``: Type of timing to use ("sync" or "async", default "sync").

`gpu_memory <https://github.com/NVIDIA/pynvml>`_
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Measures peak GPU memory usage during model loading and execution.

:Evaluation on CPU: No.
:Required:
  Path to the PrunaModel to evaluate.
  A DataLoader object that defines the dataloader to evaluate the model on.
  The model class to load the model from the path. 
:Parameters:

  | ``mode``: Memory measurement mode ("disk", "inference", or "training").
  | ``gpu_indices``: List of GPU indices to monitor (default all available GPUs).

`energy <https://github.com/mlco2/codecarbon>`_
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Measures energy consumption in kilowatt-hours (kWh) and CO2 emissions in kilograms (kg).

:Evaluation on CPU: Yes.
:Description: Measures energy consumption in kilowatt-hours (kWh) and CO2 emissions in kilograms (kg).
:Required: 
    A PrunaModel object that defines the model to evaluate.
    A DataLoader object that defines the dataloader to evaluate the model on.
:Parameters:

  | ``n_iterations``: Number of inference iterations to measure (default 100).
  | ``n_warmup_iterations``: Number of warmup iterations before measurement (default 10).
  | ``device``: Device to run inference on (default "cuda").

`model_architecture <https://github.com/Lyken17/pytorch-OpCounter>`_
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Measures the number of parameters and MACs (multiply-accumulate operations) in the model.

:Evaluation on CPU: Yes.
:Required: 
    A PrunaModel object that defines the model to evaluate.
    A DataLoader object that defines the dataloader to evaluate the model on.
:Parameters:

  | ``device``: Device to evaluate the model on (default "cuda").

Stateful Metrics
^^^^^^^^^^^^^^^^^
Metrics that maintain internal state and accumulate information across multiple batches. These are typically used for quality assessment.

Most of our stateful metrics are implemented using the TorchMetricsWrapper, which adapts metrics from the `TorchMetrics <https://github.com/Lightning-AI/torchmetrics>`_ library to work within our evaluation framework. This allows us to leverage the robust implementations provided by TorchMetrics while maintaining a consistent API.

`clip_score <https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/multimodal/clip_score.py>`_
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Measures the similarity between images and text using the CLIP model.


:Evaluation on CPU: Yes.
:Required: Inputs, ground truth and predictions.
:Parameters: Accepts all parameters from the TorchMetrics CLIPScore implementation.

`pairwise_clip_score <https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/multimodal/clip_score.py>`_
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Measures the similarity between images of first and subsequent models using the CLIP model.

:Evaluation on CPU: Yes.
:Required: Inputs, ground truth and predictions.
:Parameters: Accepts all parameters from the TorchMetrics CLIPScore implementation.


`accuracy <https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/classification/accuracy.py>`_
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Measures the proportion of correct predictions in classification tasks.


:Evaluation on CPU: Yes.
:Required: Inputs, ground truth and predictions. TorchMetrics requires a 'task' parameter to be set to 'binary', 'multiclass', or 'multilabel'. Each task type may have additional specific requirements - please refer to the TorchMetrics documentation for details.
:Parameters: Accepts all parameters from the TorchMetrics Accuracy implementation (task, num_classes, threshold, etc.).

`precision <https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/classification/precision.py>`_
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Measures the proportion of positive identifications that were actually correct.


:Evaluation on CPU: Yes.
:Required: Inputs, ground truth and predictions. TorchMetrics requires a 'task' parameter to be set to 'binary', 'multiclass', or 'multilabel'. Each task type may have additional specific requirements - please refer to the TorchMetrics documentation for details.
:Parameters: Accepts all parameters from the TorchMetrics Precision implementation (task, num_classes, threshold, etc.).

`recall <https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/classification/recall.py>`_
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Measures the proportion of actual positives that were identified correctly.

 
:Evaluation on CPU: Yes.
:Required: Inputs, ground truth and predictions. TorchMetrics requires a 'task' parameter to be set to 'binary', 'multiclass', or 'multilabel'. Each task type may have additional specific requirements - please refer to the TorchMetrics documentation for details.
:Parameters: Accepts all parameters from the TorchMetrics Recall implementation (task, num_classes, threshold, etc.).

`perplexity <https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/text/perplexity.py>`_
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Measures how well a probability model predicts a text sample.

:Evaluation on CPU: Yes.
:Required: Inputs, ground truth and predictions. 
:Parameters: Accepts all parameters from the TorchMetrics Perplexity implementation.

`fid <https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/image/fid.py>`_
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Measures the similarity between generated and real image distributions using the Frechet Distance between Gaussian distributions fitted to the Inception embeddings of the generated and real images.

FID compares the **distribution** of real and generated images in a high-dimensional feature space. Since it estimates **mean and covariance statistics**, smaller sample sizes can introduce high variance, making the metric less stable. Large-scale evaluations often use **tens of thousands of images**, but for practical use, smaller sample sizes may still provide a reasonable approximation.

**Computation Considerations**  

When generating images and computing FID on **thousands to tens of thousands of samples**, the process can take **multiple hours to several days**, even on a high-end GPU like an **A100 or RTX 4090**. On mid-range GPUs like a **3060 or 4060**, it can take **significantly longer**. A rough approximation using **a few thousand images** may still take **several hours**, even with strong hardware.

:Evaluation on CPU: No (impractical due to the high computational cost)
:Required: Inputs, ground truth and predictions.
:Parameters: Accepts all parameters from the TorchMetrics FrechetInceptionDistance implementation (feature extraction parameters, etc.).

`psnr <https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/image/psnr.py>`_
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Measures the peak signal-to-noise ratio (PSNR) between two images.

:Evaluation on CPU: Yes.
:Required: Inputs, ground truth and predictions.
:Parameters: Accepts all parameters from the TorchMetrics PSNR implementation.

`ssim <https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/image/ssim.py>`_
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Measures the structural similarity index (SSIM) between two images.

:Evaluation on CPU: Yes.
:Required: Inputs, ground truth and predictions.
:Parameters: Accepts all parameters from the TorchMetrics SSIM implementation.

`lpips <https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/image/lpip.py>`_
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Measures the Learned Perceptual Image Patch Similarity (LPIPS) between two images.

:Evaluation on CPU: Yes.
:Required: Inputs, ground truth and predictions.
:Parameters: Accepts all parameters from the TorchMetrics LPIPS implementation.
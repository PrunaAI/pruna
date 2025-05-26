Smash your first model
======================

This guide provides a quick introduction to optimizing AI models with |pruna|.

You'll learn how to use Pruna's core functionality to make your models faster, smaller, cheaper, and greener.
For installation instructions, see :doc:`Installation </setup/pip>`.

Basic Optimization Workflow
---------------------------

|pruna| follows a simple workflow for optimizing models:

.. mermaid::
   :align: center

   graph LR
      A[Load Model] --> B[Define SmashConfig]
      B --> C[Smash Model]
      C --> D[Evaluate Model]
      D --> E[Run Inference]
      style A fill:#bbf,stroke:#333,stroke-width:2px
      style B fill:#bbf,stroke:#333,stroke-width:2px
      style C fill:#bbf,stroke:#333,stroke-width:2px
      style D fill:#bbf,stroke:#333,stroke-width:2px
      style E fill:#bbf,stroke:#333,stroke-width:2px

Let's see what that looks like in code.

.. code-block:: python

    from pruna import smash, SmashConfig
    from diffusers import StableDiffusionPipeline
    from pruna.data.pruna_datamodule import PrunaDataModule
    from pruna.evaluation.evaluation_agent import EvaluationAgent
    from pruna.evaluation.task import Task

    # Load the model
    model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

    # Create and configure SmashConfig
    smash_config = SmashConfig()
    smash_config["cacher"] = "deepcache"

    # Smash the model
    optimized_model = smash(model=model, smash_config=smash_config)

    # Evaluate the model
    metrics = ['clip_score', 'psnr']
    task = Task(metrics, datamodule=PrunaDataModule.from_string('LAION256'))
    eval_agent = EvaluationAgent(task)
    eval_agent.evaluate(optimized_model)

    # Run inference
    optimized_model.set_progress_bar_config(disable=True)
    optimized_model.inference_handler.model_args.update(
        {"num_inference_steps": 1, "guidance_scale": 0.0}
    )
    optimized_model("A serene landscape with mountains").images[0]


Step-by-Step Optimisation Workflow
----------------------------------

Step 1: Load a pretrained model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, load any model using its original library, like ``transformers`` or ``diffusers``:

.. code-block:: python

    from diffusers import StableDiffusionPipeline

    base_model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")


Step 2: Define optimizations with a ``SmashConfig``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After loading the model, we can define a ``SmashConfig`` to customize the optimizations we want to apply.
This ``SmashConfig`` is a dictionary-like object that configures which optimizations to apply to your model.
You can specify multiple optimization algorithms from different categories like batching, caching and quantization.

For now, let's just use a ``cacher`` to accelerate the model during inference.

.. code-block:: python

    from pruna import SmashConfig

    smash_config = SmashConfig()
    smash_config["cacher"] = "deepcache"  # Accelerate the model with caching

Pruna support a wide range of algorithms for specific optimizations, all with different trade-offs.
To understand how to configure the right one for your scenario, see :doc:`Define a SmashConfig </docs_pruna/user_manual/configure>`.

Step 3: Apply optimizations with ``smash``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``smash()`` function is the core of Pruna. It takes your model and ``SmashConfig``, applies the specified optimizations.
Let's use the ``smash()`` function to apply the configured optimizations:

.. code-block:: python

    from pruna import smash

    optimized_model = smash(model=base_model, smash_config=smash_config)


The ``smash()`` function returns a ``PrunaModel`` - a wrapper that provides a standardized interface for the optimized model. So, we can still use the model as we would use the original one.

Step 4: Evaluate the optimized model with the ``EvaluationAgent``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To evaluate the optimized model, we can use the same interface as the original model.

.. code-block:: python

    from pruna.data.pruna_datamodule import PrunaDataModule
    from pruna.evaluation.evaluation_agent import EvaluationAgent

    metrics = ['clip_score', 'psnr']
    task = Task(metrics, datamodule=PrunaDataModule.from_string('LAION256'))
    eval_agent = EvaluationAgent(task)
    eval_agent.evaluate(optimized_model)

To understand how to run more complex evaluation workflows, see :doc:`Evaluate a model </docs_pruna/user_manual/evaluate>`.

Step 5: Run inference with the optimized model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run inference with the optimized model, we can use the same interface as the original model.

.. code-block:: python

    optimized_model.set_progress_bar_config(disable=True)
    optimized_model.inference_handler.model_args.update(
        {"num_inference_steps": 1, "guidance_scale": 0.0}
    )
    optimized_model("A serene landscape with mountains").images[0]

Example use cases
-----------------

Let's look at some specific examples for different model types.

Example 1: Diffusion Model Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from diffusers import StableDiffusionPipeline
    from pruna import smash, SmashConfig

    # Load the model
    model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

    # Create and configure SmashConfig
    smash_config = SmashConfig()
    smash_config["cacher"] = "deepcache"
    smash_config["compiler"] = "stable_fast"

    # Optimize the model
    optimized_model = smash(model=model, smash_config=smash_config)

    # Generate an image
    optimized_model("A serene landscape with mountains").images[0]

Example 2: Large Language Model Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from transformers import AutoModelForCausalLM
    from pruna import smash, SmashConfig

    # Load the model
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

    # Create and configure SmashConfig
    smash_config = SmashConfig()
    smash_config["quantizer"] = "gptq"  # Apply GPTQ quantization

    # Optimize the model
    optimized_model = smash(model=model, smash_config=smash_config)

    # Use the model for generation
    input_text = "The best way to learn programming is"
    optimized_model(input_text)


Example 3: Speech Recognition Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from transformers import AutoModelForSpeechSeq2Seq
    from pruna import smash, SmashConfig
    import torch

    # Load the model
    model_id = "openai/whisper-large-v3"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to("cuda")

    # Create and configure SmashConfig
    smash_config = SmashConfig()
    smash_config.add_processor(model_id)  # Required for Whisper
    smash_config["compiler"] = "c_whisper"
    smash_config["batcher"] = "whisper_s2t"

    # Optimize the model
    optimized_model = smash(model=model, smash_config=smash_config)

    # Use the model for transcription
    optimized_model("audio_file.wav")
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

    from diffusers import StableDiffusionPipeline

    from pruna import SmashConfig, smash
    from pruna.data.pruna_datamodule import PrunaDataModule
    from pruna.evaluation.evaluation_agent import EvaluationAgent
    from pruna.evaluation.task import Task

    # Load the model
    model = StableDiffusionPipeline.from_pretrained("Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers")

    # Create and configure SmashConfig
    smash_config = SmashConfig()
    smash_config["quantizer"] = "hqq_diffusers"

    # Smash the model
    optimized_model = smash(model=model, smash_config=smash_config)

    # Evaluate the model
    metrics = ["clip_score", "psnr"]
    task = Task(metrics, datamodule=PrunaDataModule.from_string("LAION256"))
    eval_agent = EvaluationAgent(task)
    eval_agent.evaluate(optimized_model)

    # Run inference
    optimized_model.set_progress_bar_config(disable=True)
    optimized_model("A serene landscape with mountains").images[0].save("output.png")

Step-by-Step Optimisation Workflow
----------------------------------

Step 1: Load a pretrained model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, load any model using its original library, like ``transformers`` or ``diffusers``:

.. code-block:: python

    from diffusers import StableDiffusionPipeline

    base_model = StableDiffusionPipeline.from_pretrained("segmind/tiny-sd")

Step 2: Define optimizations with a ``SmashConfig``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After loading the model, we can define a ``SmashConfig`` to customize the optimizations we want to apply.
This ``SmashConfig`` is a dictionary-like object that configures which optimizations to apply to your model.
You can specify multiple optimization algorithms from different categories like batching, caching and quantization.

For now, let's just use a ``quantizer`` to accelerate the model during inference.

.. code-block:: python

    from pruna import SmashConfig

    smash_config = SmashConfig()
    smash_config["quantizer"] = "hqq_diffusers"  # Accelerate the model with caching

Pruna support a wide range of algorithms for specific optimizations, all with different trade-offs.
To understand how to configure the right one for your scenario, see :doc:`Define a SmashConfig </docs_pruna/user_manual/configure>`.

Step 3: Apply optimizations with ``smash``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``smash()`` function is the core of Pruna. It takes your model and ``SmashConfig``, applies the specified optimizations.
Let's use the ``smash()`` function to apply the configured optimizations:

.. code-block:: python

    from pruna import SmashConfig, smash

    from diffusers import StableDiffusionPipeline

    # Load the model
    base_model = StableDiffusionPipeline.from_pretrained("segmind/tiny-sd")

    # Create and configure SmashConfig
    smash_config = SmashConfig()
    smash_config["quantizer"] = "hqq_diffusers"

    # Smash the model
    optimized_model = smash(model=base_model, smash_config=smash_config)

    # Save the optimized model
    optimized_model.save_to_hub("PrunaAI/segmind-tiny-sd-smashed")

The ``smash()`` function returns a ``PrunaModel`` - a wrapper that provides a standardized interface for the optimized model. So, we can still use the model as we would use the original one.

Step 4: Evaluate the optimized model with the ``EvaluationAgent``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To evaluate the optimized model, we can use the same interface as the original model.

.. code-block:: python

    from pruna.data.pruna_datamodule import PrunaDataModule
    from pruna.evaluation.evaluation_agent import EvaluationAgent

    # Load the optimized model
    optimized_model = PrunaModel.from_pretrained("PrunaAI/segmind-tiny-sd-smashed")

    # Define metrics
    metrics = ['clip_score', 'psnr']

    # Define task
    task = Task(metrics, datamodule=PrunaDataModule.from_string('LAION256'))

    # Evaluate the model
    eval_agent = EvaluationAgent(task)
    results = eval_agent.evaluate(optimized_model)
    print(results)

To understand how to run more complex evaluation workflows, see :doc:`Evaluate a model </docs_pruna/user_manual/evaluate>`.

Step 5: Run inference with the optimized model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run inference with the optimized model, we can use the same interface as the original model.

.. code-block:: python

    from pruna import PrunaModel

    # Load the optimized model
    optimized_model = PrunaModel.from_hub("PrunaAI/segmind-tiny-sd-smashed")

    optimized_model.set_progress_bar_config(disable=True)

    prompt = "A serene landscape with mountains"
    optimized_model(prompt).images[0].save("output.png")

Example use cases
-----------------

Let's look at some specific examples for different model types.

Example 1: Diffusion Model Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from diffusers import StableDiffusionPipeline

    from pruna import SmashConfig, smash

    # Load the model
    model = StableDiffusionPipeline.from_pretrained("segmind/tiny-sd")

    # Create and configure SmashConfig
    smash_config = SmashConfig()
    smash_config["compiler"] = "torch_compile"
    smash_config["quantizer"] = "hqq_diffusers"

    # Optimize the model
    optimized_model = smash(model=model, smash_config=smash_config)

    # Generate an image
    prompt = "A serene landscape with mountains"
    optimized_model(prompt).images[0].save("output.png")


Example 2: Large Language Model Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from transformers import pipeline

    from pruna import SmashConfig, smash

    # Load the model
    model_id = "meta-llama/Llama-3.2-1b-Instruct"
    pipe = pipeline("text-generation", model=model_id)

    # Create and configure SmashConfig
    smash_config = SmashConfig()
    smash_config["compiler"] = "torch_compile"
    smash_config["quantizer"] = "hqq"

    # Optimize the model
    optimized_model = smash(model=pipe.model, smash_config=smash_config)

    # Use the model for generation
    pipe("The best way to learn programming is", max_new_tokens=100)

Example 3: Speech Recognition Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import requests
    import torch
    from transformers import AutoModelForSpeechSeq2Seq

    from pruna import SmashConfig, smash

    # Load the model
    model_id = "openai/whisper-tiny"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda")

    # Create and configure SmashConfig
    smash_config = SmashConfig()
    smash_config.add_processor(model_id)  # Required for Whisper
    smash_config["compiler"] = "c_whisper"
    smash_config["batcher"] = "whisper_s2t"

    # Optimize the model
    optimized_model = smash(model=model, smash_config=smash_config)

    # Download and transcribe audio sample
    audio_url = "https://huggingface.co/datasets/reach-vb/random-audios/resolve/main/sam_altman_lex_podcast_367.flac"
    audio_file = "sam_altman_lex_podcast_367.flac"

    # Download audio file
    response = requests.get(audio_url)
    response.raise_for_status()  # Raise exception for bad status codes

    # Save audio file
    with open(audio_file, "wb") as f:
        f.write(response.content)

    # Transcribe audio
    transcription = optimized_model(audio_file)

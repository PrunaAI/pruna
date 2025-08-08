Save and Load Models
=====================

This guide provides a quick introduction to saving and loading optimized AI models with |pruna|.

You will learn how to save and load a ``PrunaModel`` after smashing a model using |pruna|.

Haven't smashed a model yet? Check out the :doc:`optimize guide </docs_pruna/user_manual/smash>` to learn how to do that.

Basic Save and Load Workflow
----------------------------

|pruna| follows a simple workflow for saving and loading optimized models:

.. mermaid::
    :align: center

    flowchart TB
        subgraph LoadFlow["Load Flow"]
            direction LR
            F["Model Files"] --> G{"Load Method"}
            G --> H1["from_pretrained('saved_model/')"]
            H1 --> I["Pruna Model"]
            H2 --> I
        end

        subgraph Model["Model Files"]
            direction TB
            E1["Model Weights (.safetensors)"]
            E2["Architecture (.json)"]
            E3["Smash Config (.json)"]
            E4["Tokenizer/Processor (original directory)"]
        end

        subgraph SaveFlow["Save Flow"]
            direction LR
            A["PrunaModel"] --> B{"Save Method"}
            B --> C1["save_pretrained('saved_model/')"]
            B --> C2["push_to_hub('PrunaAI/saved_model')"]
            C1 --> D["Model Files"]
            C2 --> D
        end

        SaveFlow --- Model
        Model --- LoadFlow

        style A fill:#bbf,stroke:#333,stroke-width:2px
        style F fill:#f9f,stroke:#333,stroke-width:2px
        style G fill:#bbf,stroke:#333,stroke-width:2px
        style H1 fill:#bbf,stroke:#333,stroke-width:2px
        style H2 fill:#bbf,stroke:#333,stroke-width:2px
        style I fill:#bbf,stroke:#333,stroke-width:2px
        style B fill:#bbf,stroke:#333,stroke-width:2px
        style C1 fill:#bbf,stroke:#333,stroke-width:2px
        style C2 fill:#bbf,stroke:#333,stroke-width:2px
        style D fill:#f9f,stroke:#333,stroke-width:2px

Let's see what that looks like in code.

.. code-block:: python

    from diffusers import DiffusionPipeline

    from pruna import PrunaModel, SmashConfig, smash

    # prepare the base model
    base_model = DiffusionPipeline.from_pretrained("segmind/Segmind-Vega")

    # Create and smash your model
    smash_config = SmashConfig()
    smash_config["compiler"] = "torch_compile"
    smash_config["quantizer"] = "hqq_diffusers"
    smashed_model = smash(model=base_model, smash_config=smash_config)

    # Save the model
    smashed_model.save_pretrained("saved_model/")  # or push_to_hub

    # Load the model
    loaded_model = PrunaModel.from_pretrained("saved_model/")

Saving a ``PrunaModel``
-----------------------

To save a smashed model, use the ``PrunaModel.save_pretrained()`` or ``PrunaModel.push_to_hub()`` method. This method saves all necessary model files and as well as the smash configuration to the specified directory:

.. tabs::

    .. tab:: Local Saving

        .. code-block:: python

            from pruna import smash, SmashConfig
            from diffusers import DiffusionPipeline

            # prepare the base model
            base_model = DiffusionPipeline.from_pretrained("segmind/Segmind-Vega")

            # Create and smash your model
            smash_config = SmashConfig()
            smash_config["quantizer"] = "hqq_diffusers"
            smashed_model = smash(model=base_model, smash_config=smash_config)

            # Save the model
            smashed_model.save_pretrained("saved_model")

    .. tab:: Hugging Face Hub Saving

        .. code-block:: python

            from pruna import smash, SmashConfig
            from diffusers import DiffusionPipeline

            # prepare the base model
            base_model = DiffusionPipeline.from_pretrained("segmind/Segmind-Vega")

            # Create and smash your model
            smash_config = SmashConfig()
            smash_config["quantizer"] = "hqq_diffusers"
            smashed_model = smash(model=base_model, smash_config=smash_config)

            # Save the model
            smashed_model.push_to_hub("PrunaAI/Segmind-Vega-smashed")

        .. tip::

            When saving models to the hub, we recommend to use a suffix like ``-smashed`` to indicate that the model has been smashed with |pruna|.

The save operation will:

1. Save the model weights and architecture, including information on how to load the model later on
2. Save the ``smash_config`` (including tokenizer and processor if present, data will be detached and not reloaded)

Loading a ``PrunaModel``
------------------------

To load a previously saved ``PrunaModel``, use the ``PrunaModel.from_pretrained()`` to load it from a local directory or from the Hugging Face Hub:

.. tabs::

    .. tab:: Local Loading

        .. code-block:: python
            :class: noextract

            from pruna import PrunaModel

            loaded_model = PrunaModel.from_pretrained("saved_model/")

    .. tab:: Hugging Face Hub Loading

        .. code-block:: python

            from pruna import PrunaModel

            loaded_model = PrunaModel.from_pretrained("PrunaAI/Segmind-Vega-smashed")

The load operation will:

1. Load the model architecture and weights and cast them to the device specified in the SmashConfig
2. Restore the smash configuration

Special Considerations
----------------------

Loading Keyword Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~
We generally recommend to load the smashed model in the same configuration as the base model, **in particular** if the two should be compared in terms of efficiency and quality.
So, when the base model was loaded with e.g. a specific precision:

.. code-block:: python

    import torch
    from diffusers import DiffusionPipeline

    base_model = DiffusionPipeline.from_pretrained("segmind/Segmind-Vega", torch_dtype=torch.float16)

you should also load the smashed model as follows:

.. code-block:: python

    from pruna import PrunaModel

    loaded_model = PrunaModel.from_pretrained("PrunaAI/Segmind-Vega-smashed", torch_dtype=torch.float16)

Depending on the saving function of the algorithm combination not all keyword arguments are required for loading (e.g. some are set by the algorithm combination itself).
In that case, we discard and log a warning about unused keyword arguments.

Algorithm Reapplication
~~~~~~~~~~~~~~~~~~~~~~~~
Some algorithms, particularly compilers and certain quantization methods, need to be reapplied after loading, as, for example, a compiled model can be rarely saved in its compiled state.
This happens automatically during the loading process based on the saved configuration and does not add a significant time overhead.

Warning Suppression
~~~~~~~~~~~~~~~~~~~~~
Set ``verbose=True`` when loading if you want to see warning messages as well as logs (in particular about reapplication of algorithms) that are by default suppressed:

.. code-block:: python

    from pruna import PrunaModel

    loaded_model = PrunaModel.from_pretrained("saved_model/", verbose=True)
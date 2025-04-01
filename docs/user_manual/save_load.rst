Saving and Loading Pruna Models
===============================

After smashing a model using |pruna|, you can save it to disk and load it later using the built-in save and load functionality.

Saving and Loading Models
-------------------------

To save a smashed model, use the ``PrunaModel.save_pretrained()`` or ``PrunaModel.save_to_hub()`` method. This method saves all necessary model files and as well as the smash configuration to the specified directory:

.. tabs::

    .. tab:: Local Saving

        .. code-block:: python

            from pruna import smash, SmashConfig
            from diffusers import StableDiffusionPipeline

            # prepare the base model
            base_model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

            # Create and smash your model
            smash_config = SmashConfig()
            smash_config["cacher"] = "deepcache"
            smash_config["compiler"] = "diffusers2"
            smashed_model = smash(model=base_model, smash_config=smash_config)

            # Save the model
            smashed_model.save_pretrained("saved_model/")

    .. tab:: Hugging Face Hub Saving

        .. code-block:: python

            from pruna import smash, SmashConfig
            from diffusers import StableDiffusionPipeline

            # prepare the base model
            base_model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

            # Create and smash your model
            smash_config = SmashConfig()
            smash_config["cacher"] = "deepcache"
            smash_config["compiler"] = "diffusers2"
            smashed_model = smash(model=base_model, smash_config=smash_config)

            # Save the model
            smashed_model.save_to_hub("PrunaAI/smashed-stable-diffusion-v1-4")

The save operation will:

1. Save the model weights and architecture, including information on how to load the model later on
2. Save the ``smash_config`` (including tokenizer and processor if present, data will be detached and not reloaded)

To load a previously saved ``PrunaModel``, use the ``PrunaModel.from_pretrained()`` or ``PrunaModel.from_hub()`` class method:

.. tabs::

    .. tab:: Local Loading

        .. code-block:: python

            from pruna import PrunaModel

            loaded_model = PrunaModel.from_pretrained("saved_model/")

    .. tab:: Hugging Face Hub Loading

        .. code-block:: python

            from pruna import PrunaModel

            loaded_model = PrunaModel.from_hub("PrunaAI/smashed-stable-diffusion-v1-4")

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
    from diffusers import StableDiffusionPipeline

    base_model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)

you should also load the smashed model as follows:

.. code-block:: python

    from pruna import PrunaModel

    loaded_model = PrunaModel.from_pretrained("saved_model/", torch_dtype=torch.float16)

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

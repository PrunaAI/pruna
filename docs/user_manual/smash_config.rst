SmashConfig
=========================

``SmashConfig`` is an essential tool in |pruna| for configuring parameters to optimize your models. This manual explains how to define and use a ``SmashConfig``.

Defining a simple ``SmashConfig``
---------------------------------

Define a ``SmashConfig`` using the following snippet:

.. code-block:: python

    from pruna import SmashConfig
    smash_config = SmashConfig()

After creating an empty ``SmashConfig``, you can set activate a algorithm by adding it to the ``SmashConfig``:

.. code-block:: python

    smash_config['quantizer'] = 'hqq'

Additionally, you can overwrite :doc:`the defaults of the algorithm </compression>` you have added by setting the hyperparameters in the ``SmashConfig``:

.. code-block:: python

    smash_config['hqq_weight_bits'] = 4

You're done! You created your ``SmashConfig`` and can now :doc:`pass it to the smash function. <smash>`


Adding a Dataset, Tokenizer or Processor
----------------------------------------

Some algorithms require a dataset, tokenizer or processor to be passed to the ``SmashConfig``. 
For example, the ``gptq`` quantizer requires a dataset and a tokenizer. We can pass them to the ``SmashConfig`` e.g. as follows:

.. code-block:: python

    from pruna import SmashConfig
    smash_config = SmashConfig()
    smash_config.add_tokenizer("facebook/opt-125m")
    smash_config.add_data("WikiText")

As you can see in this example, we can add a dataset simply by passing the name of the dataset. However, the ``add_data()`` function also supports other input formats. For more information, see the :doc:`dataset documentation <dataset>`.

We can now activate the ``gptq`` quantizer by adding it to the ``SmashConfig``:

.. code-block:: python

    smash_config['quantizers'] = 'gptq'

Similarly, we can add a processor to the ``SmashConfig`` if required, like for example by the ``c_whisper`` compiler:

.. code-block:: python

    from pruna import SmashConfig
    smash_config = SmashConfig()
    smash_config.add_processor("openai/whisper-large-v3")
    smash_config['compiler'] = 'c_whisper'

If you try to activate a algorithm that requires a dataset, tokenizer or processor and haven't added them to the ``SmashConfig``, you will receive an error. Make sure to add them before activating the algorithm! If you want to know which algorithms require a dataset, tokenizer or processor, you can look at :doc:`the compression algorithm overview </compression>`.

``SmashConfig`` Documentation
---------------------------------------------

.. autoclass:: pruna.config.smash_config.SmashConfig
   :members: add_data, add_tokenizer, add_processor, save_to_json, load_from_json, flush_configuration, load_dict

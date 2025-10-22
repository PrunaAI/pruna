:title: Power User Guide - Advanced Features in the Pruna package
:description: Learn how to use advanced features in the Pruna package to optimize your models even further.

Power User Guide
================

This guide provides an introduction to advanced features in the Pruna package to optimize your models even further. We will cover how to use "Target Modules" to tailor the optimization to your model's architecture and how to overwrite checks and the induced optimization order when smashing.

.. _target_modules:
.. this page presents more advanced features and is not directly presented in the user manual
.. but is referenced by algorithms that support the target_modules parameter.

Selective Smashing with Target Modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some algorithms let you target specific modules in your model via the ``target_modules`` parameter.
It specifies exactly which parts of the model the algorithm should operate on.

.. code-block:: python

    TARGET_MODULES_TYPE = Dict[Literal["include", "exclude"], List[str]]

The parameter is a dictionary with two keys, ``include`` and ``exclude``, each mapping to a list of pattern strings to match module paths.

A module is targeted if its path in the model matches at least one ``include`` pattern and does not match any ``exclude`` pattern.

Check out `this tutorial notebook <../tutorials/target_modules_quanto.ipynb>`_ to learn more about how to use ``target_modules``.

Pattern Format
--------------

Each of the ``include`` and ``exclude`` lists contains glob patterns, allowing you to match module paths like you would in a file search:

* ``*`` to match any number of characters (e.g., ``attention.*`` matches ``attention.to_q``, ``attention.to_k``, etc.)
* ``?`` to match exactly one character
* ``[abc]`` to match any single character from the set (e.g., ``to_[qk]`` matches ``to_q`` and ``to_k``)

Default Values
--------------

If ``target_modules`` is not provided (i.e., ``None``), default values are inferred automatically from the model, configuration and algorithm used.

If a ``target_modules`` dictionary is provided but missing either the ``include`` or ``exclude`` key:

* Missing ``include``: defaults to ``["*"]`` (considering all modules)
* Missing ``exclude``: defaults to ``[]`` (excluding no modules)

Usage Example ``target_modules``
---------------------------------

The following example shows how to use ``target_modules`` with the ``quanto`` quantizer to target your model's transformer, excluding the embedding layers.

.. code-block:: python

    from pruna import SmashConfig

    smash_config = SmashConfig({"quanto": {"target_modules": {
        "include": ["transformer.*"],
        "exclude": ["*embed*"]
    }}})

Previewing the Targeted Modules
-------------------------------

You can preview the targeted modules by using the ``expand_list_of_targeted_paths`` function as shown in the example below:

.. code-block:: python

    from transformers import AutoModelForCausalLM
    from pruna.config.target_modules import expand_list_of_targeted_paths

    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    target_modules = {
        "include": ["model.layers.[01].*attn.*"],
        "exclude": ["*v_proj"]
    }
    print(expand_list_of_targeted_paths(target_modules, model))

This will return the list of module paths that match the ``include`` and ``exclude`` patterns.
In this example, the output contains the first two attention modules (``model.layers.0.self_attn`` and ``model.layers.1.self_attn``) and the
``q_proj``, ``k_proj`` and ``o_proj`` layers inside them.

Note that this will list *all* modules that match the patterns, although some algorithms may only apply to the linear layers among those.


Circumventing Pre-Smash Checks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``experimental=True`` flag allows you to bypass certain safety checks before smashing your model:

1. **Algorithm Cross-Compatibility**: Checks whether active algorithms are compatible with each other
2. **Model Compatibility**: Validates that the model type and device are compatible with the selected algorithms
3. **Argument Compatibility**: Verifies that required arguments (tokenizer, processor, dataset) are provided for algorithms that need them

.. note::
   Even with ``experimental=True``, basic safety checks are still performed:
   - Algorithm availability checks
   - Device consistency checks
   - Algorithm package availability checks

.. code-block:: python

    from pruna import SmashConfig
    from pruna.smash import smash

    smash_config = SmashConfig(["quanto"])

    smashed_model = smash(
        model=model,
        smash_config=smash_config,
        experimental=True
    )

.. warning::
   Setting ``experimental=True`` can lead to:
   - Undefined behavior
   - Difficult-to-debug errors
   - Incompatible algorithm combinations
   - Model instability
   
   Only use this flag when you understand the implications and are prepared to troubleshoot any issues that may arise.


Overwriting the Induced Optimization Order
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some algorithms induce an optimization order on the model. This order is used to determine the order in which the algorithms are applied to the model.

When using multiple algorithms together, Pruna automatically determines the optimal order based on the dependencies and compatibility requirements of each algorithm. This is done through a topological sort of the algorithm dependency graph.

However, in some cases you may want to override this automatic ordering and specify a custom order for your algorithms.

The ``overwrite_algorithm_order()`` method allows you to manually specify the order in which algorithms should be applied.

.. code-block:: python

    from pruna import SmashConfig
    from pruna.smash import smash

    smash_config = SmashConfig(["torchao", "torch_structured", "torch_compile"])

    # Overwrite the automatic algorithm order
    smash_config.overwrite_algorithm_order([
        "torch_structured",  # Apply pruning first
        "torchao",           # Then quantization
        "torch_compile"      # Finally compilation
    ])

    smashed_model = smash(model=model, smash_config=smash_config)

Requirements
------------

When overwriting the algorithm order, you must:

* **Include all active algorithms**: The order you provide must contain exactly the same algorithms that are active in your configuration
* **Respect dependencies**: You may need to set ``experimental=True`` if your custom configuration and order violate algorithm compatibility requirements

.. warning::
   Changing the default algorithm order can lead to unexpected behavior or errors if the algorithms are applied in an incompatible sequence.

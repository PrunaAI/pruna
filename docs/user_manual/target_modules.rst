.. _target_modules:
.. this page presents more advanced features and is not directly presented in the user manual
.. but is referenced by algorithms that support the target_modules parameter.

Target Modules
==============

Some algorithms let you target specific modules in your model via the ``target_modules`` parameter.
It specifies exactly which parts of the model the algorithm should operate on.

.. code-block:: python

    TARGET_MODULES_TYPE = Dict[Literal["include", "exclude"], List[str]]

The parameter is a dictionary with two keys, ``include`` and ``exclude``, each mapping to a list of pattern strings to match module paths.

A module is targeted if its path in the model matches at least one ``include`` pattern and does not match any ``exclude`` pattern.

Check out :link:`this tutorial notebook </docs_pruna/tutorials/target_modules_quanto.ipynb>` to learn more about how to use ``target_modules``.

Pattern Format
--------------

The ``include`` and ``exclude`` lists each contain glob patterns, allowing you to match module paths like you would in a file search:

* ``*`` to match any number of characters (e.g., ``attention.*`` matches ``attention.to_q``, ``attention.to_k``, etc.)
* ``?`` to match exactly one character
* ``[abc]`` to match any single character from the set (e.g., ``to_[qk]`` matches ``to_q`` and ``to_k``)

Default Values
--------------

If ``target_modules`` is not provided (i.e., ``None``), default values are inferred automatically from the model, configuration and algorithm used.

If a ``target_modules`` is provided but does not specify a value for ``include`` or ``exclude``, the default values are ``["*"]`` for ``include`` (considering all modules) and ``[]`` for ``exclude`` (excluding no modules).

Usage examples ``target_modules``
---------------------------------

The following example shows how to use ``target_modules`` with the ``quanto`` quantizer to target your model's transformer, excluding the embedding layers.

.. code-block:: python

    from pruna_pro import SmashConfig
    smash_config = SmashConfig()
    smash_config["quantizer"] = "quanto"
    smash_config["quanto_target_modules"] = {
        "include": ["transformer.*"],
        "exclude": ["*embed*"]
    }

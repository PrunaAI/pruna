.. _target_modules:
.. this page presents more advanced features and is not directly presented in the user manual
.. but is referenced by algorithms that support the target_modules parameter.

Target Modules
==============

Some algorithms let you target specific modules in your model via the ``target_modules`` parameter.
It specifies exactly which parts of the model the algorithm should operate on.

The parameter is a dictionary with two keys, ``include`` and ``exclude``, each mapping to a list of Unix shell-style wildcards (fnmatch-style) patterns applied to module paths.
Glob patterns follow Unix-style matching (e.g., ``*`` matches any sequence of characters, ``?`` matches a single character).

.. code-block:: python

    TARGET_MODULES_TYPE = Dict[Literal["include", "exclude"], List[str]]

A module is targeted if its path in the model matches at least one ``include`` pattern and does not match any ``exclude`` pattern.

If ``target_modules`` is not provided (i.e., ``None``), default values are inferred automatically from the model, configuration and algorithm used.

Usage examples ``target_modules``
---------------------------------

The following example shows how to use ``target_modules`` with the ``quanto`` quantizer to target all modules under a fictional model's ``custom_transformer`` attribute.
The attribute name is provided in the ``include`` list as a glob that matches any module path starting with ``custom_transformer.``.

.. code-block:: python

    from pruna_pro import SmashConfig
    smash_config = SmashConfig()
    smash_config["quantizer"] = "quanto"
    smash_config["quanto_target_modules"] = {"include": ["custom_transformer.*"]}

Optionally, you can provide one or more additional patterns in the ``exclude`` list, as in the example below, which excludes normalization and embedding layers:

.. code-block:: python

    from pruna_pro import SmashConfig
    smash_config = SmashConfig()
    smash_config["quantizer"] = "quanto"
    smash_config["quanto_target_modules"] = {
        "include": ["custom_transformer.*"],
        "exclude": ["*norm*"]
    }

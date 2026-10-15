:title: MCP Server - Pruna AI Framework Documentation
:description: Serve the Pruna algorithm catalogue to an LLM over the Model Context Protocol, with model-free configuration validation.

MCP Server
==========

|pruna| ships a `Model Context Protocol <https://modelcontextprotocol.io>`_ server that lets an LLM
discover optimization algorithms, inspect their hyperparameters and validate configurations without
writing code by hand.

The server has two tiers of tools.

**Discovery and validation** tools are read-only. They need no GPU, load no model and answer
immediately, so they are always registered. They cover the whole algorithm catalogue, including the
hyperparameters each algorithm accepts and the algorithms it may be combined with.

**Execution** tools load a model and optimize it. They are opt-in and are not registered unless the
server is started with ``--allow-execute``.

Installation
------------

The server lives in the ``pruna[mcp]`` extra:

.. code-block:: bash
    :class: noextract

    pip install "pruna[mcp]"

This registers the ``pruna-mcp`` console script.

Run the Server
--------------

Start it over stdio, which is what Claude Desktop and Claude Code use:

.. code-block:: bash
    :class: noextract

    pruna-mcp

To additionally expose the tools that load and optimize models:

.. code-block:: bash
    :class: noextract

    pruna-mcp --allow-execute

Because execution tools download weights and need a device, they are disabled by default. Leaving
them off keeps the server cheap to run and safe to point at from any client.

Connect a Client
----------------

.. code-block:: json
    :class: noextract

    {
      "mcpServers": {
        "pruna": {
          "command": "pruna-mcp"
        }
      }
    }

Add ``"args": ["--allow-execute"]`` to the same entry to enable the execution tools.

Available Tools
---------------

``list_tags``
  List the algorithm categories, such as ``quantizer`` or ``cacher``, with a description of each.

``list_algorithms``
  List every optimization algorithm, optionally filtered by category. Reports the devices each
  algorithm runs on and whether it needs a tokenizer, processor or dataset.

``describe_algorithm``
  Describe one algorithm in full: every hyperparameter with its allowed values and default, the
  algorithms it must run before or after, and its references.

``validate_config``
  Check a proposed configuration **without loading a model**. Reports missing pip extras, missing
  tokenizers or datasets, and algorithm pairs that cannot be combined.

``list_benchmarks``, ``list_metrics`` and ``list_datasets``
  List the available evaluation benchmarks, metrics and datasets. ``list_datasets`` returns the
  exact names accepted by the ``data`` argument of ``evaluate_model``.

``smash_model`` and ``evaluate_model``
  Load a model and optimize or evaluate it. Registered only with ``--allow-execute``.

A Typical Session
-----------------

The intended workflow is discovery, then validation, then execution:

.. code-block:: text
    :class: noextract

    list_tags()
      -> {"tags": [{"name": "quantizer", ...}], "count": 11}

    list_algorithms(tag="quantizer")
      -> {"algorithms": [{"name": "hqq", ...}], "tag": "quantizer", "count": 12}

    describe_algorithm("hqq")
      -> {"hyperparameters": [{"qualified_name": "hqq_weight_bits",
                               "default": 8, "allowed": [2, 4, 8]}, ...], ...}

    validate_config(algorithms=["hqq", "torch_compile"],
                    hyperparameters={"hqq_weight_bits": 4})
      -> {"valid": true, "errors": [], "execution_order": ["hqq", "torch_compile"], ...}

    smash_model(model_path="...", algorithms=["hqq", "torch_compile"], save_to="./optimized")

Every tool answers with a single JSON object, so a client parses one payload rather than a list of
fragments. Hyperparameters are always keyed by their fully qualified name, prefixed with the
algorithm name, for example ``hqq_weight_bits``. Any hyperparameter that is omitted keeps its
default.

What Validation Covers
----------------------

``validate_config`` reuses the same routines ``smash`` runs, so it catches configuration mistakes
before any weights are downloaded. It reports:

* algorithms whose pip extra is not installed,
* algorithms that require a tokenizer, processor or dataset that has not been attached,
* pairs of algorithms that declare no compatibility with each other.

A valid report also carries the resolved value of every hyperparameter and the ``execution_order``
the algorithms will be applied in. The report keeps the same keys whether or not the configuration
was valid, so a rejected hyperparameter value is reported in the same shape as a missing dependency.

Two checks cannot run without a model and are reported as such in the ``notes`` field: whether the
model itself is compatible, and whether two algorithms target overlapping modules. Both are performed
by ``smash`` once the model is loaded.
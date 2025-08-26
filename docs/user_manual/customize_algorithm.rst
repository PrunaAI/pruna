:title: Custom Algorithm Development - Pruna AI Framework Extension
:description: Learn how to develop and integrate custom compression algorithms into Pruna AI. Step-by-step guide for creating new optimization methods and extending the framework.

Customize Algorithms
====================

If you’ve developed a new method or want to integrate a missing algorithm into |pruna|, we welcome your contribution! This tutorial guides you through the steps to integrate a new compression algorithm, making it available for all users.
If anything is unclear or you want to discuss your contribution before opening a PR, please reach out on `Discord <https://discord.gg/JFQmtFKCjd>`_ anytime!

.. tip::

   If you want to contribute to |pruna|, please refer to the :doc:`/docs_pruna/contributions/how_to_contribute` guide for more information. You can also `find finished PRs <https://github.com/PrunaAI/pruna/pulls?q=is%3Apr+label%3Aalgorithm+>`_ in the GitHub repository.

Add a Custom Algorithm
----------------------

We’ll use **Superfast**, an example compiler, to demonstrate the process.

Step 1. Identify the Algorithm Group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first step is to identify the algorithm group. This is important because it determines the folder in which the algorithm should be placed.
You can find the list of all algorithm groups in the :doc:`Compression Algorithms <../../compression>` section and determine which group fits your algorithm best by reviewing the algorithm group descriptions.

Step 2. Create the Compiler Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, navigate to ``pruna/algorithms/compilation/`` and create ``superfast.py``.

Step 3. Define Compiler Attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Define the new compiler by inheriting from ``PrunaCompiler`` and define key attributes for the compiler.
These attributes are used to provide information about the algorithm to the user, other functions in the package and even the documentation.

.. code-block:: python
    :class: noextract

    from typing import Any, Dict
    import torch
    from ConfigSpace import CategoricalHyperparameter
    from pruna.algorithms.compilation import PrunaCompiler
    from pruna.config.smash_config import SmashConfigPrefixWrapper

    class SuperfastCompiler(PrunaCompiler):
        """
        Implement Superfast Compiler using the superfast package.

        This compiler compiles anything with zero compilation time and 100x speedup.
        """

        algorithm_name = "superfast"
        references = {"GitHub": "/url/to/GitHub"}
        tokenizer_required = False
        processor_required = False
        dataset_required = False
        runs_on = ["cpu", "cuda"]
        compatible_algorithms = dict(quantizer=["quanto"])


Step 4. Add Algorithm Attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- docstring: The docstring should be concise and describe the algorithm in a way that is easy to understand. The description paragraph of the algorithm will be used to automatically generate the algorithm's documentation.
- ``algorithm_name``: Identifier used to activate the algorithm, name should be in snake case.
- ``references``: A dictionary of any references that can be provided for the algorithm, typically a link to the GitHub repository or a paper.
- ``tokenizer_required``, ``processor_required``, ``dataset_required``: Indicate required components.
- ``runs_on``: Define which of the hardwares listed in ``SUPPORTED_DEVICES`` are compatible with the algorithm.
- ``compatible_algorithms``: Lists compatible algorithms, i.e. any algorithm that can be applied on the same model together with the current algorithm. This compatibility should be specified both ways; if “quanto” is compatible with “superfast”, “superfast” must also list “quanto”.
- Additionally, you might have to specify a saving function. We provide more details on this in the section below.


Step 5. Define Hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Define hyperparameters using `ConfigSpace <https://automl.github.io/ConfigSpace/latest/reference/hyperparameters/>`_, allowing users to configure the backend and mode.
Everything that configures the algorithm or specifies the algorithm's behavior should be a hyperparameter.

.. code-block:: python
    :class: noextract

    def get_hyperparameters(self) -> list:
        """Return the hyperparameters for the algorithm."""
        return [
            CategoricalHyperparameter("backend", choices=["backend1", "backend2"], default_value="backend1", meta=dict(desc="The backend to use for the Superfast compiler.")),
            CategoricalHyperparameter("mode", choices=["mode1", "mode2"], default_value="mode1", meta=dict(desc="The mode to use for the Superfast compiler.")),
        ]

Users can now configure hyperparameters via ``smash_config["superfast_backend"] = "backend2"``.
Make sure to include descriptions of the hyperparameters with the ``desc`` key in the ``meta`` dictionary.
This will be used later to document the hyperparameters in the algorithm's documentation.


Step 6. Check Model Compatibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ensure the compiler only runs on supported models. In our example, the Superfast compiler is compatible with any model that is a subclass of ``torch.nn.Module``:

.. code-block:: python
    :class: noextract

    def model_check_fn(self, model: Any) -> bool:
        """Check if the model is supported by the algorithm."""
        return isinstance(model, torch.nn.Module)

Users can bypass this check using ``experimental=True`` when calling ``smash``, but results may be unpredictable.


Step 7. Handle External Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the compiler requires external packages, isolate their imports:

.. code-block:: python
    :class: noextract

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """Return algorithm packages required for execution."""
        from superfast import compile_func
        return dict(compile_func=compile_func)

Make sure that the dependencies are listed in ``pyproject.toml`` if they are not already included.

Step 8. Implement the Compilation Process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``_apply()`` function integrates superfast with Pruna:


.. code-block:: python
    :class: noextract

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """Compile the model using Superfast."""
        compile_func = self.import_algorithm_packages()["compile_func"]
        return compile_func(model, smash_config["backend"], smash_config["mode"])

Note that the ``smash_config`` prefix wrapper automatically prefixes hyperparameters with the algorithm name (``superfast_``).
If a user sets ``smash_config["superfast_backend"]``, it will be mapped correctly to ``"backend"`` in ``get_hyperparameters()``.

Step 9. Determine the Saving Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Saving e.g. a compiled or quantized model can be tricky and requires careful consideration. To determine the correct saving function for your algorithm, consider the decision tree below.

.. mermaid::

   graph TD;
       A["Is the original saving function retained?"] -->|Yes| B["save_fn = None"]
       A -->|No| C["Is the algorithm fast to apply, i.e. takes no more than 5 to 10 seconds?"]

       C -->|Yes| F["Will changes to the model be permanent (i.e. not discarded by the original saving function)?"]
       C -->|No| G["Is the saving logic complex and/or difficult to maintain?"]

       F -->|Yes| J["save_fn = SAVE_FUNCTIONS.save_before_apply"]
       F -->|No| K["save_fn = SAVE_FUNCTIONS.reapply"]

       G -->|Yes| L["SAVE_FUNCTIONS.pickled"]
       G -->|No| M["Introduce new saving function."]

The first decision is whether the original saving function can be retained.
For example, GPTQ-quantized transformers models still support ``.from_pretrained`` and ``.save_pretrained``, making retention possible.

If the original function cannot be retained, we consider how long the algorithm takes to apply.
If it is quick (e.g., a caching helper), we can reapply it after loading.
The key distinction is whether the modifications persist when saving. For instance, “step caching cacher” attaches a helper that is discarded by ``diffusers`` upon saving, so the model can be saved and reloaded normally before reapplying the function.
In contrast, compilation is irreversible—once compiled, a model cannot be saved in its compiled form, so we must save it beforehand and reapply compilation after loading.

If neither approach works, we must introduce a new saving function or use ``SAVE_FUNCTIONS.pickled``. We implement a new saving function following the existing saving-function pattern as well as introducing a matching loading function.
Otherwise, we can resort to saving the model in pickled format, but be aware that pickled models pose security risks and are generally not trusted by the community.

Step 10. Test the Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To integrate the algorithm into the test suite, we navigate to ``tests/algorithms/testers/compilation.py`` and add the following Tester Class:

.. container:: hidden_code

    .. code-block:: python

        # mock certain imports to make the code block runnable
        import sys
        import types
        from abc import ABC

        dummy_superfast = types.ModuleType("pruna.algorithms.compilation.superfast")
        dummy_superfast.SuperfastCompiler = "dummy_superfast"
        sys.modules["pruna.algorithms.compilation.superfast"] = dummy_superfast
        dummy_algorithm_tester = types.ModuleType("pruna.algorithms.testers.compilation")
        dummy_algorithm_tester.AlgorithmTesterBase = ABC
        sys.modules["base_tester"] = dummy_algorithm_tester


.. code-block:: python

    from base_tester import AlgorithmTesterBase
    from pruna.algorithms.compilation.superfast import SuperfastCompiler
    from pruna import PrunaModel

    class TestSuperfast(AlgorithmTesterBase):
        """Tester class for the Superfast algorithm."""

        models = ["stable_diffusion_v1_4"]
        reject_models = ["opt_125m"]
        allow_pickle_files = False
        algorithm_class = SuperfastCompiler

        def post_smash_hook(self, model: PrunaModel) -> None:
            assert is_compiled(model)

This Tester class specifies various aspects of the testing procedure:

- ``models``: List of models to test, should be a lightweight model to test the algorithm execution.
- ``reject_models``: List of models to reject, should be a model that is not supported by the algorithm.
- ``allow_pickle_files``: Whether to allow saving the model in pickle files, should be ``False`` for most cases but depends on the chosen saving function.
- ``algorithm_class``: The algorithm class to test.


This Tester class also includes a ``post_smash_hook`` method that can be used to perform additional checks on the model after it has been smashed, e.g. to verify that the compiler has been applied correctly. We encourage you to specify these checks wherever possible.
This Tester class automatically parametrizes an integration test at ``tests/algorithms/test_algorithms.py`` that covers smashing as well as saving and loading the model.
Additionally, a test is created to check that ``model_check_fn`` rejects a non-compatible model.
Before opening a PR, make sure to run the test suite locally to ensure the algorithm is working as expected.


Full Implementation
-------------------

Here’s the complete ``superfast.py`` implementation:

.. code-block:: python

    from typing import Any, Dict
    import torch
    from ConfigSpace import CategoricalHyperparameter
    from pruna.algorithms.compilation import PrunaCompiler
    from pruna.config.smash_config import SmashConfigPrefixWrapper

    class SuperfastCompiler(PrunaCompiler):
        """
        Implement Superfast Compiler using the superfast package.

        This compiler compiles anything with zero compilation time and 100x speedup.
        """

        algorithm_name = "superfast"
        references = {"GitHub": "/url/to/GitHub"}
        tokenizer_required = False
        processor_required = False
        dataset_required = False
        runs_on = ["cpu", "cuda"]
        compatible_algorithms = dict(quantizer=["quanto"])

        def get_hyperparameters(self) -> list:
            return [
                CategoricalHyperparameter("backend", choices=["backend1", "backend2"], default_value="backend1"),
                CategoricalHyperparameter("mode", choices=["mode1", "mode2"], default_value="mode1"),
            ]

        def model_check_fn(self, model: Any) -> bool:
            return isinstance(model, torch.nn.Module)

        def import_algorithm_packages(self) -> Dict[str, Any]:
            from superfast import compile_func
            return dict(compile_func=compile_func)

        def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
            compile_func = self.import_algorithm_packages()["compile_func"]
            return compile_func(model, smash_config["backend"], smash_config["mode"])

.. container:: hidden_code

    .. code-block:: python

        # test instantiation of compiler
        SuperfastCompiler()

Conclusion
----------

You’ve successfully integrated a new compiler into Pruna! 🚀
Now, users can utilize Superfast for model compilation, configure its hyperparameters, and ensure compatibility.

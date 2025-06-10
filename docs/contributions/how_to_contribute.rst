How to Contribute ? 💜
====================

Since you landed on this part of the documentation, we want to first of all say thank you! 💜
Contributions from the community are essential to improving |pruna|, we appreciate your effort in making the repository better for everyone!

Please make sure to review and adhere to the `Pruna Code of Conduct <https://github.com/PrunaAI/pruna/blob/main/CODE_OF_CONDUCT.md>`_ before contributing to Pruna.
Any violations will be handled accordingly and result in a ban from the Pruna community and associated platforms.
Contributions that do not adhere to the code of conduct will be ignored.

There are various ways you can contribute:

- Have a question? Discuss with us on `Discord <https://discord.gg/Tun8YgzxZ9>`_ or check out the :doc:`/resources/faq`
- Have an idea for a new tutorial? Open an issue with a :ref:`feature-request` or chat with us on `Discord <https://discord.gg/Tun8YgzxZ9>`_
- Found a bug? Open an issue with a :ref:`bug-report`
- Documentation improvements? Open an issue with a :ref:`documentation-improvement`
- Want a new feature? Open an issue with a :ref:`feature-request`
- Have a new algorithm to add? Check out: :doc:`/docs_pruna/user_manual/adding_algorithm`
- Have a new metric to add? Check out: :doc:`/docs_pruna/user_manual/adding_metric`
- Have a new dataset to add? Check out: :doc:`/docs_pruna/user_manual/adding_dataset`


.. _how-to-contribute:

Setup
-----

If you want to contribute to |pruna| with a Pull Request, you can do so by following these steps.
If it is your very first time contributing to an open source project, we recommend to start with `this guide <https://opensource.guide/how-to-contribute/>`_ for some generally helpful tips.

1. Clone the repository
^^^^^^^^^^^^^^^^^^^^^^^^

First, fork the repository by navigating to the original `pruna repository <https://github.com/PrunaAI/pruna>`_ on GitHub and click the **Fork** button at the top-right.
This creates a copy of the repository in your own GitHub account.
Then, clone the forked repository from your account to your local machine and change into its directory:

.. code-block:: bash

    git clone https://github.com/your_username/pruna.git
    cd pruna

To keep your fork up to date with the original repository, add the upstream remote:

.. code-block:: bash

    git remote add upstream https://github.com/PrunaAI/pruna.git

Always work on a new branch rather than the main branch. You can create a new branch for your feature or fix:

.. code-block:: bash

    git checkout -b feat/new-feature



2. Installation
^^^^^^^^^^^^^^^^^^^^^^

You can now install the dependencies using uv (our dependency manager) by running the following commands:

.. code-block:: bash

    uv sync --extra dev --extra tests
    uv pip install -e .

The first command creates a virtual environment in ``.venv/`` and installs all necessary dependencies including development and testing dependencies. The second command installs the pruna package itself in editable mode so your changes will be reflected immediately.

You can then also install the pre-commit hooks with

.. code-block:: bash

    pre-commit install

If you don't have uv installed, you can install it with:

.. code-block:: bash

    curl -LsSf https://astral.sh/uv/install.sh | sh

Or using pip:

.. code-block:: bash

    pip install uv


3. Develop your contribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You are now ready to work on your contribution. Check out a branch on your forked repository and start coding!
When committing your changes, we recommend to follow the `Conventional Commit Guidelines <https://www.conventionalcommits.org/en/v1.0.0/>`_.

.. code-block:: bash

    git checkout -b feat/new-feature
    git add .
    git commit -m "feat: new amazing feature setup"
    git push origin feat/new-feature

Make sure to develop your contribution in a way that is well documented, concise and easy to maintain.
We will do our best to have your contribution integrated and maintained into |pruna| but reserve the right to reject contributions that we do not feel are in the best interest of the project.

4. Run the tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We have a comprehensive test suite that is designed to catch potential issues before they are merged into |pruna|.
When you make a contribution, it is highly recommended to not only run the existing tests but also to add new tests that cover your contribution.

You can run the tests by running the following command:

.. code-block:: bash

    uv run pytest

If you want to run only the tests with a specific marker, e.g. fast CPU tests, you can do so by running:

.. code-block:: bash

    uv run pytest -m "cpu and not slow"

Note: ``uv run`` automatically activates the virtual environment managed by uv, so you don't need to manually activate it.


5. Create a Pull Request
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you have made your changes and tested them, you can create a Pull Request.
We will then review your Pull Request and get back to you as soon as possible.
If there are any questions along the way, please do not hesitate to reach out on `Discord <https://discord.gg/Tun8YgzxZ9>`_.
How to Contribute 💜
===============================

Since you landed on this part of the documentation, we want to first of all say thank you! 💜 
Bug reports are essential to improving |pruna|, and you are actively helping us identify and fix issues more efficiently. 
We appreciate your effort in making |pruna| better for everyone!

Please make sure to review and adhere to the `Pruna Code of Conduct <https://careers.pruna.ai/posts/code-of-conduct>`_ before contributing to Pruna. 
Any violations will be handled accordingly and result in a ban from the Pruna community and associated platforms. 
Contributions that do not adhere to the code of conduct will be ignored.

There are various ways you can contribute:

- Opening an issue with a :ref:`bug-report`
- Opening an issue with a :ref:`feature-request`
- Adding an algorithm to |pruna| :doc:`adding_algorithm`
- Adding a metric to |pruna| :doc:`adding_metric`
- Adding a dataset to |pruna| :doc:`adding_dataset`



.. _how-to-contribute:

Setup
-----

If you want to contribute to |pruna| with a Pull Request, you can do so by following these steps.


1. Clone the repository
^^^^^^^^^^^^^^^^^^^^^^^^

Check out the repository to your local machine and create a new branch for your contribution. 
We're sure it will be perfect from the start, but still 🚨 no working on the main branch! 🚨

.. code-block:: bash

    git clone https://github.com/your_username/pruna.git
    cd pruna
    git checkout -b feat/new-feature


2. Installation
^^^^^^^^^^^^^^^^^^^^^^

You can now set up a virtual environment of your choice and install the dependencies by running the following command:

.. code-block:: bash

    pip install -e .
    pip install -e .[dev]
    pip install -e .[tests]

You can then also install the pre-commit hooks with

.. code-block:: bash

    pre-commit install


3. Develop your contribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You are now ready to work on your contribution. Check out a branch on your forked repository and start coding!
When commiting your changes, we recommend to follow the `Conventional Commit Guidelines <https://www.conventionalcommits.org/en/v1.0.0/>`_. 

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

    pytest

If you want to run only the tests with a specific marker, e.g. fast CPU tests, you can do so by running:

.. code-block:: bash

    pytest -m "cpu and not slow"


5. Create a Pull Request
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you have made your changes and tested them, you can create a Pull Request. 
We will then review your Pull Request and get back to you as soon as possible.
If there are any questions along the way, please do not hesistate to reach out on `Discord <https://discord.gg/Tun8YgzxZ9>`_. 








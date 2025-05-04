Customize Datasets
==================

Our interface makes it easy to add :doc:`your own dataset <../user_manual/dataset>`.
Additionally, we provide a variety of :doc:`preconfigured datasets <../user_manual/dataset>` that can be readily used in SmashConfig for calibration or evaluation.

If you’d like to contribute a new dataset to our supported list, follow these two quick steps.
If anything is unclear or you want to discuss your contribution before opening a PR, please reach out on `Discord <https://discord.gg/Tun8YgzxZ9>`_ anytime!
If this is your first time contributing to |pruna|, please refer to the :ref:`how-to-contribute` guide for more information.

Add a Custom Dataset
--------------------

Step 1. Define the Dataset Setup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, create a setup method to prepare the training, validation, and test splits.
This usually involves downloading or generating the dataset.
For a text generation dataset, add the setup method in ``pruna/data/datasets/text_generation.py``:

.. code-block:: python

    from typing import Tuple
    from datasets import Dataset
    from pruna.data.utils import split_train_into_train_val_test
    from datasets import load_dataset

    def setup_new_dataset(seed: int) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Setup the new dataset.

        License: unspecified

        Parameters
        ----------
        seed : int
            The seed to use.

        Returns
        -------
        Tuple[Dataset, Dataset, Dataset]
            The dataset splits.
        """
        # Download or generate the dataset, for example:
        train_ds = load_dataset("SamuelYang/bookcorpus")["train"]
        # If necessary, split into training, validation, and test sets using the provided seed
        train_ds, val_ds, test_ds = split_train_into_train_val_test(train_ds, seed=42)
        # Adjust column names if necessary
        return train_ds, val_ds, test_ds

Next, register the dataset in ``pruna/data/__init__.py`` by adding it to the ``base_datasets`` dictionary together
with the matching collate function and any defaults (e.g. the default image size) you might want to set for the collate function:

.. container:: hidden_code

    .. code-block:: python

        # mock import, below code snippet will be added in base_datasets file itself
        from pruna.data import base_datasets

.. code-block:: python

    base_datasets["NewDataset"] = (setup_new_dataset, "text_generation_collate", {})

Ensure the dataset follows the expected format specified in the :doc:`collate function <../user_manual/dataset>`.
The collate function aggregates several samples into a batch and converts them to the expected format.

Now, users can add the dataset like this:

.. code-block:: python

    from pruna import SmashConfig

    smash_config = SmashConfig()
    smash_config.add_tokenizer("bert-base-uncased")
    smash_config.add_data("NewDataset")


.. container:: hidden_code

    .. code-block:: python

        # test if dataloader works as expected
        for batch in smash_config.test_dataloader():
            break



Step 2. Add a Test
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To verify that the dataset loads correctly, add it to ``tests/data/test_datamodule.py`` by parameterizing ``test_dm_from_string``

.. code-block:: python

    import pytest

    pytest.param("NewDataset", dict(img_size=512), marks=pytest.mark.slow)

Include necessary arguments for the collate function and mark the test as slow if needed.
We categorize a test as slow if it requires several minutes to download and prepare the dataset.
This ensures it runs appropriately in CI, either on GitHub Actions or nightly tests.

Conclusion
----------

That’s it! Your dataset is now available for everyone to use in |pruna|. 💜

Datasets
=========================

|pruna| provides a variety of pre-configured datasets for different tasks. This guide will help you understand how to use datasets in your |pruna| workflow.

Available Datasets
-------------------

|pruna| currently supports the following datasets categorized by task:

Text Generation
^^^^^^^^^^^^^^^

| ``WikiText``: Wikipedia text dataset for language modeling
| ``SmolTalk``: Everyday conversation dataset
| ``SmolSmolTalk``: Lightweight version of SmolTalk
| ``PubChem``: Chemical compound dataset in SELFIES format
| ``OpenAssistant``: Instruction-following dataset
| ``C4``: Large-scale web text dataset

Image Classification
^^^^^^^^^^^^^^^^^^^^

| ``ImageNet``: Large-scale image classification dataset
| ``MNIST``: Handwritten digit classification dataset

Text-to-Image
^^^^^^^^^^^^^^^^^^^^

| ``COCO``: Image captioning dataset
| ``LAION256``: Subset of LAION artwork dataset
| ``OpenImage``: Image quality preferences dataset

Audio Processing
^^^^^^^^^^^^^^^^^^^^

| ``CommonVoice``: Multi-language speech dataset
| ``AIPodcast``: AI-focused podcast audio dataset

Question Answering
^^^^^^^^^^^^^^^^^^^^

| ``Polyglot``: Fact completion dataset

Using Datasets
---------------

There are two main ways to use datasets in |pruna|:

1. Using String Identifier
^^^^^^^^^^^^^^^^^^^^^^^^^^^

What makes using the already implemented datasets so easy is that you can simply use the dataset's string identifier to add it to your :doc:`SmashConfig <smash_config>`:

.. code-block:: python

    from pruna import SmashConfig

    smash_config = SmashConfig()
    smash_config.add_tokenizer("bert-base-uncased")
    smash_config.add_data("WikiText")

2. Using Custom Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^

You can also pass your own datasets as a tuple of ``(train, validation, test)`` datasets:

.. code-block:: python

    from pruna import SmashConfig
    from pruna.data.utils import split_train_into_train_val_test
    from datasets import load_dataset

    # Load custom datasets
    train_ds = load_dataset("SamuelYang/bookcorpus")["train"]
    train_ds, val_ds, test_ds = split_train_into_train_val_test(train_ds, seed=42)

    # Add to SmashConfig
    smash_config = SmashConfig()
    smash_config.add_tokenizer("bert-base-uncased")
    smash_config.add_data(
        (train_ds, val_ds, test_ds),
        collate_fn="text_generation_collate"
    )

In this case, you need to specify the ``collate_fn`` to use for the dataset. The ``collate_fn`` is a function that takes a list of individual data samples and returns a batch of data in a unified format.
Your dataset will have to adhere to the formats expected by the ``collate_fn`` and this will be checked during a quick compatibility check when adding the dataset to the ``smash_config``.


.. autofunction:: pruna.data.collate.text_generation_collate
.. autofunction:: pruna.data.collate.image_generation_collate
.. autofunction:: pruna.data.collate.image_classification_collate
.. autofunction:: pruna.data.collate.audio_collate
.. autofunction:: pruna.data.collate.question_answering_collate

.. _prunadatamodule:

Accessing the PrunaDataModule directly
--------------------------------------

You can also create and access the PrunaDataModule directly and use it in your workflow, e.g., if you want to pass it to the :doc:`evaluation agent <evaluation_agent>`.

.. autoclass:: pruna.data.pruna_datamodule.PrunaDataModule
    :members: from_string, from_datasets

Define a SmashConfig
====================

This guide provides an introduction to configuring model optimization strategies with |pruna|.

Model optimization configuration relies on the ``SmashConfig`` class.
The ``SmashConfig`` class provides a flexible dictionary-like interface for configuring model optimization strategies.
It manages algorithms, hyperparameters, and additional components like tokenizers, processors and datasets.

Haven't smashed a model yet? Check out the :doc:`optimize guide </docs_pruna/user_manual/optimize>` to learn how to do that.

Basic Configuration Workflow
----------------------------

|pruna| follows a simple workflow for configuring model optimization strategies:

.. mermaid::
   :align: center

   graph LR
    User -->|creates| SmashConfig
    User -->|loads| PreTrainedModel["Pre-trained Model"]

    subgraph "Configuration Components"
        SmashConfig --- Algorithm["Algorithm Selection"]
        SmashConfig --- Hyperparameters
        SmashConfig --- Tokenizer["Tokenizer (optional)"]
        SmashConfig --- Processor["Processor (optional)"]
        SmashConfig --- Dataset["Dataset (optional)"]
    end

    SmashConfig -->|configures| SmashFn["smash() function"]
    PreTrainedModel -->|input to| SmashFn
    SmashFn -->|returns| OptimizedModel["Optimized PrunaModel"]

    style User fill:#bbf,stroke:#333,stroke-width:2px
    style PreTrainedModel fill:#bbf,stroke:#333,stroke-width:2px
    style SmashConfig fill:#bbf,stroke:#333,stroke-width:2px
    style SmashFn fill:#bbf,stroke:#333,stroke-width:2px
    style OptimizedModel fill:#bbf,stroke:#333,stroke-width:2px

Let's see what that looks like in code.

.. code-block:: python

    from pruna import SmashConfig

    smash_config = SmashConfig()

    # Activate IFW batching
    smash_config['batcher'] = 'ifw'

    # Set IFW batching parameters
    smash_config['ifw_weight_bits'] = 16
    smash_config['ifw_group_size'] = 4

    # Add a tokenizer and processor
    model_id = 'openai/whisper-large-v3'
    smash_config.add_tokenizer(model_id)
    smash_config.add_processor(model_id)

Configure Algorithms
--------------------

|pruna| implements a extensible architecture for optimization algorithms.
Each algorithm has its own impact on the model in terms of speed, memory and accuracy.
The table underneath provides a general overview of the impact of each algorithm group.

.. list-table::
   :widths: 10 60 10 10 10
   :header-rows: 1

   * - Technique
     - Description
     - Speed
     - Memory
     - Quality
   * - ``batcher``
     - Groups multiple inputs together to be processed simultaneously, improving computational efficiency and reducing processing time.
     - ✅
     - ❌
     - ➖
   * - ``cacher``
     - Stores intermediate results of computations to speed up subsequent operations.
     - ✅
     - ➖
     - ➖
   * - ``compiler``
     - Optimises the model with instructions for specific hardware.
     - ✅
     - ➖
     - ➖
   * - ``distiller``
     - Trains a smaller, simpler model to mimic a larger, more complex model.
     - ✅
     - ✅
     - ❌
   * - ``quantizer``
     - Reduces the precision of weights and activations, lowering memory requirements.
     - ✅
     - ✅
     - ❌
   * - ``pruner``
     - Removes less important or redundant connections and neurons, resulting in a sparser, more efficient network.
     - ✅
     - ✅
     - ❌
   * - ``recoverer``
     - Restores the performance of a model after compression.
     - ➖
     - ➖
     - ✅

✅(improves), ➖(approx. the same), ❌(worsens)

.. tip::

   The :doc:`Algorithm Overview </compression>` page provides a more detailed overview of each algorithm within the different groups.
   As well as additional information on the hardware requirements, compatibility with other algorithms and required components for each algorithm.

Configure Algorithm Groups
^^^^^^^^^^^^^^^^^^^^^^^^^^

To activate an algorithm, you assign its name to the corresponding algorithm group in the ``SmashConfig``.
The group names are outlined in the table above and the specific algorithms are shown in the :doc:`Algorithm Overview </compression>` page.

Let's activate the ``ifw`` algorithm as a ``batcher``:

.. code-block:: python

    from pruna import SmashConfig

    smash_config = SmashConfig()

    # Activate IFW batching
    smash_config['batcher'] = 'ifw'

Configure Algorithm Hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each algorithm has its own set of hyperparameters that control its behavior.
These are automatically prefixed with the algorithm name and can also be found underneath each algorithm in the :doc:`Algorithm Overview </compression>`.

Let's add the ``ifw_weight_bits`` and ``ifw_group_size`` hyperparameters for the ``ifw`` we defined above:

.. code-block:: python

    from pruna import SmashConfig

    smash_config = SmashConfig()

    # Activate IFW batching
    smash_config['batcher'] = 'ifw'

    # Set IFW batching parameters
    smash_config['ifw_weight_bits'] = 16
    smash_config['ifw_group_size'] = 4

Configure Components
--------------------

Some algorithms require a tokenizer, processor or dataset to be passed to the SmashConfig.
For example, looking at the :doc:`Algorithm Overview </compression>` we see that the ``gptq`` quantizer requires a dataset and a tokenizer.

.. list-table::
   :widths: 10 90 10
   :header-rows: 1

   * - Component
     - Description
     - Function
   * - ``tokenizer``
     - Tokenizes the input text.
     - ``add_tokenizer()``
   * - ``processor``
     - Processes the input data.
     - ``add_processor()``
   * - ``data``
     - Loads a dataset.
     - ``add_dataset()``

.. note::

  If you try to activate a algorithm that requires a dataset, tokenizer or processor and haven’t added them to the ``SmashConfig``, you will receive an error.
  Make sure to add them before activating the algorithm! If you want to know which algorithms require a dataset, tokenizer or processor, you can look at the :doc:`Algorithm Overview </compression>`.

Configure Tokenizers, Processors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

|pruna| provides a directly inherits from the ``transformers`` library.
This means, we can use the same tokenizers and processors as the ones used in the ``transformers`` library.

.. tabs::

   .. tab:: String Identifier

      Use a string identifier to use a tokenizer or processor from the Hugging Face Hub.

      .. code-block:: python

          from pruna import SmashConfig

          smash_config = SmashConfig()

          # Add a built-in dataset using a string identifier
          smash_config.add_tokenizer('facebook/opt-125m')
          smash_config.add_processor('openai/whisper-large-v3')

   .. tab:: Loading Directly

      Load a tokenizer or processor directly from the Hugging Face Hub with your own configuration.

      .. code-block:: python

          from pruna import SmashConfig
          from transformers import AutoTokenizer

          smash_config = SmashConfig()

          # Load a tokenizer directly from the Hugging Face Hub
          tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
          smash_config.add_tokenizer(tokenizer)

          # Load a processor directly from the Hugging Face Hub
          processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
          smash_config.add_processor(processor)

Now we've set up the tokenizer and processor, we can use them to process our data.

Configure Datasets
^^^^^^^^^^^^^^^^^^

|pruna| provides a variety of pre-configured datasets for different tasks.
We can use string identifiers to use a built-in dataset or use collate functions to use a custom dataset.
Underneath you can find the list of all the available datasets.

.. list-table::
   :header-rows: 1

   * - Task
     - Built-in Dataset ID
     - Custom Collate Function
     - Collate Function Arguments
   * - Text Generation
     - `WikiText <https://huggingface.co/datasets/Salesforce/wikitext>`_, `SmolTalk <https://huggingface.co/datasets/HuggingFaceTB/smoltalk>`_, `SmolSmolTalk <https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk>`_, `PubChem <https://huggingface.co/datasets/alxfgh/PubChem10M_SELFIES>`_, `OpenAssistant <https://huggingface.co/datasets/timdettmers/openassistant-guanaco>`_, `C4 <https://huggingface.co/datasets/allenai/c4>`_
     - ``text_generation_collate``
     - ``text: str``
   * - Image Generation
     - `LAION256 <https://huggingface.co/datasets/nannullna/laion_subset>`_, `OpenImage <https://huggingface.co/datasets/data-is-better-together/open-image-preferences-v1>`_, `COCO <https://huggingface.co/datasets/phiyodr/coco2017>`_
     - ``image_generation_collate``
     - ``image: PIL.Image.Image``, ``text: str``
   * - Image Classification
     - `ImageNet <https://huggingface.co/datasets/zh-plus/tiny-imagenet>`_, `MNIST <https://huggingface.co/datasets/ylecun/mnist>`_, `CIFAR10 <https://huggingface.co/datasets/uoft-cs/cifar10>`_
     - ``image_classification_collate``
     - ``image: PIL.Image.Image``, ``label: int``
   * - Audio Processing
     - `CommonVoice <https://huggingface.co/datasets/mozilla-foundation/common_voice_1_0>`_, `AIPodcast <https://huggingface.co/datasets/reach-vb/random-audios>`_
     - ``audio_processing_collate``
     - ``audio: Optional[torch.Tensor]``, ``path: Optional[str]``, ``sentence: str``
   * - Question Answering
     - `Polyglot <https://huggingface.co/datasets/Polyglot-or-Not/Fact-Completion>`_
     - ``question_answering_collate``
     - ``question: str``, ``answer: str``

Similar to the tokenizers and processors, we can use string identifiers to use a built-in dataset or use a more custom approach, i.e. using a collate function.
Additionallly, you can create a fully custom ``PrunaDataModule`` use it in your workflow.

.. tabs::

   .. tab:: String Identifier

      Use a string identifier to use a built-in dataset as defined in the table above.

      .. code-block:: python

          from pruna import SmashConfig

          smash_config = SmashConfig()

          # Add a built-in dataset using a string identifier
          smash_config.add_dataset('WikiText')

   .. tab:: Custom Dataset

      Use a custom collate function to use a custom dataset as ``(train, val, test)`` tuples.

      In this case, you need to specify the ``collate_fn`` to use for the dataset.
      The ``collate_fn`` is a function that takes a list of individual data samples and returns a batch of data in a unified format.
      Your dataset will have to adhere to the formats expected by the ``collate_fn`` as defined in the table above.

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

   .. tab:: PrunaDataModule

      You can also create a ``PrunaDataModule`` use it in your workflow.
      This process is more flexible but also more complex and need to adhere to certain configuration limitations.
      The process for defining a ``PrunaDataModule`` is highlighted in the :doc:`Evaluation </docs_pruna/user_manual/evaluate>` page but a basic example of adding it to the ``SmashConfig`` is shown below.

      .. code-block:: python

          from pruna import SmashConfig, PrunaDataModule

          # Load PrunaDataModule
          data = PrunaDataModule(...)

          # Add to SmashConfig
          smash_config = SmashConfig()
          smash_config.add_data(data)
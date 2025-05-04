smash
=========================

The ``smash`` function is the main function in |pruna| for optimizing models. In the following sections we will show you how to use it.

Calling the ``smash`` Function
---------------------------------------------

In preparation to using ``smash``, we have to load our model and define a ``SmashConfig``. As an example, we will take a simple model by loading the ``ViT-B/16`` model from ``torchvision``.

.. code-block:: python

    import torchvision

    base_model = torchvision.models.vit_b_16(weights="ViT_B_16_Weights.DEFAULT").cuda()

Next, we will define a :doc:`SmashConfig <smash_config>` and activate the ``torch_compile`` compiler.

.. code-block:: python

    from pruna import SmashConfig
    smash_config = SmashConfig()
    smash_config['compiler'] = 'torch_compile'

We are now ready to call the ``smash`` function!

We can pass the model and the ``SmashConfig`` to the ``smash`` function as follows:

.. code-block:: python

    from pruna import smash

    smashed_model = smash(
        model=base_model,
        smash_config=smash_config,
    )

The resulting smashed model can be used in the same way as the original one.

We perform compatibility checks to ensure that the model is compatible with the algorithms that you have selected at the beginning of the ``smash`` process. If you wish to skip these checks, you can set the ``experimental`` flag to ``True``:

.. code-block:: python

    smashed_model = smash(
        model=base_model,
        smash_config=smash_config,
        experimental=True,
    )

Please note that this can lead to undefined behavior or difficult-to-debug errors.

Importantly, the returned model offers save and load functionality that allows you to save the model and load it in its smashed state, see :doc:`save_load`.

``smash`` Function Documentation
---------------------------------------------

.. autofunction:: pruna.smash.smash
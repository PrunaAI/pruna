Telemetry
=========================

The telemetry functionality in |pruna| allows you to control sending usage metrics to help the Pruna team to improve the |pruna| package. This documentation explains how to configure telemetry settings.

What we track:

- Number of function executions (smash, loading a model, saving a model, calling a model)
- the smash config
- Whether the execution was a success

We do not track any information that could help us identify who the user was.

Example::

    pruna_function_calls_total{function="test_metrics_integration.test_otlp_export_to_collector.<locals>.test_decorated_function",job="unknown_service",session_id="7bb23832-d733-4404-b43e-7eea8c0b872e",smash_config="",status="success"} 12
    pruna_function_calls_total{function="test_operation",job="unknown_service",session_id="7bb23832-d733-4404-b43e-7eea8c0b872e",smash_config="the best config",status="success"} 15
    pruna_function_calls_total{function="test_operation",job="unknown_service",session_id="7bb23832-d733-4404-b43e-7eea8c0b872e",smash_config="the worse config",status="error"} 5


Configuring Telemetry
---------------------------------------------

Telemetry can be configured in three ways - using an environment variable, for the current Python kernel session, or globally across sessions.
The environment variable takes precedence, and the exposed methods to turn telemetry off or on change the environment variable.
If the environment variable is not set, the code will set it by reading the default value in the telemetry config. This is also how we enable to change the telemetry default settings.

Using Environment Variable
^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can control telemetry by setting the ``PRUNA_METRICS_ENABLED`` environment variable:

.. code-block:: bash

    # Enable metrics
    export PRUNA_METRICS_ENABLED=1

.. code-block:: bash

    # Disable metrics
    export PRUNA_METRICS_ENABLED=0

For Current Session
^^^^^^^^^^^^^^^^^^^^

To control telemetry for your current Python kernel session:

.. code-block:: python

    from pruna.telemetry import set_telemetry_metrics

    # Enable metrics for current session
    set_telemetry_metrics(True)

    # Disable metrics for current session
    set_telemetry_metrics(False)

Global Configuration
^^^^^^^^^^^^^^^^^^^^

To set telemetry preferences that persist across sessions:

.. code-block:: python

    from pruna.telemetry import set_telemetry_metrics

    # Enable metrics globally
    set_telemetry_metrics(True, set_as_default=True)

    # Disable metrics globally
    set_telemetry_metrics(False, set_as_default=True)

Telemetry Function Documentation
---------------------------------------------

.. autofunction:: pruna.telemetry.set_telemetry_metrics
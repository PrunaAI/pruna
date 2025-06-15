.. _pruna_tutorials:

Tutorials Pruna
===============

This tutorial will guide you through the process of using |pruna| to optimize your model. Looking for |pruna_pro| tutorials? Check out the :ref:`pruna_pro_tutorials` page.

.. grid:: 1 2 2 2

   .. grid-item-card:: Compress and Evaluate Image Generation Models
      :text-align: center
      :link: ./image_generation.ipynb

      Compress with a ``hq_diffusers`` ``quantizer`` and a ``deepcache`` ``cacher``, and evaluate with ``throughput``, ``total time``, ``clip_score``.

   .. grid-item-card:: Transcribe 2 hour of audio in 2 minutes with Whisper
      :text-align: center
      :link: ./asr_tutorial.ipynb

      Speed up ASR using the ``c_whisper`` ``compilation`` and ``whisper_s2t`` ``batching``.

   .. grid-item-card:: Smash your Computer Vision model with a CPU only
      :text-align: center
      :link: ./cv_cpu.ipynb

      ``Compile`` your model with ``torch_compile`` and ``openvino`` for faster inference.

   .. grid-item-card:: Speedup and Quantize any Diffusion Model
      :text-align: center
      :link: ./diffusion_quantization_acceleration.ipynb

      Speed up ``diffusers`` with ``torch_compile`` ``compilation`` and ``hqq_diffusers`` ``quantization``.

   .. grid-item-card:: Evaluating with CMMD using EvaluationAgent
      :text-align: center
      :link: ./evaluation_agent_cmmd.ipynb

      ``Evaluate`` image generation quality with ``CMMD`` and ``EvaluationAgent``.

   .. grid-item-card:: Making your LLMs 4x smaller
      :text-align: center
      :link: ./llms.ipynb

      Speed up your LLM inference with ``gptq`` ``quantization``.

   .. grid-item-card:: x2 smaller Sana diffusers in action
      :text-align: center
      :link: ./sana_diffusers_int8.ipynb

      Optimize your ``diffusion`` model with ``hqq_diffusers`` ``quantization`` in 8 bits.

   .. grid-item-card:: Make Stable Diffusion 3x Faster with DeepCache
      :text-align: center
      :link: ./sd_deepcache.ipynb

      Optimize your ``diffusion`` model with ``deepcache`` ``caching``.


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Pruna
   :glob:

   ./*
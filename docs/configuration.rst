Configuration
=============

This guide covers the various configuration options available in FlagEvalMM.

Environment Variables
---------------------

FlagEvalMM uses several environment variables for configuration:

* ``EVALMM_CACHE_DIR``: Directory for caching (default: ``~/.cache/evalmm``)
* ``EVALMM_DATASETS_CACHE_DIR``: Directory for dataset caching
* ``EVALMM_MODELS_CACHE_DIR``: Directory for model caching
* ``EVALMM_API_KEY``: API key for OpenAI models
* ``EVALMM_BASE_URL``: Base URL for API endpoints

Model Configuration
-------------------

Different backends have specific configuration requirements:

VLLM Configuration
~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
        "model_name": "Qwen/Qwen2-VL-72B-Instruct",
        "api_key": "EMPTY",
        "output_dir": "./results/Qwen2-VL-72B-Instruct",
        "min_short_side": 28,
        "num_workers": 8,
        "backend": "vllm",
        "extra_args": "--limit-mm-per-prompt image=18 --tensor-parallel-size 4 --max-model-len 32768 --trust-remote-code --mm-processor-kwargs '{\"max_dynamic_patch\":4}'"
    }


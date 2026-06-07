import os

FLAGEVALMM_CACHE_DIR = os.getenv(
    "FLAGEVALMM_CACHE", os.path.expanduser("~/.cache/flagevalmm")
)
FLAGEVALMM_DATASETS_CACHE_DIR = os.getenv(
    "FLAGEVALMM_DATASETS_CACHE_DIR", os.path.join(FLAGEVALMM_CACHE_DIR, "datasets")
)
FLAGEVALMM_MODELS_CACHE_DIR = os.getenv(
    "FLAGEVALMM_MODELS_CACHE_DIR", os.path.join(FLAGEVALMM_CACHE_DIR, "models")
)

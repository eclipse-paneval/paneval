import os as _os


def _get_env(new_key, *old_keys):
    for key in (new_key,) + old_keys:
        val = _os.environ.get(key)
        if val is not None:
            return val
    return ""


EVALMM_CACHE_DIR = _os.getenv(
    "EVALMM_CACHE", _os.path.expanduser("~/.cache/evalmm")
)
EVALMM_DATASETS_CACHE_DIR = _os.getenv(
    "EVALMM_DATASETS_CACHE_DIR", _os.path.join(EVALMM_CACHE_DIR, "datasets")
)
EVALMM_MODELS_CACHE_DIR = _os.getenv(
    "EVALMM_MODELS_CACHE_DIR", _os.path.join(EVALMM_CACHE_DIR, "models")
)

# Environment variable fallback: new names first, old names as fallback
EVALMM_API_KEY = _get_env("EVALMM_API_KEY", "FLAGEVAL_API_KEY")
EVALMM_BASE_URL = _get_env("EVALMM_BASE_URL", "FLAGEVAL_BASE_URL")
EVALMM_URL = _get_env("EVALMM_URL", "FLAGEVAL_URL")

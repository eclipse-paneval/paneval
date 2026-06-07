import requests
import logging
from flagevalmm.common.logger import get_logger

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log,
)

logger = get_logger(__name__)


class BaseImgenApiModel:
    def __init__(self, model_name: str, height: int = 1024, width: int = 1024):
        self.model_name = model_name
        self.height = height
        self.width = width

    @retry(
        wait=wait_random_exponential(min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.INFO),
        stop=stop_after_attempt(3),
    )
    async def download_and_save(self, img_url, save_path) -> None:
        img = requests.get(img_url)
        with open(save_path, "wb") as f:
            f.write(img.content)

    async def generate(self, prompt: str):
        raise NotImplementedError

import httpx
import requests
import json
import asyncio
import os
import logging

from flagevalmm.models.base_imgen_api_model import BaseImgenApiModel
from flagevalmm.common.logger import get_logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log,
)

logger = get_logger(__name__)


class HttpImageClient(BaseImgenApiModel):
    def __init__(
        self,
        model_name: str,
        height: int = 1024,
        width: int = 1024,
        api_key: str | None = None,
        url: str | httpx.URL | None = None,
    ):
        super().__init__(model_name=model_name, height=height, width=width)
        self.api_key = api_key
        self.url = url

    @retry(
        wait=wait_random_exponential(multiplier=5, max=100),
        before_sleep=before_sleep_log(logger, logging.INFO),
        stop=stop_after_attempt(5),
    )
    async def generate(self, prompt: str):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",  # Replace with your actual API key
        }
        data = {
            "model": f"{self.model_name}",
            "prompt": prompt,
            "size": f"{self.height}x{self.width}",
        }

        response = requests.post(
            self.url, headers=headers, data=json.dumps(data), timeout=600
        ).json()
        try:
            img_url = response["data"][0]["url"]
            return {"reason": "", "img_url": img_url}
        except Exception as e:
            logger.info(e)
            reason = response.get("error", "Unknown error")
            return {"reason": reason, "img_url": None}


async def main():
    client = HttpImageClient(
        "cogview-3-plus",
        api_key=os.getenv("ZHIPU_API_KEY"),
        url="https://open.bigmodel.cn/api/paas/v4/images/generations",
    )
    prompt = "3朵黄玫瑰"
    res = await client.generate(prompt)
    if res["img_url"]:
        await client.download_and_save(res["img_url"], "test.png")
    else:
        print(f"Failed to generate image: {res['reason']}")


if __name__ == "__main__":
    asyncio.run(main())

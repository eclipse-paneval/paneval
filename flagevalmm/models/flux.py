import asyncio
import os
import requests
from typing import Any, Dict
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    retry_if_result,
    before_sleep_log,
)

from flagevalmm.common.logger import get_logger
from flagevalmm.models import BaseImgenApiModel
from flagevalmm.prompt.prompt_tools import encode_image

logger = get_logger(__name__)


class Flux(BaseImgenApiModel):
    def __init__(self, model_name: str, height: int = 1024, width: int = 1024):
        super().__init__(model_name, height, width)
        self.api_key = os.getenv("BFL_API_KEY")
        self.url = "https://api.bfl.ml/v1"
        self.model_name = model_name

    @retry(
        wait=wait_fixed(2),
        stop=stop_after_attempt(30),
        retry=retry_if_result(
            lambda res: res["status"]
            not in ["Ready", "Error", "Request Moderated", "Content Moderated"]
        ),
        before_sleep=before_sleep_log(logger, logging.INFO),
        reraise=True,
    )
    def get_result(self, task_id: str, headers: Dict):
        result = requests.get(
            f"{self.url}/get_result",
            headers=headers,
            params={
                "id": task_id,
            },
        ).json()
        return result

    async def generate(self, prompt: str, image_path: str = None, **kwargs):
        headers = {
            "accept": "application/json",
            "x-key": self.api_key,
            "Content-Type": "application/json",
        }
        data: Dict[str, Any] = {"safety_tolerance": 6}
        if "ultra" in self.model_name:
            data.update({"prompt": prompt})
            if image_path:
                data.update({"image_prompt": encode_image(image_path)})
        else:
            data.update(
                {
                    "prompt": prompt,
                    "width": self.width,
                    "height": self.height,
                }
            )
        request = requests.post(
            f"{self.url}/{self.model_name}",
            headers=headers,
            json=data,
        ).json()
        request_id = request["id"]
        result = self.get_result(request_id, headers)
        logger.info(result)
        if result["status"] != "Ready":
            return {"reason": result["status"], "img_url": None}
        return {"reason": result["status"], "img_url": result["result"]["sample"]}


async def main():
    flux = Flux("flux-pro-1.1-ultra")
    prompt = "a white flower"
    res = await flux.generate(prompt)
    print(res)
    if res["img_url"]:
        await flux.download_and_save(res["img_url"], "test.png")
    else:
        print(f"Failed to generate image: {res['reason']}")


if __name__ == "__main__":
    asyncio.run(main())

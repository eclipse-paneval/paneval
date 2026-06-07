import requests
import json
import asyncio
import os
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

logger = get_logger(__name__)


class SenseMirage(BaseImgenApiModel):
    def __init__(self, model_name: str, height: int = 1024, width: int = 1024):
        super().__init__(model_name, height, width)
        self.api_key = os.environ.get("SENSE_MIRAGE_API_KEY")
        self.url = "https://api.sensenova.cn/v1/imgen/internal/generation_tasks"

    @retry(
        wait=wait_fixed(2),
        stop=stop_after_attempt(30),
        retry=retry_if_result(lambda res: res["state"] not in ["SUCCESS", "FAILED"]),
        before_sleep=before_sleep_log(logger, logging.INFO),
        reraise=True,
    )
    def get_result(self, task_id: str, headers: dict):
        url = f"{self.url}/{task_id}"
        res = requests.get(url=url, headers=headers).json()
        return res["task"]

    async def generate(self, prompt: str):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
        }

        data = {
            "model_id": "artist-v5",
            "prompt": prompt,
            "width": self.width,
            "height": self.height,
        }
        data_str = json.dumps(data)

        res = requests.post(url=self.url, data=data_str, headers=headers).json()

        if "task_id" not in res:
            print(prompt)
            logger.error(f"Prompt is invalid. {res}")
            return {"reason": "Prompt Forbidden", "img_url": None}
        task_result = self.get_result(res["task_id"], headers)
        print(task_result)
        err_msg = task_result["result"][0]["error"]
        if task_result["state"] != "SUCCESS" or len(err_msg) > 0:
            return {"reason": err_msg, "img_url": None}
        img_url = task_result["result"][0]["raw"]
        return {"reason": task_result["result"][0]["error"], "img_url": img_url}


async def main():
    model = SenseMirage("sense_mirage")
    prompt = "a white flower"
    res = await model.generate(prompt)
    if res["img_url"]:
        await model.download_and_save(res["img_url"], "test.png")
    else:
        print(f"Failed to generate image: {res['reason']}")


if __name__ == "__main__":
    asyncio.run(main())

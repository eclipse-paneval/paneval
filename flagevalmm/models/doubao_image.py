import asyncio
import os
import base64
import logging
from io import BytesIO
from PIL import Image
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log,
)

from flagevalmm.common.logger import get_logger
from flagevalmm.models import BaseImgenApiModel

logger = get_logger(__name__)


class DoubaoImage(BaseImgenApiModel):
    def __init__(self, model_name: str, height: int = 512, width: int = 512):
        super().__init__(model_name, height, width)
        from volcengine.visual.VisualService import VisualService

        self.model_name = model_name.replace("doubao_", "")
        self.client = VisualService()
        self.client.set_ak(os.environ.get("DOUBAO_AK"))
        self.client.set_sk(os.environ.get("DOUBAO_SK"))

    @retry(
        wait=wait_random_exponential(min=2, max=10),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.INFO),
    )
    async def generate(self, prompt: str):
        data = {
            "req_key": self.model_name,
            "prompt": prompt,
        }
        try:
            resp = self.client.cv_process(data)
        except Exception as e:
            return {"reason": str(e), "img_mode": True}

        result = {"reason": resp["message"], "img_mode": True}

        if resp["code"] == 10000:
            image_data = base64.b64decode(resp["data"]["binary_data_base64"][0])
            image = Image.open(BytesIO(image_data))
            result["image"] = image
        return result


async def main():
    model = DoubaoImage("high_aes_general_v21_L")
    prompt = "three white flower in a yellow stone"
    res = await model.generate(prompt)
    if res.get("image", None):
        res["image"].save("test.png")
    else:
        print(f"Failed to generate image: {res['reason']}")


if __name__ == "__main__":
    asyncio.run(main())

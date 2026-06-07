import requests
import json
import asyncio
import os
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from flagevalmm.models import BaseImgenApiModel


class Kolors(BaseImgenApiModel):
    def __init__(self, model_name: str, height: int = 1024, width: int = 1024):
        super().__init__(model_name, height, width)
        self.api_key = os.getenv("KOLORS_API_KEY")
        self.url = "https://mmu-kolors.kuaishou.com/v2/api/kolors/textToImage"

    @retry(wait=wait_random_exponential(min=2, max=10), stop=stop_after_attempt(3))
    async def generate(self, prompt: str):
        header = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
        }

        data = {
            "prompt": prompt,
            "count": 1,
            "width": self.width,
            "height": self.height,
        }
        data_str = json.dumps(data)

        res = requests.post(url=self.url, data=data_str, headers=header)
        res_json = res.json()
        img_url = (
            res_json["images"][0]["url"] if res_json["status"] == "success" else None
        )
        if img_url is None and "超过限流值" in res_json["reason"]:
            raise Exception("Rate limit exceeded")
        return {"reason": res_json["reason"], "img_url": img_url}


async def main():
    kolors = Kolors("kolors")
    prompt = "a white flower"
    res = await kolors.generate(prompt)
    if res["img_url"]:
        await kolors.download_and_save(res["img_url"], "test.png")
    else:
        print(f"Failed to generate image: {res['reason']}")


if __name__ == "__main__":
    asyncio.run(main())

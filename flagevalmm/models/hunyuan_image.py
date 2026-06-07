import json
import os
import asyncio
import hashlib
import hmac
import logging
import time
from datetime import datetime, timezone
from http.client import HTTPSConnection
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


def sign(key, msg):
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


class HunyuanImage(BaseImgenApiModel):
    def __init__(self, model_name: str, height: int = 1024, width: int = 1024):
        super().__init__(model_name, height, width)
        self.secret_id = os.getenv("HUNYUAN_AK")
        self.secret_key = os.getenv("HUNYUAN_SK")
        self.url = "hunyuan.tencentcloudapi.com"
        self.service = "hunyuan"
        self.max_prompt_length = 100

    def generate_headers(self, payload, action):
        algorithm = "TC3-HMAC-SHA256"
        timestamp = int(time.time())
        date = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")

        # Step 1: Construct the canonical request
        http_request_method = "POST"
        canonical_uri = "/"
        canonical_querystring = ""
        ct = "application/json; charset=utf-8"
        canonical_headers = (
            f"content-type:{ct}\nhost:{self.url}\nx-tc-action:{action.lower()}\n"
        )
        signed_headers = "content-type;host;x-tc-action"
        hashed_request_payload = hashlib.sha256(payload).hexdigest()
        canonical_request = f"{http_request_method}\n{canonical_uri}\n{canonical_querystring}\n{canonical_headers}\n{signed_headers}\n{hashed_request_payload}"

        # Step 2: Construct the string to sign
        credential_scope = f"{date}/{self.service}/tc3_request"
        hashed_canonical_request = hashlib.sha256(
            canonical_request.encode("utf-8")
        ).hexdigest()
        string_to_sign = (
            f"{algorithm}\n{timestamp}\n{credential_scope}\n{hashed_canonical_request}"
        )

        # Step 3: Calculate the signature
        secret_date = sign(("TC3" + self.secret_key).encode("utf-8"), date)
        secret_service = sign(secret_date, self.service)
        secret_signing = sign(secret_service, "tc3_request")
        signature = hmac.new(
            secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        # Step 4: Construct the Authorization header
        authorization = f"{algorithm} Credential={self.secret_id}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}"

        # Step 5: Make the request
        headers = {
            "Authorization": authorization,
            "Content-Type": ct,
            "Host": self.url,
            "X-TC-Action": action,
            "X-TC-Timestamp": timestamp,
            "X-TC-Version": "2023-09-01",
            "X-TC-Region": "ap-guangzhou",
        }
        return headers

    @retry(
        wait=wait_fixed(5),
        stop=stop_after_attempt(30),
        retry=retry_if_result(lambda res: res["JobStatusCode"] not in ["4", "5"]),
        before_sleep=before_sleep_log(logger, logging.INFO),
        reraise=True,
    )
    def get_result(self, job_id: str):
        data = {"JobId": job_id}
        data_str = json.dumps(data).encode("utf-8")
        headers = self.generate_headers(data_str, action="QueryHunyuanImageJob")
        req = HTTPSConnection(self.url)
        req.request("POST", "/", headers=headers, body=data_str)
        res = json.loads(req.getresponse().read().decode())["Response"]
        return res

    async def generate(self, prompt: str):
        if len(prompt) > self.max_prompt_length:
            prompt = prompt[: self.max_prompt_length]
            logger.warning("Prompt is too long, truncated to 100 characters")
        data = {
            "Prompt": prompt,
            "LogoAdd": 0,
            "Resolution": f"{self.height}:{self.width}",
        }
        data_str = json.dumps(data).encode("utf-8")
        headers = self.generate_headers(data_str, action="SubmitHunyuanImageJob")

        req = HTTPSConnection(self.url)
        req.request("POST", "/", headers=headers, body=data_str)
        res = json.loads(req.getresponse().read().decode())
        if "JobId" not in res["Response"]:
            logger.error(f"Prompt is invalid. {res}")
            return {"reason": "Prompt Forbidden", "img_url": None}
        job_id = res["Response"]["JobId"]
        task_result = self.get_result(job_id)
        if task_result["JobErrorCode"]:
            return {"reason": task_result["JobErrorMessage"], "img_url": None}
        img_url = (
            task_result["ResultImage"][0] if task_result["ResultImage"][0] else None
        )
        return {"reason": task_result["JobStatusMsg"], "img_url": img_url}


async def main():
    model = HunyuanImage("hunyuan")
    prompt = "two white flower and a tiger"
    res = await model.generate(prompt)
    if res["img_url"]:
        await model.download_and_save(res["img_url"], "test.png")
    else:
        print(f"Failed to generate image: {res['reason']}")


if __name__ == "__main__":
    asyncio.run(main())

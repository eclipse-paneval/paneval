from typing import List, Optional

import numpy as np
import requests
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from evalmm.prompt.prompt_tools import encode_image

# Client errors that can succeed on retry; every other 4xx fails immediately.
RETRYABLE_CLIENT_STATUS = {408, 429}


class EmbeddingRequestError(RuntimeError):
    """Non-retryable rejection (e.g. 400 when the model cannot embed images)."""


class EmbeddingClient:
    """OpenAI text embeddings and vLLM image embeddings on /v1/embeddings."""

    def __init__(
        self,
        model_name: str,
        url: str,
        api_key: Optional[str] = None,
        text_batch_size: int = 32,
        request_timeout: float = 300,
        max_image_size: Optional[int] = None,
        min_short_side: Optional[int] = None,
        max_long_side: Optional[int] = None,
        dimensions: Optional[int] = None,
        truncate_prompt_tokens: Optional[int] = None,
        query_instruction: Optional[str] = None,
        document_instruction: Optional[str] = None,
    ):
        if not model_name or not url:
            raise ValueError("model_name and url are required")
        if text_batch_size < 1:
            raise ValueError("text_batch_size must be positive")
        base = url.rstrip("/")
        for suffix in ("/v1/embeddings", "/v1/chat/completions", "/v1"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        self.url = f"{base}/v1/embeddings"
        self.model_name = model_name
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.text_batch_size = text_batch_size
        self.request_timeout = request_timeout
        self.max_image_size = max_image_size
        self.min_short_side = min_short_side
        self.max_long_side = max_long_side
        self.dimensions = dimensions
        self.truncate_prompt_tokens = truncate_prompt_tokens
        self.query_instruction = query_instruction or ""
        self.document_instruction = document_instruction

    def _payload(self):
        payload = {"model": self.model_name, "encoding_format": "float"}
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        return payload

    @retry(
        wait=wait_random_exponential(min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_not_exception_type(EmbeddingRequestError),
        reraise=True,
    )
    def _request(self, payload, count: int) -> np.ndarray:
        response = requests.post(
            self.url,
            headers=self.headers,
            json=payload,
            timeout=self.request_timeout,
        )
        if response.status_code != 200:
            message = (
                f"Embedding request failed with status code {response.status_code}: "
                f"{response.text}"
            )
            if (
                400 <= response.status_code < 500
                and response.status_code not in RETRYABLE_CLIENT_STATUS
            ):
                raise EmbeddingRequestError(message)
            raise RuntimeError(message)
        data = response.json()["data"]
        if len(data) != count:
            raise ValueError(f"Expected {count} embeddings, received {len(data)}")
        if "input" in payload:
            data = sorted(data, key=lambda item: item["index"])
            if [item["index"] for item in data] != list(range(count)):
                raise ValueError("Embedding response indices do not match the inputs")
        embeddings = np.asarray([item["embedding"] for item in data], dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] == 0:
            raise ValueError("Expected nonempty embedding vectors")
        if not np.isfinite(embeddings).all():
            raise ValueError("Embedding response contains non-finite values")
        return embeddings

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        batches = []
        for start in range(0, len(texts), self.text_batch_size):
            batch = texts[start : start + self.text_batch_size]
            payload = self._payload()
            payload["input"] = [self.query_instruction + text for text in batch]
            if self.truncate_prompt_tokens is not None:
                payload["truncate_prompt_tokens"] = self.truncate_prompt_tokens
            batches.append(self._request(payload, len(batch)))
        return (
            np.concatenate(batches) if batches else np.empty((0, 0), dtype=np.float32)
        )

    def embed_images(self, paths: List[str]) -> np.ndarray:
        batches = []
        for path in paths:
            image = encode_image(
                path,
                max_size=self.max_image_size,
                min_short_side=self.min_short_side,
                max_long_side=self.max_long_side,
            )
            content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            ]
            if self.document_instruction is not None:
                content.append({"type": "text", "text": self.document_instruction})
            payload = self._payload()
            payload["messages"] = [{"role": "user", "content": content}]
            batches.append(self._request(payload, 1))
        return (
            np.concatenate(batches) if batches else np.empty((0, 0), dtype=np.float32)
        )

import base64
from io import BytesIO
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image
from tenacity import wait_none

from evalmm.models import EmbeddingClient
from evalmm.models.embedding_client import EmbeddingRequestError


@pytest.fixture(autouse=True)
def no_retry_wait(monkeypatch):
    monkeypatch.setattr(EmbeddingClient._request.retry, "wait", wait_none())


def response(vectors, indices=None):
    indices = range(len(vectors)) if indices is None else indices
    return Mock(
        status_code=200,
        json=lambda: {
            "data": [
                {"index": i, "embedding": vector} for i, vector in zip(indices, vectors)
            ]
        },
    )


@pytest.mark.parametrize(
    "suffix",
    (
        "",
        "/",
        "/v1",
        "/v1/",
        "/v1/embeddings",
        "/v1/embeddings/",
        "/v1/chat/completions",
        "/v1/chat/completions/",
    ),
)
def test_url_normalization(monkeypatch, suffix):
    post = Mock(return_value=response([[1, 2]]))
    monkeypatch.setattr("requests.post", post)
    client = EmbeddingClient("test-model", f"https://example.com/proxy{suffix}")
    client.embed_texts(["caption"])
    assert post.call_args.args == ("https://example.com/proxy/v1/embeddings",)


def test_text_batching_instructions_and_response_order(monkeypatch):
    post = Mock(side_effect=[response([[3, 4], [1, 2]], [1, 0]), response([[5, 6]])])
    monkeypatch.setattr("requests.post", post)
    client = EmbeddingClient(
        "test-model",
        "https://example.com",
        api_key="secret",
        text_batch_size=2,
        request_timeout=12,
        dimensions=2,
        truncate_prompt_tokens=77,
        query_instruction="query: ",
    )
    result = client.embed_texts(["first", "second", "third"])
    np.testing.assert_array_equal(result, [[1, 2], [3, 4], [5, 6]])
    assert result.dtype == np.float32
    assert post.call_count == 2
    assert post.call_args_list[0].kwargs == {
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer secret",
        },
        "json": {
            "model": "test-model",
            "input": ["query: first", "query: second"],
            "encoding_format": "float",
            "dimensions": 2,
            "truncate_prompt_tokens": 77,
        },
        "timeout": 12,
    }
    assert post.call_args_list[1].kwargs["json"]["input"] == ["query: third"]


def test_optional_fields_omitted(monkeypatch):
    post = Mock(return_value=response([[1, 2]]))
    monkeypatch.setattr("requests.post", post)
    EmbeddingClient("test-model", "https://example.com").embed_texts(["caption"])
    assert post.call_args.kwargs == {
        "headers": {"Content-Type": "application/json"},
        "json": {
            "model": "test-model",
            "input": ["caption"],
            "encoding_format": "float",
        },
        "timeout": 300,
    }


def test_default_text_batch_size(monkeypatch):
    def post(url, **kwargs):
        inputs = kwargs["json"]["input"]
        sizes.append(len(inputs))
        return response([[i, 1] for i in range(len(inputs))])

    sizes = []
    monkeypatch.setattr("requests.post", post)
    result = EmbeddingClient("test-model", "https://example.com").embed_texts(
        [str(i) for i in range(33)]
    )
    assert sizes == [32, 1]
    assert result.shape == (33, 2)


@pytest.mark.parametrize("instruction", [None, "Represent this document"])
def test_image_messages(monkeypatch, tmp_path, instruction):
    path = tmp_path / "image.png"
    Image.new("RGBA", (20, 10), color="red").save(path)
    post = Mock(return_value=response([[3, 4]]))
    monkeypatch.setattr("requests.post", post)
    client = EmbeddingClient(
        "test-model",
        "https://example.com",
        dimensions=2,
        document_instruction=instruction,
        max_long_side=8,
        min_short_side=6,
        truncate_prompt_tokens=77,
    )
    result = client.embed_images([str(path), str(path)])
    np.testing.assert_array_equal(result, [[3, 4], [3, 4]])
    assert result.dtype == np.float32
    assert post.call_count == 2
    payload = post.call_args.kwargs["json"]
    assert set(payload) == {"model", "encoding_format", "dimensions", "messages"}
    assert payload["model"] == "test-model"
    assert payload["encoding_format"] == "float"
    assert payload["dimensions"] == 2
    assert len(payload["messages"]) == 1
    message = payload["messages"][0]
    assert message["role"] == "user"
    content = message["content"]
    assert content[0]["type"] == "image_url"
    data_uri = content[0]["image_url"]["url"]
    prefix, encoded = data_uri.split(",", 1)
    assert prefix == "data:image/jpeg;base64"
    with Image.open(BytesIO(base64.b64decode(encoded))) as image:
        assert image.format == "JPEG"
        assert image.mode == "RGB"
        assert image.size == (8, 6)
    if instruction is None:
        assert len(content) == 1
    else:
        assert content[1:] == [{"type": "text", "text": instruction}]


def test_image_resize_options_forwarded(monkeypatch):
    encode = Mock(return_value="encoded")
    monkeypatch.setattr("evalmm.models.embedding_client.encode_image", encode)
    monkeypatch.setattr("requests.post", Mock(return_value=response([[1, 2]])))
    client = EmbeddingClient(
        "test-model",
        "https://example.com",
        max_image_size=2000,
        min_short_side=10,
        max_long_side=100,
    )
    client.embed_images(["image.jpg"])
    encode.assert_called_once_with(
        "image.jpg", max_size=2000, min_short_side=10, max_long_side=100
    )


def test_image_response_without_index(monkeypatch):
    monkeypatch.setattr(
        "evalmm.models.embedding_client.encode_image", lambda *a, **k: "encoded"
    )
    monkeypatch.setattr(
        "requests.post",
        Mock(
            return_value=Mock(
                status_code=200, json=lambda: {"data": [{"embedding": [1, 2]}]}
            )
        ),
    )
    result = EmbeddingClient("test-model", "https://example.com").embed_images(["a"])
    np.testing.assert_array_equal(result, [[1, 2]])


def test_http_failure_retries_with_status_and_body(monkeypatch):
    post = Mock(return_value=Mock(status_code=503, text="pooling unavailable"))
    monkeypatch.setattr("requests.post", post)
    with pytest.raises(RuntimeError, match="503: pooling unavailable"):
        EmbeddingClient("test-model", "https://example.com").embed_texts(["caption"])
    assert post.call_count == 3


@pytest.mark.parametrize("status", [400, 401, 404, 422])
def test_client_error_fails_without_retry(monkeypatch, status):
    post = Mock(return_value=Mock(status_code=status, text="unsupported input"))
    monkeypatch.setattr("requests.post", post)
    with pytest.raises(EmbeddingRequestError, match=f"{status}: unsupported input"):
        EmbeddingClient("test-model", "https://example.com").embed_texts(["caption"])
    assert post.call_count == 1


@pytest.mark.parametrize("status", [408, 429])
def test_retryable_client_errors_are_retried(monkeypatch, status):
    post = Mock(return_value=Mock(status_code=status, text="slow down"))
    monkeypatch.setattr("requests.post", post)
    with pytest.raises(RuntimeError, match=f"{status}: slow down") as excinfo:
        EmbeddingClient("test-model", "https://example.com").embed_texts(["caption"])
    assert not isinstance(excinfo.value, EmbeddingRequestError)
    assert post.call_count == 3


@pytest.mark.parametrize(
    "method, inputs", [("embed_texts", ["a", "b"]), ("embed_images", ["a"])]
)
def test_count_mismatch(monkeypatch, method, inputs):
    monkeypatch.setattr(
        "evalmm.models.embedding_client.encode_image", lambda *a, **k: "encoded"
    )
    post = Mock(return_value=response([]))
    monkeypatch.setattr("requests.post", post)
    with pytest.raises(ValueError, match="Expected .* embeddings, received 0"):
        getattr(EmbeddingClient("test-model", "https://example.com"), method)(inputs)
    assert post.call_count == 3


def test_duplicate_response_indices_rejected(monkeypatch):
    monkeypatch.setattr(
        "requests.post", Mock(return_value=response([[1, 2], [3, 4]], [0, 0]))
    )
    with pytest.raises(ValueError, match="indices"):
        EmbeddingClient("test-model", "https://example.com").embed_texts(["a", "b"])


def test_retry_recovers(monkeypatch):
    post = Mock(
        side_effect=[Mock(status_code=500, text="temporary"), response([[1, 2]])]
    )
    monkeypatch.setattr("requests.post", post)
    result = EmbeddingClient("test-model", "https://example.com").embed_texts(
        ["caption"]
    )
    np.testing.assert_array_equal(result, [[1, 2]])
    assert post.call_count == 2


def test_empty_inputs_do_not_request(monkeypatch):
    post = Mock()
    monkeypatch.setattr("requests.post", post)
    client = EmbeddingClient("test-model", "https://example.com")
    assert client.embed_texts([]).shape == (0, 0)
    assert client.embed_images([]).shape == (0, 0)
    post.assert_not_called()


@pytest.mark.parametrize("batch_size", [0, -1])
def test_invalid_batch_size(batch_size):
    with pytest.raises(ValueError, match="text_batch_size must be positive"):
        EmbeddingClient("test-model", "https://example.com", text_batch_size=batch_size)

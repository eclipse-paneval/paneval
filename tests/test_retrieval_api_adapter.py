"""Offline tests for embedding API retrieval, caching, and data dispatch."""

import json
import sys
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from evalmm.models import base_model_adapter
from evalmm import eval as eval_cli
from evalmm.server import evaluation_server
from model_zoo.retrieval.api_model import model_adapter


class FakeTaskManager:
    def __init__(self):
        # Three captions per image, deliberately different from COCO's five.
        self.captions = [f"caption-{i}" for i in range(6)]
        self.paths = [f"image-{i}.jpg" for i in range(2)]
        self.calls = []

    def get_retrieval_data(self, task_name, index, data_type):
        self.calls.append((task_name, index, data_type))
        if data_type == "text":
            return {"caption": self.captions[index]}
        assert data_type == "img"
        return {"img_path": self.paths[index]}


class FakeEmbeddingClient:
    text_batch_size = 2

    def __init__(self, task_manager):
        self.text_vectors = dict(
            zip(
                task_manager.captions,
                ([6, 8], [8, -6], [5, 0], [0, 9], [-3, 4], [-8, -6]),
            )
        )
        self.image_vectors = dict(zip(task_manager.paths, ([3, 4], [0, 7])))
        self.text_calls = []
        self.image_calls = []
        self.fail_text_batches = set()

    def embed_texts(self, texts):
        batch = tuple(texts)
        self.text_calls.append(batch)
        if batch in self.fail_text_batches:
            raise RuntimeError("simulated embedding failure")
        # Float64 input also checks the adapter's float32 conversion.
        return np.asarray([self.text_vectors[text] for text in texts], dtype=np.float64)

    def embed_images(self, paths):
        self.image_calls.append(tuple(paths))
        return np.asarray(
            [self.image_vectors[path] for path in paths], dtype=np.float64
        )


@pytest.fixture
def retrieval_case(tmp_path):
    manager = FakeTaskManager()
    client = FakeEmbeddingClient(manager)
    adapter = model_adapter.ModelAdapter.__new__(model_adapter.ModelAdapter)
    adapter.task_manager = manager
    adapter.model = client
    adapter.model_name = "fake-embedding-model"
    adapter.task_info = {"num_workers": 3}
    meta = {
        "name": "tiny_retrieval",
        "image_number": 2,
        "caption_number": 6,
        "output_dir": str(tmp_path),
    }
    return SimpleNamespace(
        adapter=adapter, manager=manager, client=client, meta=meta, output=tmp_path
    )


def assert_complete_result(case):
    matrix = np.load(case.output / "tiny_retrieval.npy", allow_pickle=False)
    assert matrix.shape == (2, 6)
    assert matrix.dtype == np.float32
    # Hand-computed cosines; normalizing just one modality cannot give these.
    expected = [
        [1.0, 0.0, 0.6, 0.8, 0.28, -0.96],
        [0.8, -0.6, 0.0, 1.0, 0.8, -0.6],
    ]
    np.testing.assert_allclose(matrix, expected, rtol=1e-6, atol=1e-7)
    metadata = json.loads((case.output / "tiny_retrieval.json").read_text())
    assert metadata == {
        "model_name": "fake-embedding-model",
        "embedding_dim": 2,
        "n_images": 2,
        "n_captions": 6,
    }


def assert_no_completed_result(case):
    assert not (case.output / "tiny_retrieval.npy").exists()
    assert not (case.output / "tiny_retrieval.json").exists()


def test_cosine_matrix_preserves_caption_indices_across_batches(retrieval_case):
    case = retrieval_case
    case.adapter.run_one_task("retrieval-task", case.meta)

    assert_complete_result(case)
    assert sorted(case.client.text_calls) == [
        ("caption-0", "caption-1"),
        ("caption-2", "caption-3"),
        ("caption-4", "caption-5"),
    ]
    assert sorted(case.client.image_calls) == [("image-0.jpg",), ("image-1.jpg",)]
    assert sorted(case.manager.calls) == sorted(
        [("retrieval-task", i, "text") for i in range(6)]
        + [("retrieval-task", i, "img") for i in range(2)]
    )
    assert {path.name for path in (case.output / "emb_cache").iterdir()} == {
        "text_0000000_0000002.npy",
        "text_0000002_0000004.npy",
        "text_0000004_0000006.npy",
        "image_0000000_0000001.npy",
        "image_0000001_0000002.npy",
    }


def test_out_of_order_batch_completion_keeps_caption_columns(
    retrieval_case, monkeypatch
):
    case = retrieval_case
    later_batch_done = Event()
    first_batch_started = Event()
    embed_texts = case.client.embed_texts
    completed = []

    def embed_out_of_order(texts):
        if texts[0] == "caption-0":
            first_batch_started.set()
            assert later_batch_done.wait(timeout=5), "later text batch did not run"
        elif texts[0] == "caption-2":
            assert first_batch_started.wait(timeout=5), "first text batch did not run"
        result = embed_texts(texts)
        completed.append(texts[0])
        if texts[0] == "caption-2":
            later_batch_done.set()
        return result

    monkeypatch.setattr(case.client, "embed_texts", embed_out_of_order)
    case.adapter.run_one_task("retrieval-task", case.meta)

    assert completed.index("caption-2") < completed.index("caption-0")
    assert_complete_result(case)


def test_second_run_reuses_cache_without_model_or_data_calls(
    retrieval_case, monkeypatch
):
    case = retrieval_case
    case.adapter.run_one_task("retrieval-task", case.meta)
    case.client.text_calls.clear()
    case.client.image_calls.clear()
    case.manager.calls.clear()
    # Remove completion artifacts so this also proves the cache can reconstruct them.
    (case.output / "tiny_retrieval.npy").unlink()
    (case.output / "tiny_retrieval.json").unlink()
    forbidden = Mock(side_effect=AssertionError("cached batches must not be fetched"))
    monkeypatch.setattr(case.client, "embed_texts", forbidden)
    monkeypatch.setattr(case.client, "embed_images", forbidden)
    monkeypatch.setattr(case.manager, "get_retrieval_data", forbidden)

    case.adapter.run_one_task("retrieval-task", case.meta)

    forbidden.assert_not_called()
    assert_complete_result(case)


def test_interrupted_run_resumes_only_missing_batches(retrieval_case):
    case = retrieval_case
    case.adapter.task_info["num_workers"] = 1
    failed_batch = ("caption-2", "caption-3")
    case.client.fail_text_batches.add(failed_batch)

    with pytest.raises(RuntimeError, match="simulated embedding failure"):
        case.adapter.run_one_task("retrieval-task", case.meta)

    assert_no_completed_result(case)
    cache = case.output / "emb_cache"
    assert (cache / "text_0000000_0000002.npy").exists()
    assert not (cache / "text_0000002_0000004.npy").exists()
    cached_names = {path.name for path in cache.glob("*.npy")}
    assert not list(cache.glob("*.tmp"))
    case.client.fail_text_batches.clear()
    case.client.text_calls.clear()
    case.client.image_calls.clear()
    case.manager.calls.clear()

    case.adapter.run_one_task("retrieval-task", case.meta)

    expected_text_calls = [
        tuple(case.manager.captions[start : start + 2])
        for start in range(0, 6, 2)
        if f"text_{start:07d}_{start + 2:07d}.npy" not in cached_names
    ]
    expected_image_calls = [
        (case.manager.paths[i],)
        for i in range(2)
        if f"image_{i:07d}_{i + 1:07d}.npy" not in cached_names
    ]
    assert case.client.text_calls == expected_text_calls
    assert failed_batch in case.client.text_calls
    assert case.client.image_calls == expected_image_calls
    assert ("retrieval-task", 0, "text") not in case.manager.calls
    assert_complete_result(case)


def test_failure_cancels_queued_batches(retrieval_case, monkeypatch):
    case = retrieval_case
    case.adapter.task_info["num_workers"] = 1
    case.client.fail_text_batches.add(("caption-0", "caption-1"))
    cancelled = Event()

    class RecordingPool(model_adapter.ThreadPoolExecutor):
        def shutdown(self, wait=True, *, cancel_futures=False):
            if cancel_futures:
                # Cancel the queue first, then let the in-flight batch finish.
                super().shutdown(wait=False, cancel_futures=True)
                cancelled.set()
            super().shutdown(wait=wait, cancel_futures=cancel_futures)

    embed_texts = case.client.embed_texts

    def hold_second_batch(texts):
        if texts[0] == "caption-2":
            # Fake calls are instant; without this the single worker could drain
            # the whole queue before the main thread sees the failure.
            cancelled.wait(timeout=5)
        return embed_texts(texts)

    monkeypatch.setattr(model_adapter, "ThreadPoolExecutor", RecordingPool)
    monkeypatch.setattr(case.client, "embed_texts", hold_second_batch)

    with pytest.raises(RuntimeError, match="simulated embedding failure"):
        case.adapter.run_one_task("retrieval-task", case.meta)

    assert cancelled.is_set()
    # 5 batches queued (3 text, 2 image): the failed one and the one already in
    # flight ran; the remaining 3 were cancelled.
    assert case.client.text_calls == [
        ("caption-0", "caption-1"),
        ("caption-2", "caption-3"),
    ]
    assert case.client.image_calls == []
    assert_no_completed_result(case)


def test_video_guard_precedes_all_client_and_data_access(tmp_path):
    adapter = model_adapter.ModelAdapter.__new__(model_adapter.ModelAdapter)
    adapter.model = Mock()
    adapter.task_manager = Mock()
    output = tmp_path / "video-output"

    with pytest.raises(NotImplementedError, match="Video retrieval is not supported"):
        adapter.run_one_task(
            "video-task",
            {
                "name": "tiny_video",
                "video_number": 2,
                "caption_number": 6,
                "output_dir": str(output),
            },
        )

    assert adapter.model.mock_calls == []
    assert adapter.task_manager.mock_calls == []
    assert not output.exists()


@pytest.mark.parametrize(
    ("embeddings", "message"),
    [
        ([[1, 2]], "Expected 2 embedding rows"),
        ([1, 2], "Expected 2 embedding rows"),
        ([[], []], "Expected 2 embedding rows"),
        ([[0, 0], [1, 2]], "zero embedding vectors"),
        ([[np.nan, 1], [1, 2]], "non-finite"),
        ([[np.inf, 1], [1, 2]], "non-finite"),
    ],
    ids=["row-count", "rank", "empty-dimension", "zero", "nan", "infinity"],
)
def test_invalid_batch_aborts_without_final_artifacts(
    retrieval_case, monkeypatch, embeddings, message
):
    case = retrieval_case
    monkeypatch.setattr(case.client, "embed_texts", lambda texts: embeddings)

    with pytest.raises(ValueError, match=message):
        case.adapter.run_one_task("retrieval-task", case.meta)

    assert_no_completed_result(case)
    assert not list((case.output / "emb_cache").glob("text_*.npy"))


def test_image_text_dimension_mismatch_aborts_without_final_artifacts(retrieval_case):
    case = retrieval_case
    case.client.image_vectors = {path: [1, 2, 3] for path in case.manager.paths}

    with pytest.raises(
        ValueError, match="Image and text embedding dimensions do not match"
    ):
        case.adapter.run_one_task("retrieval-task", case.meta)

    assert_no_completed_result(case)


def test_invalid_cached_vectors_are_rejected(retrieval_case):
    case = retrieval_case
    cache = case.output / "emb_cache"
    cache.mkdir()
    np.save(cache / "text_0000000_0000002.npy", np.zeros((2, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="zero embedding vectors"):
        case.adapter.run_one_task("retrieval-task", case.meta)

    assert ("caption-0", "caption-1") not in case.client.text_calls
    assert_no_completed_result(case)


@pytest.mark.parametrize("explicit_model_name", [False, True])
def test_model_init_forwards_client_configuration_only(
    monkeypatch, explicit_model_name
):
    adapter = model_adapter.ModelAdapter.__new__(model_adapter.ModelAdapter)
    adapter.model_name = "fallback-model"
    client_config = {
        "url": "https://embeddings.invalid/v1/embeddings",
        "api_key": "test-key",
        "text_batch_size": 7,
        "request_timeout": 12,
        "max_image_size": 1024,
        "min_short_side": 32,
        "max_long_side": 512,
        "dimensions": 64,
        "truncate_prompt_tokens": 77,
        "query_instruction": "query: ",
        "document_instruction": "document: ",
    }
    if explicit_model_name:
        client_config["model_name"] = "configured-model"
    constructor = Mock()
    monkeypatch.setattr(model_adapter, "EmbeddingClient", constructor)

    adapter.model_init(
        {
            **client_config,
            "task_names": ["tiny"],
            "model_path": "/unused",
            "num_workers": 3,
        }
    )

    expected = {"model_name": "fallback-model", **client_config}
    constructor.assert_called_once_with(**expected)
    assert adapter.model is constructor.return_value


@pytest.mark.parametrize("data_type,index", [("img", 1), ("text", 4)])
def test_server_local_loading_matches_flask_route(
    monkeypatch, tmp_path, data_type, index
):
    manager = FakeTaskManager()
    dataset = SimpleNamespace(
        get_data=Mock(
            side_effect=lambda index, kind: manager.get_retrieval_data(
                "tiny", index, kind
            )
        )
    )
    dataset_config = {"type": "FakeRetrievalDataset"}
    build = Mock(return_value=dataset)
    monkeypatch.setattr(evaluation_server.DATASETS, "build", build)
    config = {"tiny": SimpleNamespace(dataset=dataset_config)}
    local = evaluation_server.EvaluationServer(
        config, model_path="fake", output_dir=str(tmp_path), local_mode=True
    )
    remote = evaluation_server.EvaluationServer(
        config, model_path="fake", output_dir=str(tmp_path), local_mode=False
    )

    expected = local.get_task_retrieval_data("tiny", index, data_type)
    assert local.get_task_retrieval_data("tiny", index, data_type) == expected
    assert build.call_count == 1  # Repeated reads reuse the loaded dataset.
    response = (
        remote.get_flask_app()
        .test_client()
        .get(
            "/get_retrieval_data",
            query_string={"task": "tiny", "index": index, "type": data_type},
        )
    )

    assert response.status_code == 200
    assert response.get_json() == expected
    assert build.call_count == 2
    build.assert_called_with(dataset_config)
    dataset.get_data.assert_called_with(index, data_type)


@pytest.mark.parametrize("local_mode", [True, False])
def test_task_manager_dispatches_retrieval_data_with_remote_timeout(
    monkeypatch, local_mode
):
    manager = base_model_adapter.TaskManager.__new__(base_model_adapter.TaskManager)
    manager.local_mode = local_mode
    manager.server_ip = "http://example.invalid"
    manager.server_port = 5678
    manager.timeout = 17
    manager.evaluation_server = Mock()
    local = manager.evaluation_server.get_task_retrieval_data
    remote = Mock()
    monkeypatch.setattr(base_model_adapter, "get_retrieval_data", remote)

    result = manager.get_retrieval_data("tiny", 4, "text")

    if local_mode:
        local.assert_called_once_with("tiny", 4, "text")
        remote.assert_not_called()
        assert result is local.return_value
    else:
        remote.assert_called_once_with(
            4, "tiny", "text", "http://example.invalid", 5678, 17
        )
        local.assert_not_called()
        assert result is remote.return_value


@pytest.mark.parametrize("config_source", ["inline", "file"])
@pytest.mark.parametrize("local_mode", [True, False])
def test_main_propagates_cli_options_and_merges_config(
    monkeypatch, tmp_path, config_source, local_mode
):
    config = {
        "model_name": "config-model",
        "url": "https://config.invalid",
        "text_batch_size": 8,
        "dimensions": 64,
        "document_instruction": "document: ",
    }
    config_argument = json.dumps(config)
    if config_source == "file":
        config_path = tmp_path / "model.json"
        config_path.write_text(config_argument)
        config_argument = str(config_path)
    argv = [
        "model_adapter.py",
        "--cfg",
        config_argument,
        "--tasks",
        "tiny-task.py",
        "--model",
        "cli-model",
        "--url",
        "https://cli.invalid/v1/embeddings",
        "--api-key",
        "cli-key",
        "--text-batch-size",
        "3",
        "--request-timeout",
        "9.5",
        "--num-workers",
        "2",
        "--query-instruction",
        "query: ",
        "--output-dir",
        str(tmp_path),
        "--data-root",
        "/fake/data",
        "--server-ip",
        "http://server.invalid",
        "--server-port",
        "5432",
        "--timeout",
        "19",
        "--try-run",
        "--quiet",
    ]
    if not local_mode:
        argv.append("--no-local-mode")
    constructor = Mock()
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(model_adapter, "ModelAdapter", constructor)

    model_adapter.main()

    expected_config = {
        **config,
        "model_name": "cli-model",
        "model_path": "cli-model",
        "url": "https://cli.invalid/v1/embeddings",
        "api_key": "cli-key",
        "text_batch_size": 3,
        "request_timeout": 9.5,
        "num_workers": 2,
        "query_instruction": "query: ",
    }
    constructor.assert_called_once_with(
        server_ip="http://server.invalid",
        server_port=5432,
        timeout=19,
        enable_accelerate=False,
        extra_cfg=expected_config,
        local_mode=local_mode,
        task_names=["tiny-task.py"],
        output_dir=str(tmp_path),
        model_path="cli-model",
        data_root="/fake/data",
        debug=True,
        quiet=True,
    )
    constructor.return_value.run.assert_called_once_with()


def test_eval_cli_merges_embedding_options(monkeypatch, tmp_path):
    config = {
        "model_name": "configured-model",
        "text_batch_size": 64,
        "dimensions": 128,
        "query_instruction": "configured prefix",
        "document_instruction": "configured document",
    }
    config_file = tmp_path / "embedding.json"
    config_file.write_text(json.dumps(config))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evalmm",
            "--cfg",
            str(config_file),
            "--text-batch-size",
            "7",
            "--request-timeout",
            "15",
            "--truncate-prompt-tokens",
            "0",
            "--query-instruction",
            "",
            "--max-image-size",
            "1000",
            "--min-short-side",
            "10",
            "--max-long-side",
            "100",
        ],
    )
    args = model_adapter.parse_args()
    result = eval_cli.update_cfg_from_args(args)
    assert result["text_batch_size"] == 7
    assert result["request_timeout"] == 15
    assert result["truncate_prompt_tokens"] == 0
    assert result["query_instruction"] == ""
    assert result["max_image_size"] == 1000
    assert result["min_short_side"] == 10
    assert result["max_long_side"] == 100
    assert result["dimensions"] == 128
    assert result["document_instruction"] == "configured document"


@pytest.mark.parametrize("local_mode", [True, False])
def test_eval_command_forwards_mode_and_config(monkeypatch, tmp_path, local_mode):
    argv = [
        "evalmm",
        "--tasks",
        "tiny-task.py",
        "--exec",
        model_adapter.__file__,
        "--model",
        "embedding-model",
        "--url",
        "https://example.invalid/v1",
        "--dimensions",
        "128",
        "--output-dir",
        str(tmp_path),
        "--quiet",
    ]
    if not local_mode:
        argv.append("--no-local-mode")
    monkeypatch.setattr(sys, "argv", argv)
    args = model_adapter.parse_args()
    wrapper = eval_cli.ServerWrapper.__new__(eval_cli.ServerWrapper)
    wrapper.args = args
    wrapper.exec = args.exec
    wrapper.cfg = eval_cli.update_cfg_from_args(args)
    wrapper.evaluation_server_ip = args.server_ip
    wrapper.port = args.server_port
    wrapper.local_mode = args.local_mode

    command = wrapper._build_command()

    assert ("--no-local-mode" in command) == (not local_mode)
    assert ("--tasks" in command) == local_mode
    assert "--quiet" in command
    config = json.loads(command[command.index("--cfg") + 1])
    assert config["model_name"] == "embedding-model"
    assert config["url"] == "https://example.invalid/v1"
    assert config["dimensions"] == 128

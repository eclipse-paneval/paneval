import json
import os
import os.path as osp
from concurrent.futures import ThreadPoolExecutor
from tempfile import NamedTemporaryFile
from typing import Any, Dict

import numpy as np

from evalmm.models import EmbeddingClient
from evalmm.models.base_model_adapter import BaseModelAdapter
from evalmm.server.utils import parse_args


CLIENT_CONFIG_KEYS = (
    "model_name",
    "url",
    "api_key",
    "text_batch_size",
    "request_timeout",
    "max_image_size",
    "min_short_side",
    "max_long_side",
    "dimensions",
    "truncate_prompt_tokens",
    "query_instruction",
    "document_instruction",
)


def save_array(path: str, array: np.ndarray):
    """Publish only complete cache batches and similarity matrices."""
    with NamedTemporaryFile(dir=osp.dirname(path), suffix=".tmp", delete=False) as f:
        temporary_path = f.name
        try:
            np.save(f, array)
        except BaseException:
            os.unlink(temporary_path)
            raise
    try:
        os.replace(temporary_path, path)
    finally:
        if osp.exists(temporary_path):
            os.unlink(temporary_path)


class ModelAdapter(BaseModelAdapter):
    def model_init(self, task_info: Dict):
        config = {key: task_info[key] for key in CLIENT_CONFIG_KEYS if key in task_info}
        config.setdefault("model_name", self.model_name)
        self.model = EmbeddingClient(**config)

    def _embed_batch(self, task_name, cache_dir, data_type, start, end):
        cache_file = osp.join(cache_dir, f"{data_type}_{start:07d}_{end:07d}.npy")
        if osp.exists(cache_file):
            embeddings = np.load(cache_file, allow_pickle=False)
        else:
            if data_type == "text":
                inputs = [
                    self.task_manager.get_retrieval_data(task_name, i, "text")[
                        "caption"
                    ]
                    for i in range(start, end)
                ]
                embeddings = self.model.embed_texts(inputs)
            else:
                inputs = [
                    self.task_manager.get_retrieval_data(task_name, i, "img")[
                        "img_path"
                    ]
                    for i in range(start, end)
                ]
                embeddings = self.model.embed_images(inputs)
            embeddings = self._validate_batch(embeddings, end - start)
            save_array(cache_file, embeddings)
        return self._validate_batch(embeddings, end - start)

    @staticmethod
    def _validate_batch(embeddings, count):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if (
            embeddings.ndim != 2
            or embeddings.shape[0] != count
            or embeddings.shape[1] == 0
        ):
            raise ValueError(f"Expected {count} embedding rows, got {embeddings.shape}")
        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain non-finite values")
        if np.any(np.linalg.norm(embeddings, axis=1) == 0):
            raise ValueError("Cannot normalize zero embedding vectors")
        return embeddings

    def run_one_task(self, task_name: str, meta_info: Dict[str, Any]):
        if "video_number" in meta_info and "image_number" not in meta_info:
            raise NotImplementedError(
                "Video retrieval is not supported by the embedding API adapter"
            )
        n_images = meta_info["image_number"]
        n_captions = meta_info["caption_number"]
        if n_images < 1 or n_captions < 1:
            raise ValueError("Image retrieval requires images and captions")
        output_dir = meta_info["output_dir"]
        cache_dir = osp.join(output_dir, "emb_cache")
        os.makedirs(cache_dir, exist_ok=True)
        batches = [
            ("text", start, min(start + self.model.text_batch_size, n_captions))
            for start in range(0, n_captions, self.model.text_batch_size)
        ] + [("image", i, i + 1) for i in range(n_images)]

        pool = ThreadPoolExecutor(max_workers=self.task_info.get("num_workers", 1))
        try:
            futures = [
                pool.submit(self._embed_batch, task_name, cache_dir, kind, start, end)
                for kind, start, end in batches
            ]
            # Consume in dataset order even when requests complete out of order.
            results = [future.result() for future in futures]
        except BaseException:
            # Abort promptly: drop queued batches instead of running them all
            # (and their retries) before the error surfaces. Batches already
            # in flight finish and stay cached for the next run.
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        pool.shutdown(wait=True)

        text_batch_count = (n_captions + self.model.text_batch_size - 1) // (
            self.model.text_batch_size
        )
        text_emb = np.concatenate(results[:text_batch_count])
        image_emb = np.concatenate(results[text_batch_count:])
        if image_emb.shape[1] != text_emb.shape[1]:
            raise ValueError("Image and text embedding dimensions do not match")
        text_emb /= np.linalg.norm(text_emb, axis=1, keepdims=True)
        image_emb /= np.linalg.norm(image_emb, axis=1, keepdims=True)
        sim = image_emb @ text_emb.T
        save_array(osp.join(output_dir, f"{meta_info['name']}.npy"), sim)
        with open(osp.join(output_dir, f"{meta_info['name']}.json"), "w") as f:
            json.dump(
                {
                    "model_name": self.model_name,
                    "embedding_dim": int(image_emb.shape[1]),
                    "n_images": n_images,
                    "n_captions": n_captions,
                },
                f,
                indent=2,
            )


def main():
    args = parse_args()
    config = {}
    if args.cfg:
        if osp.isfile(args.cfg):
            with open(args.cfg) as f:
                config = json.load(f)
        else:
            config = json.loads(args.cfg)
    for key in (*CLIENT_CONFIG_KEYS, "num_workers"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    if args.model:
        config["model_name"] = args.model
        config["model_path"] = args.model
    model_adapter = ModelAdapter(
        server_ip=args.server_ip,
        server_port=args.server_port,
        timeout=args.timeout,
        enable_accelerate=False,
        extra_cfg=config,
        local_mode=args.local_mode,
        task_names=args.tasks,
        output_dir=args.output_dir,
        model_path=args.model,
        data_root=args.data_root,
        debug=args.debug or args.try_run,
        quiet=args.quiet,
    )
    model_adapter.run()


if __name__ == "__main__":
    main()

import os
from typing import Dict, Any
from transformers import CLIPModel
from transformers import CLIPProcessor
import torch
import numpy as np
from PIL import Image
from flagevalmm.models.base_model_adapter import BaseModelAdapter

from flagevalmm.server.utils import parse_args, get_retrieval_data


class ModelAdapter(BaseModelAdapter):
    def model_init(self, task_info: Dict):
        ckpt_path = task_info["model_path"]
        self.model = CLIPModel.from_pretrained(ckpt_path).cuda()
        self.preprocess = CLIPProcessor.from_pretrained(ckpt_path)

    def get_image(self, image_id, task_name):
        response = get_retrieval_data(
            index=image_id,
            task_name=task_name,
            data_type="img",
            server_ip=self.server_ip,
            server_port=self.server_port,
            timeout=self.timeout,
        )
        image = Image.open(response["img_path"]).convert("RGB")
        processed = self.preprocess(images=image, return_tensors="pt")
        return processed["pixel_values"].squeeze(0).cuda()

    def get_caption(self, caption_id, task_name):
        response = get_retrieval_data(
            index=caption_id,
            task_name=task_name,
            data_type="text",
            server_ip=self.server_ip,
            server_port=self.server_port,
            timeout=self.timeout,
        )
        return response["caption"]

    def run_one_task(self, task_name: str, meta_info: Dict[str, Any]):
        output_dir = meta_info["output_dir"]
        max_cap_len = 77
        itd = 50
        N = meta_info["image_number"]

        acc_image_embeddings = None
        acc_text_embeddings = None

        for i in range(0, N, itd):
            if i % itd == 0:
                print("{}/{}=={}%".format(i, N, 100.0 * i / N))
            _s, _e = i, min(i + itd, N)

            images = [self.get_image(image_id, task_name) for image_id in range(_s, _e)]
            images = torch.stack(images, 0).squeeze()

            caption = [
                self.get_caption(caption_id, task_name)
                for caption_id in range(_s * 5, _e * 5)
            ]
            texts = [
                self.preprocess(
                    text=cap,
                    padding="max_length",
                    truncation=True,
                    max_length=max_cap_len,
                    return_tensors="pt",
                )["input_ids"].cuda()
                for cap in caption
            ]
            texts = torch.cat(texts, 0)

            with torch.no_grad():
                image_features = self.model.get_image_features(images)
                text_features = self.model.get_text_features(texts)
                image_embeddings = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                text_embeddings = text_features / text_features.norm(
                    dim=-1, keepdim=True
                )

            torch.cuda.empty_cache()

            # Convert to numpy and accumulate
            batch_image_embeddings = image_embeddings.cpu().numpy()
            batch_text_embeddings = text_embeddings.cpu().numpy()

            if acc_image_embeddings is None:
                acc_image_embeddings = batch_image_embeddings
                acc_text_embeddings = batch_text_embeddings
            else:
                acc_image_embeddings = np.concatenate(
                    (acc_image_embeddings, batch_image_embeddings), axis=0
                )
                acc_text_embeddings = np.concatenate(
                    (acc_text_embeddings, batch_text_embeddings), axis=0
                )

        # Final similarity computation
        acc_image_embeddings = torch.from_numpy(acc_image_embeddings).cuda()
        acc_text_embeddings = torch.from_numpy(acc_text_embeddings).cuda()
        acc_sim = acc_image_embeddings.mm(acc_text_embeddings.T)
        acc_sim = acc_sim.cpu().numpy()

        full_save_path = os.path.join(output_dir, meta_info["name"])
        np.save("{}".format(full_save_path), acc_sim)
        return acc_sim


if __name__ == "__main__":
    args = parse_args()
    model_adapter = ModelAdapter(
        server_ip=args.server_ip,
        server_port=args.server_port,
        timeout=args.timeout,
        extra_cfg=args.cfg,
    )
    model_adapter.run()

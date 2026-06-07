from typing import Dict, Any, Optional, Callable, List
import torch
import os.path as osp
import json
from torch.utils.data import DataLoader
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

from flagevalmm.server.utils import (
    process_images_symbol,
    load_pil_image,
    parse_args,
    default_collate_fn,
    get_task_info,
)
from flagevalmm.models.base_model_adapter import BaseModelAdapter
from flagevalmm.server import ServerDataset

from vllm import LLM, SamplingParams


class CustomDataset(ServerDataset):
    def __getitem__(self, index):
        data = self.get_data(index)

        question_id = data["question_id"]
        qs = data["question"]

        qs, idx = process_images_symbol(qs, "")

        image_list, _ = load_pil_image(
            data["img_path"], idx, reduplicate=True, reqiures_img=True
        )
        return question_id, qs, image_list


class ModelAdapterVllm(BaseModelAdapter):
    def __init__(
        self,
        server_ip: str,
        server_port: int,
        timeout: int = 1000,
        extra_cfg: str | Dict | None = None,
    ) -> None:
        self.server_ip: str = server_ip
        self.server_port: int = server_port

        self.timeout: int = timeout
        task_info: Dict[str, Any] = get_task_info(server_ip, server_port)
        self.tasks: List[str] = task_info["task_names"]

        if isinstance(extra_cfg, str):
            try:
                with open(extra_cfg, "r") as f:
                    extra_cfg = json.load(f)
            except Exception as e:
                print(f"Error loading extra config file: {e}")
        if extra_cfg is not None:
            task_info.update(extra_cfg)
        self.task_info: Dict[str, Any] = task_info
        self.model_name: str = task_info.get("model_name", None)
        if self.model_name is None and "model_path" in task_info:
            self.model_name = osp.basename(task_info["model_path"])
        self.model_init(task_info)

    def model_init(self, task_info: Dict):
        ckpt_path = task_info["model_path"]
        self.model = LLM(
            model=ckpt_path,
            limit_mm_per_prompt={"image": 18},
            max_model_len=32768,
            tensor_parallel_size=4,
        )
        self.sampling_params = SamplingParams(
            max_tokens=1024, temperature=0.0, stop_token_ids=None
        )
        self.processor = AutoProcessor.from_pretrained(ckpt_path)

    def build_prompt(self, question, images):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                ]
                + [{"type": "image"} for _ in range(len(images))],
            },
        ]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        return prompt

    def create_data_loader(
        self,
        dataset_cls: type[ServerDataset],
        task_name: str,
        collate_fn: Optional[Callable] = None,
        batch_size: int = 1,
        num_workers: int = 2,
    ):
        dataset = dataset_cls(task_name, self.server_ip, self.server_port, self.timeout)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            shuffle=False,
        )
        return data_loader

    def run_one_task(self, task_name: str, meta_info: Dict[str, Any]):
        results = []
        data_loader = self.create_data_loader(
            CustomDataset,
            task_name,
            collate_fn=default_collate_fn,
            batch_size=4,
            num_workers=2,
        )
        for question_id, batch_question, batch_images in data_loader:
            inputs = []
            for question, images in zip(batch_question, batch_images):
                prompt = self.build_prompt(question, images)
                inputs.append({"prompt": prompt, "multi_modal_data": {"image": images}})
            outputs = self.model.generate(inputs, sampling_params=self.sampling_params)
            for i, output in enumerate(outputs):
                response = output.outputs[0].text.strip()
                print(f"{batch_question[i]}\n{response}\n\n")
                results.append(
                    {
                        "question_id": question_id[i],
                        "answer": response,
                        "prompt": batch_question[i],
                    }
                )
        self.save_result(results, meta_info)


class ModelAdapter(ModelAdapterVllm):
    def model_init(self, task_info: Dict):
        ckpt_path = task_info["model_path"]
        with self.accelerator.main_process_first():
            self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                ckpt_path,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
        self.model = self.accelerator.prepare_model(self.model, evaluation_mode=True)
        self.processor = AutoProcessor.from_pretrained(ckpt_path)

    def run_one_task(self, task_name: str, meta_info: Dict[str, Any]):
        results = []
        data_loader = self.create_data_loader(
            CustomDataset,
            task_name,
            collate_fn=default_collate_fn,
            batch_size=1,
            num_workers=0,
        )
        cnt = 0
        for question_id, batch_question, batch_images in data_loader:
            cnt += 1
            for qid, question, images in zip(question_id, batch_question, batch_images):
                prompt = self.build_prompt(question, images)
                inputs = self.processor(
                    images=images, text=prompt, return_tensors="pt"
                ).to(self.model.device, torch.float16)
                output = self.model.generate(
                    **inputs, max_new_tokens=1024, do_sample=False
                )
                output_ids = output[0][inputs["input_ids"].shape[1] :]
                response = self.processor.decode(output_ids, skip_special_tokens=True)
                print(f"{question}\n{response}\n")
                results.append(
                    {
                        "question_id": qid,
                        "answer": response,
                        "prompt": question,
                    }
                )
        self.save_result(results, meta_info)


if __name__ == "__main__":
    args = parse_args()
    model_adapter = ModelAdapterVllm(
        server_ip=args.server_ip,
        server_port=args.server_port,
        timeout=args.timeout,
        extra_cfg=args.cfg,
    )
    model_adapter.run()

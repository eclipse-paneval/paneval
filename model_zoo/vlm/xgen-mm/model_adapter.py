from typing import Dict, Any
import torch

import time
from transformers import AutoImageProcessor, AutoModelForVision2Seq, AutoTokenizer

from flagevalmm.server.utils import parse_args, default_collate_fn, load_pil_image
from flagevalmm.server.server_dataset import ServerDataset
from flagevalmm.models.base_model_adapter import BaseModelAdapter
import re


def replace_images_symbol(text):
    pattern = r"<image (\d+)>"
    matches = re.findall(pattern, text)
    for i, match in enumerate(matches):
        text = text.replace(f"<image {match}>", "<image>", 1)
    return text, [int(num) - 1 for num in matches]


class CustomDataset(ServerDataset):
    def __getitem__(self, index):
        data = self.get_data(index)
        qs, idx = replace_images_symbol(data["question"])
        question_id = data["question_id"]
        img_path = data["img_path"]
        image_list = []
        image_list, _ = load_pil_image(img_path, idx, reqiures_img=True)
        return question_id, qs, image_list

    def __len__(self):
        return self.length


class ModelAdapter(BaseModelAdapter):
    def model_init(self, task_info: Dict):
        ckpt_path = task_info["model_path"]

        torch.set_grad_enabled(False)
        with self.accelerator.main_process_first():
            model = AutoModelForVision2Seq.from_pretrained(
                ckpt_path, device_map="cuda", trust_remote_code=True, torch_dtype="auto"
            ).eval()

            self.processor = AutoImageProcessor.from_pretrained(
                ckpt_path, trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                ckpt_path, trust_remote_code=True, use_fast=False, legacy=False
            )
            tokenizer = model.update_special_tokens(tokenizer)
            tokenizer.eos_token = "<|end|>"
            tokenizer.padding_side = "left"
            self.tokenizer = tokenizer

        model = self.accelerator.prepare_model(model, evaluation_mode=True)
        if hasattr(model, "module"):
            model = model.module
        self.model = model

    def apply_prompt_template(self, query):
        s = (
            "<|system|>\nA chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
            f"<|user|>\n{query}<|end|>\n<|assistant|>\n"
        )
        return s

    def run_one_task(self, task_name: str, meta_info: Dict[str, Any]):
        results = []
        cnt = 0

        data_loader = self.create_data_loader(
            CustomDataset,
            task_name,
            collate_fn=default_collate_fn,
            batch_size=1,
            num_workers=2,
        )
        for question_id, question, images in data_loader:
            if cnt == 1:
                start_time = time.perf_counter()
            cnt += 1

            content, images_processed, image_sizes = question[0], [], []
            for image in images[0]:
                images_processed.append(
                    self.processor([image], image_aspect_ratio="anyres")[
                        "pixel_values"
                    ].to("cuda", dtype=torch.bfloat16)
                )
                image_sizes.append(image.size)

            print(cnt)

            inputs = {"pixel_values": [images_processed]}
            prompt = self.apply_prompt_template(content)
            language_inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
            inputs.update(language_inputs)

            generation_args = {
                "max_new_tokens": 1024,
                "temperature": 0.0,
                "do_sample": False,
                "top_p": None,
                "num_beams": 1,
            }

            generate_ids = self.model.generate(
                **inputs,
                image_size=[image_sizes],
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **generation_args,
            )

            # remove input tokens
            response = self.tokenizer.decode(
                generate_ids[0], skip_special_tokens=True
            ).split("<|end|>")[0]
            self.accelerator.print(f"{question[0]}\n{response}\n\n")
            results.append(
                {
                    "question_id": question_id[0],
                    "answer": response.strip(),
                    "prompt": question[0],
                }
            )
        rank = self.accelerator.state.local_process_index

        # save results for the rank
        self.save_result(results, meta_info, rank=rank)
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            correct_num = self.collect_results_and_save(meta_info)
            total_time = time.perf_counter() - start_time
            print(
                f"Total time: {total_time}\nAverage time:{total_time / cnt}\nResults_collect number: {correct_num}"
            )

        print("rank", rank, "finished")


if __name__ == "__main__":
    args = parse_args()
    model_adapter = ModelAdapter(
        server_ip=args.server_ip,
        server_port=args.server_port,
        timeout=args.timeout,
        extra_cfg=args.cfg,
    )
    model_adapter.run()

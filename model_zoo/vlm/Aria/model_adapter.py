from typing import Dict, Any
import torch

import time
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor

from flagevalmm.server.utils import parse_args, load_pil_image, default_collate_fn
from flagevalmm.server.server_dataset import ServerDataset
from flagevalmm.models.base_model_adapter import BaseModelAdapter
import re


def replace_images_symbol(text):
    pattern = r"<image (\d+)>"
    matches = re.findall(pattern, text)
    for i, match in enumerate(matches):
        text = text.replace(f"<image {match}>", f"<Image {i + 1}>", 1)
    return text, [int(num) - 1 for num in matches]


class CustomDataset(ServerDataset):
    def __getitem__(self, index):
        data = self.get_data(index)
        qs, idx = replace_images_symbol(data["question"])
        question_id = data["question_id"]
        img_path = data["img_path"]
        image_list, idx = load_pil_image(img_path, idx, reqiures_img=True)
        return question_id, qs, image_list


class ModelAdapter(BaseModelAdapter):
    def model_init(self, task_info: Dict):
        ckpt_path = task_info["model_path"]

        torch.set_grad_enabled(False)
        with self.accelerator.main_process_first():
            model = AutoModelForCausalLM.from_pretrained(
                ckpt_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

            self.processor = AutoProcessor.from_pretrained(
                ckpt_path, trust_remote_code=True
            )
        model = self.accelerator.prepare_model(model, evaluation_mode=True)
        if hasattr(model, "module"):
            model = model.module
        self.model = model

    def build_message(
        self,
        query: str,
        images=[],
    ) -> str:
        messages = []
        messages.append(
            {
                "role": "user",
                "content": [],
            }
        )
        for i in range(len(images)):
            messages[-1]["content"].extend(
                [
                    {"text": f"Image {i+1}: ", "type": "text"},
                    {"text": None, "type": "image"},
                    {"text": "\n", "type": "text"},
                ]
            )
        messages[-1]["content"].append(
            {
                "type": "text",
                "text": query,
            }
        )
        return messages

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
            images = images[0]
            messages = self.build_message(question[0], images)
            text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self.processor(
                text=text,
                images=images,
                return_tensors="pt",
                max_image_size=980,
                split_image=False,
            )
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with (
                torch.inference_mode(),
                torch.amp.autocast("cuda", dtype=torch.bfloat16),
            ):
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    stop_strings=["<|im_end|>"],
                    tokenizer=self.processor.tokenizer,
                    do_sample=False,
                )
                output_ids = output[0][inputs["input_ids"].shape[1] :]
                response = self.processor.decode(
                    output_ids, skip_special_tokens=True
                ).replace("<|im_end|>", "")

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

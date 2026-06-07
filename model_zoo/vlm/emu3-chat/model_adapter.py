from typing import Dict, Any
import torch
from flagevalmm.server.utils import (
    process_images_symbol,
    load_pil_image,
    parse_args,
    default_collate_fn,
)
from flagevalmm.models.base_model_adapter import BaseModelAdapter
from flagevalmm.server import ServerDataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoImageProcessor,
    AutoModelForCausalLM,
)
from transformers.generation.configuration_utils import GenerationConfig
from flagevalmm.common.image_utils import concat_images

import sys
import math
from PIL import Image


class CustomDataset(ServerDataset):
    def __getitem__(self, index):
        data = self.get_data(index)

        question_id = data["question_id"]
        qs = data["question"]

        qs, idx = process_images_symbol(qs)
        image_list, _ = load_pil_image(
            data["img_path"], idx, reduplicate=False, reqiures_img=True
        )
        # check image absolute aspect ratio which must be smaller than 5
        # if not, padding the image to the aspect ratio of 5:1 or 1:5
        img = concat_images(image_list) if len(image_list) > 1 else image_list[0]
        h, w = img.size
        if h / w > 5 or w / h > 5:
            # Calculate new dimensions to maintain max 5:1 ratio
            new_w = math.ceil(h / 5) if h / w > 5 else w
            new_h = math.ceil(w / 5) if w / h > 5 else h
            img_new = Image.new("RGB", (new_w, new_h), (255, 255, 255))
            img_new.paste(img, (0, 0))
            img = img_new
        return question_id, qs, img


class ModelAdapter(BaseModelAdapter):
    def model_init(self, task_info: Dict):
        ckpt_path = task_info["model_path"]
        sys.path.append(ckpt_path + "-Chat")
        from processing_emu3 import Emu3Processor

        with self.accelerator.main_process_first():
            model = AutoModelForCausalLM.from_pretrained(
                ckpt_path + "-Chat",
                device_map="cuda:0",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
            self.image_processor = AutoImageProcessor.from_pretrained(
                ckpt_path + "-VisionTokenizer", trust_remote_code=True
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            ckpt_path + "-Chat", trust_remote_code=True, padding_side="left"
        )
        self.image_tokenizer = AutoModel.from_pretrained(
            ckpt_path + "-VisionTokenizer", device_map="cuda:0", trust_remote_code=True
        ).eval()
        self.processor = Emu3Processor(
            self.image_processor, self.image_tokenizer, self.tokenizer
        )
        model = self.accelerator.prepare_model(model, evaluation_mode=True)
        if hasattr(model, "module"):
            model = model.module
        self.model = model

    def run_one_task(self, task_name: str, meta_info: Dict[str, Any]):
        results = []
        data_loader = self.create_data_loader(
            CustomDataset,
            task_name,
            collate_fn=default_collate_fn,
            batch_size=1,
            num_workers=0,
        )
        for question_id, batch_question, batch_images in data_loader:
            for qid, question, image in zip(question_id, batch_question, batch_images):
                inputs = self.processor(
                    text=question,
                    image=image,
                    mode="U",
                    return_tensors="pt",
                    padding="longest",
                )
                # prepare hyper parameters
                GENERATION_CONFIG = GenerationConfig(
                    pad_token_id=self.tokenizer.pad_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=1024,
                )

                # generate
                outputs = self.model.generate(
                    inputs.input_ids.to("cuda:0"),
                    GENERATION_CONFIG,
                    attention_mask=inputs.attention_mask.to("cuda:0"),
                )
                outputs = outputs[:, inputs.input_ids.shape[-1] :]
                response = self.processor.batch_decode(
                    outputs, skip_special_tokens=True
                )[0]
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
    model_adapter = ModelAdapter(
        server_ip=args.server_ip,
        server_port=args.server_port,
        timeout=args.timeout,
        extra_cfg=args.cfg,
    )
    model_adapter.run()

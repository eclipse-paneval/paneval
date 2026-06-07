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
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from flagevalmm.common.image_utils import concat_images

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class CustomDataset(ServerDataset):
    def __getitem__(self, index):
        data = self.get_data(index)

        question_id = data["question_id"]
        qs = data["question"]

        qs, idx = process_images_symbol(qs, dst_pattern="")
        qs = "<image>\n" + qs
        image_list, _ = load_pil_image(
            data["img_path"], idx, reduplicate=True, reqiures_img=True
        )
        image_list = concat_images(image_list) if len(image_list) > 1 else image_list[0]
        return question_id, qs, image_list


class ModelAdapter(BaseModelAdapter):
    def model_init(self, task_info: Dict):
        ckpt_path = task_info["model_path"]
        with self.accelerator.main_process_first():
            model = (
                AutoModel.from_pretrained(
                    ckpt_path,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
                .eval()
                .cuda()
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            ckpt_path, trust_remote_code=True, use_fast=False
        )
        model = self.accelerator.prepare_model(model, evaluation_mode=True)
        if hasattr(model, "module"):
            model = model.module
        self.model = model

    def run_one_task(self, task_name: str, meta_info: Dict[str, Any]):
        results = []
        generation_config = dict(max_new_tokens=1024, do_sample=True)
        data_loader = self.create_data_loader(
            CustomDataset,
            task_name,
            collate_fn=default_collate_fn,
            batch_size=1,
            num_workers=2,
        )
        for question_id, batch_question, batch_images in data_loader:
            for qid, question, image in zip(question_id, batch_question, batch_images):
                pixel_values = load_image(image, max_num=12).to(torch.bfloat16).cuda()
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    question,
                    generation_config=generation_config,
                )
                print(f"{question}\n{response}\n\n")
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

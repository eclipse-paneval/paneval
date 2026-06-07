import os.path as osp
import os
import json
from datasets import load_dataset
import tqdm


def process(cfg):
    data_dir, split = cfg.dataset_path, cfg.split
    name = cfg.get("dataset_name", "")
    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    os.makedirs(osp.join(output_dir, "image"), exist_ok=True)
    dataset = load_dataset(data_dir, name, split=split)

    content = []
    select_keys = [
        "question",
        "problem_index",
        "answer",
        "question_type",
        "metadata",
        "problem_version",
    ]
    for data in tqdm.tqdm(dataset):
        new_data = {
            "question_id": data["sample_index"],
        }
        for key in select_keys:
            new_data[key] = data[key]
        new_data["img_path"] = f'image/{data["sample_index"]}.png'
        if not osp.exists(osp.join(output_dir, new_data["img_path"])):
            data["image"].save(osp.join(output_dir, new_data["img_path"]))
        content.append(new_data)
    out_file_name = "data.json"
    with open(osp.join(output_dir, out_file_name), "w") as f:
        json.dump(content, f, indent=2)

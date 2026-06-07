import os.path as osp
import json
from datasets import load_dataset
import tqdm
import os


def process(cfg):
    data_dir, split = cfg.dataset_path, cfg.split
    dataset = load_dataset(data_dir, split=split)
    name = cfg.get("dataset_name", "")
    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    content = []
    select_keys = ["category", "subcategory", "set_id", "figure_id", "question"]
    for i, data in enumerate(tqdm.tqdm(dataset)):
        new_data = {}
        for key in select_keys:
            new_data[key] = data[key]
        new_data["question_id"] = str(i)
        new_data["question_id_ori"] = data["question_id"]
        new_data["img_path"] = data["filename"].replace("./", "")
        new_data["answer"] = "yes" if data["gt_answer"] == "1" else "no"
        new_data["question_type"] = "yes-no"
        content.append(new_data)
        if not osp.exists(osp.join(output_dir, new_data["img_path"])):
            os.makedirs(
                osp.dirname(osp.join(output_dir, new_data["img_path"])), exist_ok=True
            )
            data["image"].save(osp.join(output_dir, new_data["img_path"]))
    with open(osp.join(output_dir, "data.json"), "w") as f:
        json.dump(content, f)

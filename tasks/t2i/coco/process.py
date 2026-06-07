import json
import os
import os.path as osp
from datasets import load_dataset


def process(cfg):
    data_dir, split = cfg.dataset_path, cfg.split

    name = cfg.get("dataset_name", "")
    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    selected_keys = ["prompt", "id", "lan"]
    content = []
    data = load_dataset(data_dir, name=name, split=split)
    os.makedirs(osp.join(output_dir, "image"), exist_ok=True)
    for i, annotation in enumerate(data):
        info = {k: annotation[k] for k in selected_keys}
        info["img_path"] = osp.join("image", f"{info['id']}.jpg")
        annotation["image"].save(osp.join(output_dir, info["img_path"]))
        content.append(info)
    json.dump(content, open(osp.join(output_dir, "data.json"), "w"), indent=2)

import json
from datasets import load_dataset
import os.path as osp
import os
import tqdm


def process(cfg):
    data_dir, split = cfg.dataset_path, cfg.split
    dataset = load_dataset(data_dir, split=split)
    name = cfg.get("dataset_name", "")
    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    content = []
    select_keys = [
        "id",
        "question",
        "correct_option",
        "answer",
        "image_type",
        "difficulty",
        "domain",
        "emotion",
        "rhetoric",
        "explanation",
        "metaphorical_meaning",
        "local_path",
    ]
    NUM_OPTIONS = 6

    for data in tqdm.tqdm(dataset):
        new_data = {key: data[key] for key in select_keys}
        new_data["question_id"] = new_data.pop("id")
        new_data["img_path"] = new_data.pop("local_path").replace("test/", "")
        new_data["question_type"] = "multiple-choice"
        new_data["options"] = []
        for i in range(NUM_OPTIONS):
            if f"option{i+1}" not in data:
                break
            new_data["options"].append(data[f"option{i+1}"])
        if data["image"].mode != "RGB":
            data["image"] = data["image"].convert("RGB")
        img_path = osp.join(output_dir, new_data["img_path"])
        if not osp.exists(osp.dirname(img_path)):
            os.makedirs(osp.dirname(img_path))
        data["image"].save(img_path)

        content.append(new_data)

    out_file = osp.join(output_dir, "data.json")
    with open(out_file, "w") as fout:
        json.dump(content, fout, indent=2, ensure_ascii=False)

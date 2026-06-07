import json
import os.path as osp
from datasets import load_dataset
import os
import tqdm


def process(cfg):
    data_dir, split = cfg.dataset_path, cfg.split
    name = cfg.get("dataset_name", "")
    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    os.makedirs(osp.join(output_dir, "image"), exist_ok=True)
    dataset = load_dataset(data_dir, name, split=split)

    content = []
    id_set = set()
    for data in tqdm.tqdm(dataset):
        question_id = data["idx"]
        if question_id in id_set:
            print(f"Duplicate id: {question_id}")
        id_set.add(question_id)
        converted = {
            "question_id": question_id,
            "type_ori": data["type"],
            "question_type": "multiple-choice",
            "answer": data["answer"][1],
            "options": data["choices"],
            "img_path": data["filename"],
        }
        assert "A" <= converted["answer"] <= "Z"
        keys = [
            "task",
            "question",
            "source",
            "source_dataset",
            "source_filename",
            "target_class",
            "target_size",
            "bbox",
        ]
        for key in keys:
            converted[key] = data[key]
        # save image
        img_path = osp.join(output_dir, data["filename"])
        os.makedirs(osp.dirname(img_path), exist_ok=True)
        data["image"].save(img_path)

        content.append(converted)
    with open(osp.join(output_dir, "data.json"), "w") as fout:
        json.dump(content, fout, indent=2)

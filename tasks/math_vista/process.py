import json
from datasets import load_dataset
import os.path as osp
import os
import tqdm


def get_answer(row):
    if row["question_type"] != "multi_choice":
        return row["answer"]
    choices = row["choices"]
    for i, choice in enumerate(choices):
        if choice == row["answer"]:
            return chr(ord("A") + i)
    return row["answer"]


def process(cfg):
    data_dir, split = cfg.dataset_path, cfg.split
    dataset = load_dataset(data_dir, split=split)
    name = cfg.get("dataset_name", "")
    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    os.makedirs(osp.join(output_dir, "images"), exist_ok=True)

    converted_data = []
    type_mp = {
        "free_form": "short-answer",
        "multi_choice": "multiple-choice",
    }
    for row in tqdm.tqdm(dataset):
        info = {
            "question_id": row["pid"],
            "img_path": row["image"],
            "question": row["query"],
            "question_type": type_mp[row["question_type"]],
            "answer": get_answer(row),
            "answer_raw": row["answer"],
            "answer_type": row["answer_type"],
            "precision": row["precision"],
            "unit": row["unit"],
            "metadata": row["metadata"],
        }
        row["decoded_image"].convert("RGB").save(osp.join(output_dir, row["image"]))
        converted_data.append(info)
    json.dump(
        converted_data,
        open(f"{output_dir}/data.json", "w"),
        indent=2,
        ensure_ascii=False,
    )

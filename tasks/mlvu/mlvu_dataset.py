import json
import os
import os.path as osp
import numpy as np
from torch.utils.data import Dataset
from flagevalmm.registry import DATASETS


@DATASETS.register_module()
class MLVUDataset(Dataset):
    def __init__(self, data_root, name, split, debug=False, with_label=True):
        self.name = name
        self.with_label = with_label or debug

        video_dir = osp.join(data_root, "video")
        if split == "dev":
            data_info = {
                "count": ("4_count.json", f"{video_dir}/4_count", "video"),
                "ego": ("3_ego.json", f"{video_dir}/3_ego", "video"),
                "needle": ("2_needle.json", f"{video_dir}/2_needle", "video"),
                "order": ("5_order.json", f"{video_dir}/5_order", "video"),
                "plotQA": ("1_plotQA.json", f"{video_dir}/1_plotQA", "video"),
                "anomaly_reco": (
                    "6_anomaly_reco.json",
                    f"{video_dir}/6_anomaly_reco",
                    "video",
                ),
                "topic_reasoning": (
                    "7_topic_reasoning.json",
                    f"{video_dir}/7_topic_reasoning",
                    "video",
                ),
            }
        elif split == "test":
            data_info = {"test": ("multi_choice_QA.json", f"{video_dir}", "video")}
        else:
            raise ValueError(f"Invalid split: {split}")
        self.data_list = []
        for k, v in data_info.items():
            with open(os.path.join(data_root, "json", v[0]), "r") as f:
                json_data = json.load(f)
            for data in json_data:
                # get hash of data
                self.data_list.append(
                    {
                        "task_type": data["question_type"],
                        "prefix": v[1],
                        "data_type": v[2],
                        "data": data,
                    }
                )
        if debug:
            self.data_list = self.data_list[:32]

    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data["task_type"] not in len_list:
                len_list[data["task_type"]] = 0
            len_list[data["task_type"]] += 1
            if data["task_type"] not in option_list:
                option_list[data["task_type"]] = 0
            option_list[data["task_type"]] += len(data["data"]["candidates"])

        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()

    def __len__(self):
        return len(self.data_list)

    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array(
            [
                int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
                for idx in range(self.num_segments)
            ]
        )
        return frame_indices

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data["answer"]
        answer_idx = -1
        for idx, c in enumerate(data["candidates"]):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

    def __getitem__(self, idx):
        video_path = os.path.join(
            self.data_list[idx]["prefix"], self.data_list[idx]["data"]["video"]
        )
        question, answer = self.qa_template(self.data_list[idx]["data"])

        ret = {
            "question_id": self.data_list[idx]["data"]["question_id"],
            "video_path": [video_path],
            "question": question,
            "task_type": self.data_list[idx]["task_type"],
        }
        if self.with_label:
            ret["answer"] = answer
        return ret

    def meta_info(self):
        return {"name": self.name, "length": self.__len__(), "type": "mvu"}

    def get_annotation(self):
        anno_dict = {}
        for data in self.data_list:
            _, answer = self.qa_template(data["data"])
            question_id = data["data"]["question_id"]
            anno_dict[str(question_id)] = {"data": data, "answer": answer}
        return anno_dict

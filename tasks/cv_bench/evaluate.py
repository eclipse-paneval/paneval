from typing import Dict, List
from flagevalmm.evaluator.common_types import evaluate_multiple_choice


def cal_accuracy(
    annotations: Dict, predictions: List[Dict], target_source: str
) -> float:
    right = 0
    total = 0
    for pred in predictions:
        question_id = str(pred["question_id"])
        gt = annotations[question_id]
        if gt["source"] != target_source:
            continue
        total += 1
        is_correct = evaluate_multiple_choice(gt, pred)
        pred["correct"] = is_correct
        pred["label"] = gt["answer"]
        right += is_correct
    return round(right / (total + 1e-10) * 100, 2)


def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    results = {}
    targets = {
        "ADE20K": "accuracy_2d_ade",
        "COCO": "accuracy_2d_coco",
        "Omni3D": "accuracy_3d_omni",
    }
    for target, metric in targets.items():
        results[metric] = cal_accuracy(annotations, predictions, target)
    results["accuracy_2d"] = round(
        (results["accuracy_2d_ade"] + results["accuracy_2d_coco"]) / 2, 2
    )
    results["accuracy_3d"] = results["accuracy_3d_omni"]
    results["accuracy"] = round(
        (results["accuracy_2d"] + results["accuracy_3d"]) / 2, 2
    )
    return results

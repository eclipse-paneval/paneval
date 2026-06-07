from typing import Dict, List, Optional
from collections import defaultdict

# https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py#L81


def relaxed_correctness(
    target: str, prediction: str, max_relative_change: float = 0.05
) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith("%"):
                # Convert percentages to floats.
                return float(text.rstrip("%")) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    right = 0
    results = defaultdict(lambda: {"correct": 0, "total": 0, "accuracy": 0})
    for pred in predictions:
        question_id = str(pred["question_id"])
        gt = annotations[question_id]
        is_correct = relaxed_correctness(gt["answer"], pred["answer"])
        pred["correct"] = is_correct
        pred["label"] = gt["answer"]
        right += is_correct
        results[gt["ori_type"]]["correct"] += is_correct
        results[gt["ori_type"]]["total"] += 1
    for k, v in results.items():
        results[k]["accuracy"] = round(v["correct"] / v["total"] * 100, 2)
    results["accuracy"] = round(right / len(predictions) * 100, 2)
    return results

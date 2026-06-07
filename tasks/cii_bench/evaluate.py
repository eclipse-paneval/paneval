from typing import Dict, List
from collections import defaultdict
from flagevalmm.evaluator.pre_process import process_multiple_choice


def maybe_clean_answer(answer: str) -> str:
    if len(answer) == 1:
        return answer.upper()
    answer = process_multiple_choice(answer)
    return answer


def evaluate_multiple_choice(gt: Dict, pred: Dict) -> bool:
    pred["raw_answer"] = pred["answer"]
    pred["answer"] = maybe_clean_answer(pred["answer"])
    if len(pred["answer"]) > 1:
        pred["answer"] = pred["answer"][0]
    is_correct = gt["answer"].upper() == pred["answer"]
    return is_correct


domain_map = {
    "中华传统文化": "CTC",
    "生活": "Life",
    "艺术": "Art",
    "社会": "Society",
    "政治": "Politics",
    "环境": "Env.",
}

emotion_map = {"积极": "Positive", "消极": "Negative", "中性": "Neutral"}


def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    right = 0
    detailed_score = defaultdict(list)
    for pred in predictions:
        question_id = str(pred["question_id"])
        gt = annotations[question_id]
        is_correct = evaluate_multiple_choice(gt, pred)
        pred["correct"] = is_correct
        pred["label"] = gt["answer"]
        right += is_correct
        detailed_score[gt["domain"]].append(is_correct)
        detailed_score[gt["emotion"]].append(is_correct)
    result = {"accuracy": round(right / len(predictions) * 100, 2)}
    result["domain_score"] = {}
    result["emotion_score"] = {}
    for key, score in detailed_score.items():
        if key in domain_map:
            result["domain_score"][domain_map[key]] = round(
                sum(score) / len(score) * 100, 2
            )
        elif key in emotion_map:
            result["emotion_score"][emotion_map[key]] = round(
                sum(score) / len(score) * 100, 2
            )
    return result

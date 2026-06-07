from word2number import w2n
from flagevalmm.evaluator.common_types import evaluate_multiple_choice
from flagevalmm.evaluator.pre_process import normalize_string
from typing import Dict, List
import re


def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def extract_answer(answer, answer_type, precision):
    extraction = normalize_string(answer)
    extraction_bak = extraction
    if answer_type == "integer":
        if is_float(extraction):
            try:
                extraction = str(int(float(extraction)))
            except BaseException:
                extraction = None
        else:
            try:
                extraction = str(w2n.word_to_num(extraction))
            except BaseException:
                extraction = None
    elif answer_type == "float":
        try:
            extraction = str(round(float(extraction), int(precision)))
        except BaseException:
            extraction = None
    elif answer_type == "list":
        try:
            extraction = str(extraction)
        except BaseException:
            extraction = None
    if answer_type in ["integer", "float"] and extraction is None:
        pattern = r"\d+(?:,\d+)*(?:\.\d+)?"
        res = re.findall(pattern, extraction_bak)
        if len(res) > 0:
            extraction = res[0]

    return extraction


def evaluate_open_form(gt, answer):
    answer["raw_answer"] = answer["answer"]
    answer["answer"] = extract_answer(
        answer["answer"], gt["answer_type"], gt["precision"]
    )
    return gt["answer"] == answer["answer"]


def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    right = 0
    results = {}
    for answer in predictions:
        question_id = str(answer["question_id"])
        gt = annotations[question_id]
        if gt["question_type"] == "multiple-choice":
            is_correct = evaluate_multiple_choice(gt, answer)
        else:
            is_correct = evaluate_open_form(gt, answer)
        answer["correct"] = is_correct
        answer["label"] = gt["answer"]
        right += is_correct
    accuracy = right / len(predictions) * 100
    results["accuracy"] = accuracy
    return results

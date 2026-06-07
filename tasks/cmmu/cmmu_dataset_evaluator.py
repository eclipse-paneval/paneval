import json
import ast
import re
import numpy as np
from collections import defaultdict
import os.path as osp
from flagevalmm.registry import EVALUATORS
from flagevalmm.evaluator.pre_process import process_multiple_choice
from flagevalmm.evaluator import BaseEvaluator

EVALUATION_SYSTEM_PROMPT = """
You are an expert evaluator specializing in assessing fill-in-the-blank questions in primary school to hight school exams. I will give you a question, the expected correct answer, and a test-taker's response to the question.
You need to understand the given question, compare the standard answer with the provided response, and fill in the following values:
- analysis: If the answer is incomplete or incorrect, you need to give a reason for the error. If the answer is correct, you can leave it blank. The analysis must be a string, not exceeding 500 characters.
- correct: Whether the answer to the question is correct. Return 1 for correct, 0 for incorrect.
The above values should be returned in JSON format. I should be able to directly load the return value into a dict variable using the json.loads function in Python.

Remember, your output should only contain the following json format:
{
"analysis":,
"correct":
}
Be sure to use double backslashes if necessary, not single backslashe.
"""

EVALUATION_USER_TEMPLATE = """
Here is the fill-in-the-blank question:
"{}"

The expected correct answer to this problem:
"{}"

Response to the problem:
"{}"
"""


@EVALUATORS.register_module()
class CmmuEvaluator(BaseEvaluator):
    def __init__(
        self,
        output_dir=None,
        result_file_name=None,
        filter_types=None,
        shift_check=True,
        use_llm_evaluator: bool = False,
        is_clean=True,
        **kwargs,
    ) -> None:
        if use_llm_evaluator:
            from flagevalmm.models import GPT

            self.llm = GPT(
                model_name=kwargs.pop("model_name"),
                api_key=kwargs.pop("api_key"),
                base_url=kwargs.pop("base_url"),
                json_mode=True,
                **kwargs,
            )
            self.evaluate_fill_blank = self.evaluate_fill_blank_llm
        else:
            print("Evaluate by rules")
            self.evaluate_fill_blank = self.evaluate_fill_blank_by_rule
        self.output_dir = output_dir
        self.result_file_name = result_file_name
        if isinstance(filter_types, list):
            self.filter_types = set(filter_types)
        else:
            self.filter_types = None
        self.is_clean = is_clean
        self.shift_check = shift_check
        self.evaluated_ids_count = defaultdict(list)  # only used for circular_eval
        self.difficulty = ["normal", "hard"]
        # difficulty -> ques_type -> split -> accuracy
        self.eval_results = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(lambda: {"correct": 0, "total": 0, "accuracy": 0})
            )
        )

        self.subject_grade_results = defaultdict(
            lambda: {"correct": 0, "total": 0, "accuracy": 0}
        )
        self.position_dict = defaultdict(lambda: defaultdict(int))
        self.options = set(["A", "B", "C", "D"])
        self.results = []

    def evaluate_fill_blank_llm(self, gt, answer):
        prompt = EVALUATION_USER_TEMPLATE.format(
            gt["question_info"], gt["answer"], answer["answer"]
        )
        message = self.llm.build_message(
            query=prompt, system_prompt=EVALUATION_SYSTEM_PROMPT
        )
        print(f"gt: {gt['answer']}\nans: {answer['answer']}")
        try:
            ans = self.llm.infer(chat_messages=message, temperature=0, top_p=1, seed=42)
        except Exception as e:
            print(e)
            return False, answer["answer"]
        try:
            ans_parsed = ast.literal_eval(ans)
            return ans_parsed["correct"], answer["answer"]
        except Exception as e:
            print(e)
            pattern = re.compile(r'"correct"\s*:\s*1')
            match = re.search(pattern, ans)
            if match:
                return True, answer["answer"]
            return False, answer["answer"]

    def maybe_clean_answer(self, answer):
        if not self.is_clean:
            return answer
        if len(answer) == 1:
            return answer
        answer = process_multiple_choice(answer)
        return answer

    def evaluate_mulitple_choice(self, gt, answer):
        cleaned_answer = self.maybe_clean_answer(answer["answer"])
        is_right = gt["answer"] == cleaned_answer[:1]
        if self.shift_check:
            question_id = str(answer["question_id"])
            question_id_base = question_id.split("-")[0]
            self.evaluated_ids_count[question_id_base].append(
                [is_right, cleaned_answer[:1]]
            )
        return is_right, cleaned_answer

    def show(self, content):
        print(content)
        self.results.append(content)

    def print_nested_dict(self, d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                self.show("\t" * indent + str(key))
                self.print_nested_dict(value, indent + 1)
            else:
                self.show("\t" * indent + str(key) + ": " + str(value))

    def collect_one_result(self, gt, answer, is_right, cleaned_answer):
        answer["correct"] = is_right
        answer["label"] = gt["answer"]
        answer["answer_raw"] = answer["answer"]
        answer["answer"] = cleaned_answer
        difficulty = gt["difficulty"]
        ques_type = gt["type"]
        grade = gt["grade_band"]
        subject = gt["subject"]
        split = gt["split"]

        if ques_type == "multiple-choice":
            question_id = str(answer["question_id"])
            question_id_base = question_id.split("-")[0]

            if len(self.evaluated_ids_count[question_id_base]) == len(gt["options"]):
                is_right = len(gt["options"]) == sum(
                    [x[0] for x in self.evaluated_ids_count[question_id_base]]
                )
                if len(gt["options"]) == 4 and not is_right:
                    for x in self.evaluated_ids_count[question_id_base]:
                        if x[1] in self.options:
                            self.position_dict[split][x[1]] += 1
            else:
                # Not finished
                return

        self.eval_results[split][ques_type][difficulty]["correct"] += is_right
        self.eval_results[split][ques_type][difficulty]["total"] += 1
        self.subject_grade_results[subject]["correct"] += is_right
        self.subject_grade_results[subject]["total"] += 1
        self.subject_grade_results[grade]["correct"] += is_right
        self.subject_grade_results[grade]["total"] += 1
        self.subject_grade_results[f"{split}-overall"]["correct"] += is_right
        self.subject_grade_results[f"{split}-overall"]["total"] += 1

    def calculate_accuracy(self):
        for _, v in self.subject_grade_results.items():
            v["accuracy"] = round(v["correct"] / v["total"] * 100, 2)
        for _, v in self.eval_results.items():
            for _, v1 in v.items():
                for _, v2 in v1.items():
                    v2["accuracy"] = round(v2["correct"] / v2["total"] * 100, 2)

        for k in self.subject_grade_results:
            if "overall" in k:
                self.eval_results[k] = self.subject_grade_results[k]
                split = k.split("-")[0]
                position_list = [self.position_dict[split][o] for o in self.options]

                position_prob = np.array(position_list) / sum(position_list) * 100
                bias_rate = np.var(position_prob)
                self.eval_results[k]["bias_rate"] = round(bias_rate, 2)

    def cal_accuracy(self, annotation, answers, target_type=None):
        if target_type is not None:
            self.show(f"\nEvaluate {target_type}")
        else:
            self.show("\nEvaluate all types of questions")
        for answer in answers:
            question_id = str(answer["question_id"])
            if "type" in answer and answer["type"] not in self.filter_types:
                continue
            if question_id not in annotation:
                continue
            gt = annotation[question_id]

            if target_type is not None and gt["type"] != target_type:
                continue
            if gt["type"] == "fill-in-the-blank":
                is_right, cleaned_answer = self.evaluate_fill_blank(gt, answer)
                cleaned_answer = cleaned_answer
            elif gt["type"] == "multiple-choice":
                is_right, cleaned_answer = self.evaluate_mulitple_choice(gt, answer)
            else:
                is_right, cleaned_answer = self.evaluate_multiple_response(gt, answer)
            self.collect_one_result(gt, answer, is_right, cleaned_answer)

    def dump_results(self, judged_answers):
        print("\n\n=======Final Results=======")
        data = "\n".join(self.results)
        print(data)
        with open(
            osp.join(osp.join(self.output_dir, self.result_file_name + ".summary")), "w"
        ) as fout:
            fout.write(data)
        with open(
            osp.join(self.output_dir, self.result_file_name + "_judged.json"), "w"
        ) as fout:
            json.dump(judged_answers, fout, ensure_ascii=False, indent=2)

        json.dump(
            self.eval_results,
            open(
                osp.join(self.output_dir, self.result_file_name + "_result.json"), "w"
            ),
            indent=2,
            ensure_ascii=False,
        )
        json.dump(
            self.subject_grade_results,
            open(
                osp.join(
                    self.output_dir,
                    self.result_file_name + "_subject_grade_results.json",
                ),
                "w",
            ),
            indent=2,
            ensure_ascii=False,
        )

    def process(self, dataset, output_dir, **kwargs):
        """
        Args:
            dataset (Dataset): dataset instance
            answers (list): list of answers
        """
        annotation = dataset.get_annotation()
        dataset_name = dataset.name
        if osp.isfile(output_dir):
            result_file = output_dir
        else:
            result_file = osp.join(output_dir, dataset_name + ".json")
        result_file = (
            output_dir
            if osp.isfile(output_dir)
            else osp.join(output_dir, dataset_name + ".json")
        )
        self.result_file_name = self.result_file_name or dataset_name
        self.output_dir = self.output_dir or osp.dirname(result_file)
        predictions = json.load(open(result_file))
        results_tmp = {}
        predictions, filtered_predictions = self.filter_rejected(
            predictions, results_tmp
        )
        if self.filter_types is None:
            self.cal_accuracy(annotation, predictions)
        else:
            for ques_type in self.filter_types:
                self.cal_accuracy(annotation, predictions, ques_type)
        self.calculate_accuracy()
        self.eval_results.update(results_tmp)
        self.print_nested_dict(self.eval_results)
        self.dump_results(predictions + filtered_predictions)
        return self.eval_results

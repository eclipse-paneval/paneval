config = dict(
    dataset_path="MathLLMs/MathVision",
    split="test",
    processed_dataset_path="MathVision",
    processor="process.py",
)

pre_prompt = """
Please solve the problem and put your answer in one "\\boxed{}". If it is a multiple choice question, only one letter is allowed in the "\\boxed{}".
"""
dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(type="PromptTemplate", pre_prompt=pre_prompt, post_prompt=""),
    config=config,
    name="math_vision_test",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")

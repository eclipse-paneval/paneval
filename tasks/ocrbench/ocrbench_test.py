config = dict(
    dataset_path="echo840/OCRBench",
    split="test",
    processed_dataset_path="OCRBench",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    config=config,
    prompt_template=dict(type="PromptTemplate", post_prompt=""),
    name="ocrbench_test",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")

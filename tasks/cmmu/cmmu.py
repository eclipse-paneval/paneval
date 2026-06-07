register_dataset = {"cmmu_dataset.py": "CmmuDataset"}
register_evaluator = {"cmmu_dataset_evaluator.py": "CmmuEvaluator"}
shift_check = True
filter_types = ["multiple-response", "multiple-choice", "fill-in-the-blank"]

config = dict(
    dataset_path="BAAI/CMMU",
    split="val",
    processed_dataset_path="CMMU",
    processor="process.py",
)

dataset = dict(
    type="CmmuDataset",
    config=config,
    filter_types=filter_types,
    shift_check=shift_check,
    prompt_template=dict(type="PromptTemplate"),
    name="cmmu",
)

evaluator = dict(
    type="CmmuEvaluator",
    shift_check=shift_check,
    filter_types=filter_types,
    use_llm_evaluator=False,
)

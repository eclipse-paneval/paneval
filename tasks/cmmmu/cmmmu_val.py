config = dict(
    dataset_path="lmms-lab/CMMMU",
    split="val",
    processed_dataset_path="CMMMU",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    config=config,
    prompt_template=dict(type="PromptTemplate"),
    name="cmmmu_val",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")

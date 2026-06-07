config = dict(
    dataset_path="lmms-lab/MMMU",
    split="validation",
    processed_dataset_path="MMMU",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(type="PromptTemplate"),
    config=config,
    name="mmmu_val",
)

evaluator = dict(type="MmmuEvaluator")

config = dict(
    dataset_path="MMMU/MMMU_Pro",
    dataset_name="vision",
    split="test",
    processed_dataset_path="MMMU_Pro",
    processor="process_vision.py",
)

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt="Answer with the option letter from the given choices directly. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options.",
    ),
    config=config,
    name="mmmu_pro_vision_test",
)

evaluator = dict(type="MmmuEvaluator")

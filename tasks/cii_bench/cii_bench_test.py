config = dict(
    dataset_path="m-a-p/CII-Bench",
    split="test",
    processed_dataset_path="CII-Bench",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(
        type="PromptTemplate",
        pre_prompt="请根据提供的图片尝试回答下面的单选题。直接回答正确选项，不要包含额外的解释。请使用以下格式：“答案：$LETTER”，其中$LETTER是你认为正确答案的字母。\n",
        post_prompt="答案：",
    ),
    config=config,
    name="cii_bench_test",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")

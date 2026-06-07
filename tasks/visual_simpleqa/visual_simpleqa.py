from evalmm.common.const import EVALMM_API_KEY, EVALMM_BASE_URL

config = dict(
    dataset_path="WYLing/VisualSimpleQA",
    split="data",
    processed_dataset_path="visual_simpleqa",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(type="PromptTemplate", post_prompt=""),
    name="visual_simpleqa",
    config=config,
)


evaluator = dict(
    type="ExtractEvaluator",
    eval_method="simpleqa",
    eval_model_name="gpt-5-mini",
    use_llm_evaluator=True,
    use_cache=True,
    base_url=EVALMM_BASE_URL,
    api_key=EVALMM_API_KEY,
)

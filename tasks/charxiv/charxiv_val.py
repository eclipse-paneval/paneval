from evalmm.common.const import EVALMM_API_KEY, EVALMM_BASE_URL

config = dict(
    dataset_path="~/.cache/evalmm/datasets/CharXiv",
    split="val",
    processed_dataset_path="CharXiv",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    config=config,
    prompt_template=dict(type="PromptTemplate", post_prompt=""),
    name="charxiv_val",
)

evaluator = dict(
    type="BaseEvaluator",
    eval_func="evaluate.py",
    use_llm_evaluator=True,
    use_cache=True,
    base_url=EVALMM_BASE_URL,
    api_key=EVALMM_API_KEY,
    eval_model_name="gpt-5-mini",
    chat_name="charxiv_val",
)

from evalmm.common.const import EVALMM_API_KEY, EVALMM_BASE_URL

config = dict(
    dataset_path="whyu/mm-vet-v2",
    split="test",
    processed_dataset_path="mm-vet-v2",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(type="PromptTemplate", post_prompt=""),
    name="mmvet_v2",
    config=config,
)

evaluator = dict(
    type="BaseEvaluator",
    eval_func="evaluate.py",
    use_llm_evaluator=True,
    use_cache=False,
    base_url=EVALMM_BASE_URL,
    api_key=EVALMM_API_KEY,
    eval_model_name="gpt-5-mini",
    chat_name="mmvet_v2_eval",
)

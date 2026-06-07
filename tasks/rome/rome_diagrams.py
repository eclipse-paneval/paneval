from evalmm.common.const import EVALMM_API_KEY, EVALMM_BASE_URL


register_evaluator = {"rome_evaluator.py": "ROMEEvaluator"}

_base_ = "post_prompt.py"

dataset = dict(
    type="VqaBaseDataset",
    data_root="BAAI/ROME/data",
    anno_file="test-diagrams.json",
    name="rome_diagrams",
)

base_url = EVALMM_BASE_URL
api_key = EVALMM_API_KEY

evaluator = dict(
    type="ROMEEvaluator",
    tracker_type="image_type",
    tracker_subtype="image_subtype",
    use_llm_evaluator=True,
    aggregation_fields=["judgement_response", "raw_answer"],
    eval_model_name="gpt-4.1-mini",
    api_key=api_key,
    base_url=base_url,
)

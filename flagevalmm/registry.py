from mmengine.registry import Registry

DATASETS = Registry("dataset", locations=["flagevalmm.dataset"])

EVALUATORS = Registry("evaluator", locations=["flagevalmm.evaluator"])

PROMPTS = Registry("prompt", locations=["flagevalmm.prompt"])

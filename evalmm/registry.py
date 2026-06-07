from mmengine.registry import Registry

DATASETS = Registry("dataset", locations=["evalmm.dataset"])

EVALUATORS = Registry("evaluator", locations=["evalmm.evaluator"])

PROMPTS = Registry("prompt", locations=["evalmm.prompt"])

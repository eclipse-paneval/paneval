import torch
import json
import os.path as osp
from collections import defaultdict
from typing import List, Dict
from transformers import AutoProcessor
from PIL import Image
from flagevalmm.registry import EVALUATORS
from flagevalmm.common.logger import get_logger
from flagevalmm.common.video_utils import read_video_pyav

logger = get_logger(__name__)

try:
    from mantis.models.idefics2 import Idefics2ForSequenceClassification
except ImportError:
    logger.warning("mantis is not installed, video score evaluator is not available")


REGRESSION_QUERY_PROMPT = """
Suppose you are an expert in judging and evaluating the quality of AI-generated videos,
please watch the following frames of a given video and see the text prompt for generating the video,
then give scores from 5 different dimensions:
(1) visual quality: the quality of the video in terms of clearness, resolution, brightness, and color
(2) temporal consistency, both the consistency of objects or humans and the smoothness of motion or movements
(3) dynamic degree, the degree of dynamic changes
(4) text-to-video alignment, the alignment between the text prompt and the video content
(5) factual consistency, the consistency of the video content with the common-sense and factual knowledge

for each dimension, output a float number from 1.0 to 4.0,
the higher the number is, the better the video performs in that sub-score,
the lowest 1.0 means Bad, the highest 4.0 means Perfect/Real (the video is like a real video)
Here is an output example:
visual quality: 3.2
temporal consistency: 2.7
dynamic degree: 4.0
text-to-video alignment: 2.3
factual consistency: 1.8

For this video, the text prompt is "{text_prompt}",
all the frames of video are as follows:
"""


@EVALUATORS.register_module()
class VideoScoreEvaluator:
    """
    The evaluation method is adapted from the VideoScore project:
    - GitHub: https://github.com/TIGER-AI-Lab/VideoScore
    - Paper: "VideoScore: Building Automatic Metrics to Simulate Fine-grained Human Feedback for Video Generation" (https://arxiv.org/abs/2409.08833)
    """

    def __init__(self, model: str, max_num_frames: int = 16, **kwargs) -> None:
        self.vieco_score_processor = AutoProcessor.from_pretrained(
            model, torch_dtype=torch.bfloat16
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.video_score_model = (
            Idefics2ForSequenceClassification.from_pretrained(
                model, torch_dtype=torch.bfloat16
            )
            .eval()
            .to(device)
        )
        self.max_num_frames = max_num_frames

    def cal_video_score(self, images, info):
        frames = [Image.fromarray(x) for x in images]
        eval_prompt = REGRESSION_QUERY_PROMPT.format(text_prompt=info["prompt"])
        num_image_token = eval_prompt.count("<image>")
        if num_image_token < len(frames):
            eval_prompt += "<image> " * (len(frames) - num_image_token)

        flatten_images = []
        for x in [frames]:
            if isinstance(x, list):
                flatten_images.extend(x)
            else:
                flatten_images.append(x)
        flatten_images = [
            Image.open(x) if isinstance(x, str) else x for x in flatten_images
        ]
        inputs = self.vieco_score_processor(
            text=eval_prompt, images=flatten_images, return_tensors="pt"
        )
        inputs = {k: v.to(self.video_score_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.video_score_model(**inputs)

        logits = outputs.logits
        num_aspects = logits.shape[-1]

        aspect_scores = []
        aspect_names = [
            "video_score_visual_quality",
            "video_score_temporal_consistency",
            "video_score_dynamic_degree",
            "video_score_text-to-video_alignment",
            "video_score_factual_consistency",
        ]
        results = {}
        for i in range(num_aspects):
            aspect_scores.append(round(logits[0, i].item(), 4))
            results[aspect_names[i]] = aspect_scores[-1]
        results["video_score_mean"] = round(logits[0].mean().item(), 4)
        print(results)
        return results

    def get_metric_results(
        self, output_info: List[Dict], output_dir: str, **kwargs
    ) -> Dict:
        metric_results = defaultdict(list)
        for info in output_info:
            image_or_video_path = osp.join(output_dir, info["video_path"])
            images = read_video_pyav(
                video_path=image_or_video_path,
                max_num_frames=self.max_num_frames,
                return_tensors=False,
            )
            video_score_results = self.cal_video_score(images, info)
            for k, v in video_score_results.items():
                metric_results[k].append(v)
                info[k] = v

        results = {}
        for k, v in metric_results.items():
            results[k] = round(sum(v) / len(v), 4)
        return results

    def process(self, dataset, output_dir, **kwargs):
        """
        Args:
            dataset (Dataset): dataset instance
            answers (list): list of answers
        """
        dataset_name = dataset.name
        result_file = osp.join(output_dir, f"{dataset_name}.json")
        output_info = json.load(open(result_file))

        results = self.get_metric_results(output_info, output_dir)
        json.dump(
            results, open(osp.join(output_dir, f"{dataset_name}_result.json"), "w")
        )
        # save evaluation results
        json.dump(
            output_info,
            open(osp.join(output_dir, f"{dataset_name}_evaluated.json"), "w"),
            ensure_ascii=False,
            indent=2,
        )
        print(results)
        return results

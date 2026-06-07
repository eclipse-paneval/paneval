import argparse
from typing import List, Dict
from mmengine.config import Config
from flagevalmm.server.evaluation_server import EvaluationServer
from flagevalmm.common.logger import get_logger
from flagevalmm.server.utils import maybe_register_class, merge_args

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Infer a model")
    parser.add_argument("--config", help="train config file path")
    parser.add_argument("--tasks", nargs="+", help="tasks to run")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--try-run", action="store_true", help="try run mode")
    parser.add_argument("--data-root", type=str, help="data root")
    parser.add_argument("--output-dir", type=str, help="output dir")
    parser.add_argument("--checkpoint", type=str, help="checkpoint path")
    parser.add_argument("--port", type=int, default=5000, help="server port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="server host")
    parser.add_argument("--quiet", action="store_true", help="quiet mode")
    args = parser.parse_args()
    return args


def load_tasks(task_config_files: List[str]) -> Dict[str, Config]:
    config_dict = {}
    for task_config_file in task_config_files:
        cfg = Config.fromfile(task_config_file, lazy_import=False)
        task_name = cfg.dataset.name
        cfg = merge_args(cfg, task_config_file, args)
        maybe_register_class(cfg, task_config_file)
        config_dict[task_name] = cfg
    logger.info(f"Loaded {len(config_dict)} tasks: {config_dict.keys()}")
    return config_dict


if __name__ == "__main__":
    args = parse_args()
    config_dict = load_tasks(args.tasks)
    server = EvaluationServer(
        config_dict,
        output_dir=args.output_dir,
        model_path=args.checkpoint,
        port=args.port,
        host=args.host,
        debug=args.debug,
        quiet=args.quiet,
    )
    server.run()

import argparse
from mmengine.config import Config
import importlib
import os.path as osp
from typing import Optional, Union
from flagevalmm.common import get_logger
from flagevalmm.common.const import FLAGEVALMM_DATASETS_CACHE_DIR

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Process a dataset")
    parser.add_argument("-c", "--config", help="config file path")
    parser.add_argument("-f", "--force", action="store_true", help="force overwrite")
    parser.add_argument(
        "--cache-dir", type=str, default=FLAGEVALMM_DATASETS_CACHE_DIR, help="cache dir"
    )
    args = parser.parse_args()
    return args


class DataPreprocessor:
    def __init__(
        self,
        config: Union[str, Config],
        config_dir: Optional[str] = None,
        force: bool = False,
        cache_dir: str = FLAGEVALMM_DATASETS_CACHE_DIR,
    ) -> None:
        if isinstance(config, str):
            self.cfg = Config.fromfile(config).config
            config_dir = osp.dirname(config)
        else:
            self.cfg = config
        self.force = force
        self.cache_dir = osp.expanduser(cache_dir)
        self.parse_config(config_dir)

    def parse_config(self, config_dir: str) -> None:
        cfg = self.cfg
        cfg.processor = osp.expanduser(cfg.processor)
        if not osp.isabs(cfg.processor):
            cfg.processor = osp.join(config_dir, cfg.processor)

        if cfg.get("processed_dataset_path"):
            processed_dataset_path = osp.expanduser(cfg.processed_dataset_path)
        else:
            processed_dataset_path = osp.join(self.cache_dir, cfg.dataset_name)
        if osp.isabs(processed_dataset_path):
            processed_dataset_path = processed_dataset_path
        else:
            processed_dataset_path = osp.join(self.cache_dir, processed_dataset_path)

        cfg.processed_dataset_path = processed_dataset_path

    def process(self) -> None:
        cfg = self.cfg
        output_dir = osp.join(cfg.processed_dataset_path, cfg.split)
        if osp.exists(output_dir) and not self.force:
            logger.info(f"Processed dataset already exists at {output_dir}")
            return
        spec = importlib.util.spec_from_file_location(
            "process", cfg.processor  # module name  # full path to file
        )
        # Create the module
        module = importlib.util.module_from_spec(spec)
        # Execute the module
        spec.loader.exec_module(module)
        process_func = getattr(module, "process")
        process_func(cfg)
        logger.info(f"Processed dataset {cfg.dataset_path} saved to {output_dir}")


if __name__ == "__main__":
    args = parse_args()

    preprocessor = DataPreprocessor(
        config=args.config, force=args.force, cache_dir=args.cache_dir
    )
    preprocessor.process()

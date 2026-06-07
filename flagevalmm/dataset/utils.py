from typing import Optional
import os.path as osp

from mmengine.config import Config
from flagevalmm.dataset.data_preprocessor import DataPreprocessor
from flagevalmm.common.logger import get_logger

logger = get_logger(__name__)


def get_data_root(
    data_root: Optional[str], config: Config, cache_dir: str, base_dir: str
) -> str:
    if data_root is not None:
        return data_root
    else:
        # get data from cache
        cache_dir = osp.expanduser(cache_dir)
        dataset_name = config.get("dataset_name", "")
        split = config.get("split", "")
        data_root = osp.join(
            cache_dir, config.processed_dataset_path, dataset_name, split
        )
        if not osp.exists(data_root):
            # Download the dataset
            logger.info(f"Downloading and processing dataset {dataset_name}...")
            data_preprocessor = DataPreprocessor(
                config=config, config_dir=base_dir, cache_dir=cache_dir
            )
            data_preprocessor.process()
        return data_root

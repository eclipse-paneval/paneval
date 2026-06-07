import os
import os.path as osp


def process(cfg):
    data_dir = osp.expanduser(cfg.dataset_path)
    os.system(f"git clone https://huggingface.co/datasets/BAAI/CMMU {data_dir}")

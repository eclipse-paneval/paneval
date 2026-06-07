from typing import Optional, Any, Dict, List, Tuple
from torch.utils.data import Dataset
import json
import os
import os.path as osp
from PIL import Image
from flagevalmm.registry import DATASETS
from flagevalmm.common.const import FLAGEVALMM_DATASETS_CACHE_DIR
from flagevalmm.dataset.utils import get_data_root


@DATASETS.register_module()
class RetrievalBaseDataset(Dataset):
    def __init__(
        self,
        *,
        name: str,
        data_root: Optional[str] = None,
        anno_file: Optional[str] = None,
        cache_dir: str = FLAGEVALMM_DATASETS_CACHE_DIR,
        caption_per_image: int = 5,
        config: Optional[dict] = None,
        base_dir: Optional[str] = None,
        debug: bool = False,
        **kwargs,
    ) -> None:
        self.data_root = get_data_root(
            data_root=data_root, config=config, cache_dir=cache_dir, base_dir=base_dir
        )

        anno_file = "data.json" if anno_file is None else anno_file
        self.annotations = json.load(open(osp.join(self.data_root, anno_file)))
        self.name = name
        if debug:
            self.annotations = self.annotations[:160]
        # flatten the caption list

        self.captions = [
            caption
            for annotation in self.annotations
            for caption in annotation["caption"][:caption_per_image]
        ]

    def __getitem__(self, index: int) -> Tuple[Image.Image, List[str]]:
        root = self.data_root
        annotation = self.annotations[index]
        img_path = annotation["img_path"]
        image = Image.open(os.path.join(root, img_path)).convert("RGB")
        return image, annotation["caption"]

    def image_number(self) -> int:
        return len(self.annotations)

    def caption_number(self) -> int:
        return len(self.captions)

    def get_image(self, image_index: int) -> Dict[str, str]:
        assert image_index < self.image_number()
        annotation = self.annotations[image_index]
        return {"img_path": osp.join(self.data_root, annotation["img_path"])}

    def get_caption(self, catpion_index: int) -> Dict[str, str]:
        assert catpion_index < self.caption_number()
        caption = self.captions[catpion_index]
        return {"caption": caption}

    def meta_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "image_number": self.image_number(),
            "caption_number": self.caption_number(),
            "type": "retrieval",
        }

    def get_data(self, index: int, data_type: str) -> Dict[str, str]:
        if data_type == "img":
            return self.get_image(index)
        elif data_type == "text":
            return self.get_caption(index)
        else:
            raise Exception("Invalid data type")

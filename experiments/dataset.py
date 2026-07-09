"""Dataset loading utilities."""
import torch.nn as nn
from torchvision import transforms

from .config import CFG
from .extractors import _registry_info

_IQADS_NAME_MAP = {
    "LIVE": "LIVE",
    "CSIQ": "CSIQ",
    "TID2013": "TID2013",
    "KADID": "KADID-10k",
    "PIPAL": "PIPAL",
}


def get_transform():
    info = _registry_info(CFG.backbone)
    w = info["weights"]
    return w.transforms()


def get_dataset(name: str):
    if name == "AIC4":
        from .custom_datasets import get_aic4_dataset
        return get_aic4_dataset(CFG.dataset_root, transform=get_transform())

    from iqadataset import load_dataset_pytorch
    return load_dataset_pytorch(
        _IQADS_NAME_MAP[name],
        dataset_root=CFG.dataset_root,
        attributes=["dis_img_path", "ref_img_path", "score"],
        transform=get_transform(),
    )

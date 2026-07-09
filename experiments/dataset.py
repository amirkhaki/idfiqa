"""Dataset loading utilities."""
from torchvision import transforms

from .config import CFG

TO_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))
])

_IQADS_NAME_MAP = {
    "LIVE": "LIVE",
    "CSIQ": "CSIQ",
    "TID2013": "TID2013",
    "KADID": "KADID-10k",
    "PIPAL": "PIPAL",
}


def get_dataset(name: str):
    if name == "AIC4":
        from .custom_datasets import get_aic4_dataset
        return get_aic4_dataset(CFG.dataset_root, transform=TO_TENSOR)

    from iqadataset import load_dataset_pytorch
    return load_dataset_pytorch(
        _IQADS_NAME_MAP[name],
        dataset_root=CFG.dataset_root,
        attributes=["dis_img_path", "ref_img_path", "score"],
        transform=TO_TENSOR,
    )

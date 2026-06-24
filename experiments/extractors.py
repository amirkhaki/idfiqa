"""Feature extractor factory functions."""
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor

from .config import _registry_info


def make_single_extractor(backbone: str, feature_layer: str):
    info = _registry_info(backbone)
    m = info["model_fn"]()
    w = info["weights"]
    ext = create_feature_extractor(m, {feature_layer: "features"})
    return ext, w.transforms(), "features"


def make_dual_extractor(backbone: str, feature_layer: str, weight_layer: str):
    info = _registry_info(backbone)
    m = info["model_fn"]()
    w = info["weights"]
    return_nodes = {feature_layer: "features"}
    if weight_layer != feature_layer:
        return_nodes[weight_layer] = "weights"
    else:
        return_nodes[weight_layer + "|alias"] = "weights"
        ext = create_feature_extractor(m, {feature_layer: "features"})

        class _DualAlias(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
            def forward(self, x):
                out = self.inner(x)
                return {"features": out["features"], "weights": out["features"]}

        return _DualAlias(ext), w.transforms()

    ext = create_feature_extractor(m, return_nodes)
    return ext, w.transforms()

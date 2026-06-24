"""Central configuration and backbone registry."""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


def _build_registry() -> Dict[str, dict]:
    reg: Dict[str, dict] = {}

    def _add(name, model_fn, weights_enum, default_feat, default_wt):
        reg[name] = {
            "model_fn": model_fn,
            "weights": weights_enum,
            "default_feature_layer": default_feat,
            "default_weight_layer": default_wt,
        }

    try:
        import torchvision.models as _m

        _add("vgg16",
             lambda: _m.vgg16(weights=_m.VGG16_Weights.IMAGENET1K_V1),
             _m.VGG16_Weights.IMAGENET1K_V1,
             "features.23", "features.19")

        _add("efficientnet_b0",
             lambda: _m.efficientnet_b0(weights=_m.EfficientNet_B0_Weights.IMAGENET1K_V1),
             _m.EfficientNet_B0_Weights.IMAGENET1K_V1,
             "features.6.0.block.1", "features.4.0.block.1")

        _add("efficientnet_b4",
             lambda: _m.efficientnet_b4(weights=_m.EfficientNet_B4_Weights.IMAGENET1K_V1),
             _m.EfficientNet_B4_Weights.IMAGENET1K_V1,
             "features.6.5.block.1", "features.4.2.block.1")

        _add("resnet50",
             lambda: _m.resnet50(weights=_m.ResNet50_Weights.IMAGENET1K_V2),
             _m.ResNet50_Weights.IMAGENET1K_V2,
             "layer3", "layer2")

        _add("convnext_base",
             lambda: _m.convnext_base(weights=_m.ConvNeXt_Base_Weights.IMAGENET1K_V1),
             _m.ConvNeXt_Base_Weights.IMAGENET1K_V1,
             "features.5", "features.3")

        _add("convnext_tiny",
             lambda: _m.convnext_tiny(weights=_m.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
             _m.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
             "features.5", "features.3")

    except ImportError:
        pass

    return reg


BACKBONE_REGISTRY: Dict[str, dict] = _build_registry()


def _registry_info(backbone: str) -> dict:
    if backbone not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{backbone}'. "
            f"Available: {sorted(BACKBONE_REGISTRY)}"
        )
    return BACKBONE_REGISTRY[backbone]


def _is_feature_node(name: str) -> bool:
    return name in [f"features.{d}" for d in range(1, 30, 2)]


_NODE_CACHE: Dict[str, Dict[str, str]] = {}


def get_all_feature_nodes(backbone: str) -> Dict[str, str]:
    if backbone in _NODE_CACHE:
        return _NODE_CACHE[backbone]

    from torchvision.models.feature_extraction import get_graph_node_names

    info = _registry_info(backbone)
    model = info["model_fn"]()
    _, eval_names = get_graph_node_names(model)
    del model

    nodes: Dict[str, str] = {}
    for name in eval_names:
        if _is_feature_node(name):
            nodes[name] = name

    _NODE_CACHE[backbone] = nodes
    return nodes


@dataclass
class Config:
    dataset_root: str = "datasets"
    output_dir: str = "results"
    backbone: str = "vgg16"
    feature_layer: Optional[str] = None
    weight_layer: Optional[str] = None
    percent_features: float = 0.6
    window_size: int = 4
    patch_size: int = 8
    aggregation: str = "max"

    large_ds_threshold: int = 1000
    large_ds_n_samples: int = 1000
    large_ds_n_runs: int = 5
    ablation_patch_sizes: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    ablation_window_sizes: List[int] = field(default_factory=lambda: [2, 4, 6, 8])
    ablation_pf_values: List[float] = field(default_factory=lambda: [0.4, 0.5, 0.6, 0.7, 0.8])
    ablation_aggregations: List[str] = field(default_factory=lambda: [
        "max", "mean", "softmax_0.5", "softmax_1.0", "softmax_2.0", "uniform"
    ])
    all_datasets: List[str] = field(default_factory=lambda: ["LIVE", "CSIQ", "TID2013", "KADID", "PIPAL"])
    ablation_datasets: List[str] = field(default_factory=lambda: ["TID2013"])
    large_datasets: List[str] = field(default_factory=lambda: ["KADID", "PIPAL"])
    complexity_image_size: tuple = (512, 512)
    complexity_n_runs: int = 20
    comparison_backbones: List[str] = field(default_factory=lambda: ["vgg16", "efficientnet_b4"])
    diagnose: bool = False

    def get_feature_layer(self, backbone=None):
        bb = backbone or self.backbone
        if self.feature_layer and bb == self.backbone:
            return self.feature_layer
        return _registry_info(bb)["default_feature_layer"]

    def get_weight_layer(self, backbone=None):
        bb = backbone or self.backbone
        if self.weight_layer and bb == self.backbone:
            return self.weight_layer
        return _registry_info(bb)["default_weight_layer"]

    def candidate_layers(self, backbone=None):
        return get_all_feature_nodes(backbone or self.backbone)


CFG = Config()

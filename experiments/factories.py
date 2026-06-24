"""Model factory functions."""
from .config import CFG
from .extractors import make_single_extractor, make_dual_extractor
from .models import IDFIQA_Baseline, IDFIQA_Causal, WeightedPatchIDFIQA


def make_baseline_model(device, *, backbone=None, feature_layer=None, pf=None, ws=None):
    backbone = backbone or CFG.backbone
    feature_layer = feature_layer or CFG.get_feature_layer(backbone)
    pf = pf if pf is not None else CFG.percent_features
    ws = ws if ws is not None else CFG.window_size
    ext, norm, key = make_single_extractor(backbone, feature_layer)
    return IDFIQA_Baseline(ext, norm, feature_node_key=key,
                           device=device, percent_features_to_keep=pf, window_size=ws)


def make_causal_model(device, *, backbone=None, feature_layer=None,
                      pf=None, ws=None, noise_std=None):
    backbone = backbone or CFG.backbone
    feature_layer = feature_layer or CFG.get_feature_layer(backbone)
    pf = pf if pf is not None else CFG.percent_features
    ws = ws if ws is not None else CFG.window_size
    noise_std = noise_std if noise_std is not None else CFG.noise_std
    ext, norm, key = make_single_extractor(backbone, feature_layer)
    return IDFIQA_Causal(ext, norm, feature_node_key=key,
                         device=device, percent_features_to_keep=pf,
                         window_size=ws, noise_std=noise_std)


def make_patch_model(device, *, backbone=None, feature_layer=None, weight_layer=None,
                     pf=None, ws=None, ps=None, aggregation=None):
    backbone = backbone or CFG.backbone
    feature_layer = feature_layer or CFG.get_feature_layer(backbone)
    weight_layer = weight_layer or CFG.get_weight_layer(backbone)
    pf = pf if pf is not None else CFG.percent_features
    ws = ws if ws is not None else CFG.window_size
    ps = ps if ps is not None else CFG.patch_size
    aggregation = aggregation or CFG.aggregation
    ext, norm = make_dual_extractor(backbone, feature_layer, weight_layer)
    return WeightedPatchIDFIQA(ext, norm, device=device,
                               percent_features_to_keep=pf,
                               window_size=ws, patch_size=ps,
                               aggregation=aggregation)

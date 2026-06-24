"""Model factory functions."""
from .config import CFG
from .extractors import make_single_extractor, make_dual_extractor
from .models import IDFIQA_Baseline, IDFIQA_Causal, WeightedPatchIDFIQA, IDFIQA_SpatialCausal


def make_baseline_model(device, *, backbone=None, feature_layer=None, pf=None, ws=None):
    backbone = backbone or CFG.backbone
    feature_layer = feature_layer or CFG.get_feature_layer(backbone)
    pf = pf if pf is not None else CFG.percent_features
    ws = ws if ws is not None else CFG.window_size
    ext, norm, key = make_single_extractor(backbone, feature_layer)
    return IDFIQA_Baseline(ext, norm, feature_node_key=key,
                           device=device, percent_features_to_keep=pf, window_size=ws)


def make_causal_model(device, *, backbone=None, feature_layer=None,
                      pf=None, ws=None, causal_method=None,
                      max_intensity=None, n_steps=None):
    backbone = backbone or CFG.backbone
    feature_layer = feature_layer or CFG.get_feature_layer(backbone)
    pf = pf if pf is not None else CFG.percent_features
    ws = ws if ws is not None else CFG.window_size
    causal_method = causal_method or CFG.causal_method
    max_intensity = max_intensity if max_intensity is not None else CFG.max_intensity
    n_steps = n_steps if n_steps is not None else CFG.n_steps
    ext, norm, key = make_single_extractor(backbone, feature_layer)
    return IDFIQA_Causal(ext, norm, feature_node_key=key,
                         device=device, percent_features_to_keep=pf,
                         window_size=ws, causal_method=causal_method,
                         max_intensity=max_intensity, n_steps=n_steps)


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


def make_spatial_model(device, *, backbone=None, feature_layer=None,
                       pf=None, ws=None, ps=None, patch_score_method=None,
                       max_intensity=None, n_steps=None):
    backbone = backbone or CFG.backbone
    feature_layer = feature_layer or CFG.get_feature_layer(backbone)
    pf = pf if pf is not None else CFG.percent_features
    ws = ws if ws is not None else CFG.window_size
    ps = ps if ps is not None else CFG.patch_size_spatial
    patch_score_method = patch_score_method or CFG.patch_score_method
    max_intensity = max_intensity if max_intensity is not None else CFG.max_intensity
    n_steps = n_steps if n_steps is not None else CFG.n_steps
    ext, norm, key = make_single_extractor(backbone, feature_layer)
    return IDFIQA_SpatialCausal(ext, norm, feature_node_key=key,
                                device=device, percent_features_to_keep=pf,
                                window_size=ws, patch_size=ps,
                                patch_score_method=patch_score_method,
                                max_intensity=max_intensity, n_steps=n_steps)

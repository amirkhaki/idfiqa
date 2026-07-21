"""Shared helper functions for experiments."""
from typing import Optional

from .config import CFG


def layer_slug(layer: str) -> str:
    """Sanitise a layer node name for use in a filename."""
    return layer.replace(".", "_").replace("|", "_").replace("/", "_")


def run_slug(
    backbone: str,
    feature_layer: str,
    wt_layer: Optional[str] = None,
    percent_features: Optional[float] = None,
    window_size: Optional[int] = None,
    patch_size: Optional[int] = None,
    aggregation: Optional[str] = None,
    sp: Optional[str] = None,
) -> str:
    """Build a compact, deterministic filename slug."""
    slug = backbone + "_" + layer_slug(feature_layer)
    if wt_layer is not None:
        slug += "_wt" + layer_slug(wt_layer)

    pf = percent_features if percent_features is not None else CFG.percent_features
    ws = window_size if window_size is not None else CFG.window_size
    ps = patch_size if patch_size is not None else CFG.patch_size
    agg = aggregation if aggregation is not None else CFG.aggregation

    slug += f"_pf{pf}"
    slug += f"_ws{ws}"
    slug += f"_ps{ps}"
    slug += f"_agg{agg.replace('.', '_')}"
    if sp:
        slug += f"_sp{sp}"

    return slug


def run_config(
    backbone: str,
    feature_layer: str,
    wt_layer: Optional[str] = None,
    percent_features: Optional[float] = None,
    window_size: Optional[int] = None,
    patch_size: Optional[int] = None,
    aggregation: Optional[str] = None,
    sp: Optional[str] = None,
) -> dict:
    """Return the full run configuration as a flat dict."""
    return {
        "backbone": backbone,
        "feature_layer": feature_layer,
        "weight_layer": wt_layer,
        "percent_features": percent_features if percent_features is not None else CFG.percent_features,
        "window_size": window_size if window_size is not None else CFG.window_size,
        "patch_size": patch_size if patch_size is not None else CFG.patch_size,
        "aggregation": aggregation if aggregation is not None else CFG.aggregation,
        "sp": sp,
    }

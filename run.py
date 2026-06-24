"""CLI entrypoint for IDFIQA experiments."""
import os
import argparse
from datetime import datetime

import torch

from experiments import CFG, BACKBONE_REGISTRY
from experiments.experiments import (
    experiment_layer_search,
    experiment_baseline,
    experiment_causal,
    experiment_patch_weighted,
    experiment_ablation_weight_source,
    experiment_ablation_aggregation,
    experiment_ablation_patch_window,
    experiment_ablation_percent_features,
    experiment_backbone_comparison,
    experiment_geometric_robustness,
    experiment_complexity,
    compute_delta_table,
)

EXPERIMENT_CHOICES = [
    "all", "layer_search", "baseline", "causal", "patch_weighted",
    "ablation_weight_source", "ablation_aggregation", "ablation_patch_window",
    "ablation_percent_features", "backbone_comparison", "geometric_robustness",
    "complexity",
]


def main():
    _backbone_choices = sorted(BACKBONE_REGISTRY) or ["vgg16", "efficientnet_b4"]

    parser = argparse.ArgumentParser(
        description="Resumable IDFIQA experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--experiment", choices=EXPERIMENT_CHOICES, default="all")
    parser.add_argument("--datasets", nargs="+", default=None, choices=CFG.all_datasets)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default=CFG.output_dir)
    parser.add_argument("--backbone", type=str, default=CFG.backbone, choices=_backbone_choices)
    parser.add_argument("--backbones", nargs="+", default=None, choices=_backbone_choices)
    parser.add_argument("--feature-layer", type=str, default=None)
    parser.add_argument("--weight-layer", type=str, default=None)
    parser.add_argument("--percent-features", type=float, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--aggregation", type=str, default=None)
    parser.add_argument("--noise-std", type=float, default=None)
    parser.add_argument("--diagnose", action="store_true")
    args = parser.parse_args()

    CFG.output_dir = args.output_dir
    CFG.backbone = args.backbone
    if args.backbones:
        CFG.comparison_backbones = args.backbones
    if args.feature_layer:
        CFG.feature_layer = args.feature_layer
    if args.weight_layer:
        CFG.weight_layer = args.weight_layer
    if args.percent_features is not None:
        CFG.percent_features = args.percent_features
    if args.window_size is not None:
        CFG.window_size = args.window_size
    if args.patch_size is not None:
        CFG.patch_size = args.patch_size
    if args.aggregation:
        CFG.aggregation = args.aggregation
    if args.noise_std is not None:
        CFG.noise_std = args.noise_std
    CFG.diagnose = args.diagnose

    os.makedirs(CFG.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device:     {device}")
    print(f"Backbone:   {CFG.backbone}")
    print(f"Registered: {sorted(BACKBONE_REGISTRY)}")
    print(f"Comparison: {CFG.comparison_backbones}")
    print(f"Output:     {os.path.abspath(CFG.output_dir)}")
    print(f"Time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    exp = args.experiment
    all_ds = args.datasets or CFG.all_datasets
    ablation_ds = args.datasets or CFG.ablation_datasets
    nw = args.num_workers
    force = args.force

    baseline_results = {}
    patch_results = {}

    if exp in ("layer_search"):
        experiment_layer_search(all_ds, nw, force, device)

    if exp in ("all", "baseline"):
        baseline_results = experiment_baseline(all_ds, nw, force, device)

    if exp in ("all", "causal"):
        experiment_causal(all_ds, nw, force, device)

    if exp in ("all", "patch_weighted"):
        patch_results = experiment_patch_weighted(all_ds, nw, force, device)

    if exp == "all" and baseline_results and patch_results:
        compute_delta_table(baseline_results, patch_results)

    if exp in ("all", "ablation_weight_source"):
        experiment_ablation_weight_source(ablation_ds, nw, force, device)

    if exp in ("all", "ablation_aggregation"):
        experiment_ablation_aggregation(ablation_ds, nw, force, device)

    if exp in ("all", "ablation_patch_window"):
        experiment_ablation_patch_window(ablation_ds, nw, force, device)

    if exp in ("all", "ablation_percent_features"):
        experiment_ablation_percent_features(ablation_ds, nw, force, device)

    if exp in ("all", "backbone_comparison"):
        experiment_backbone_comparison(all_ds, nw, force, device)

    if exp in ("all", "geometric_robustness"):
        experiment_geometric_robustness(all_ds, nw, force, device)

    if exp in ("all", "complexity"):
        experiment_complexity(force, device)

    print(f"\nDone. All outputs in: {os.path.abspath(CFG.output_dir)}/")


if __name__ == "__main__":
    main()

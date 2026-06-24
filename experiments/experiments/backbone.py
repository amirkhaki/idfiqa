"""Backbone comparison experiment."""
from ..config import CFG, BACKBONE_REGISTRY
from ..extractors import make_dual_extractor
from ..models import WeightedPatchIDFIQA
from ..utils import out_path, save_json, already_done, compute_metrics
from ..evaluation import run_evaluation
from .helpers import run_slug, run_config


def experiment_backbone_comparison(datasets, num_workers, force, device):
    print("\n=== Phase 2: Backbone Comparison ===")
    backbones = CFG.comparison_backbones
    print(f"  Backbones: {backbones}")
    summary = {}

    for bb in backbones:
        summary[bb] = {}
        feat_layer = CFG.get_feature_layer(bb)
        wt_layer = CFG.get_weight_layer(bb)
        print(f"  {bb:25s}  feat={feat_layer}  wt={wt_layer}")

        for ds in datasets:
            ext, norm = make_dual_extractor(bb, feat_layer, wt_layer)
            model = WeightedPatchIDFIQA(ext, norm, device=device,
                                        percent_features_to_keep=CFG.percent_features,
                                        window_size=CFG.window_size,
                                        patch_size=CFG.patch_size).eval()
            slug = run_slug(bb, feat_layer, wt_layer)
            csv_name = f"backbone_{slug}_{ds}.csv"
            if already_done(csv_name, force):
                with open(out_path(csv_name), newline="") as f:
                    rows = list(csv.DictReader(f))
                srcc, plcc = compute_metrics([float(r["score"]) for r in rows],
                                             [float(r["mos_label"]) for r in rows])
                print(f"  {bb:25s}  {ds:10s}  SRCC={srcc:.4f}  PLCC={plcc:.4f}  (cached)")
            else:
                srcc, plcc, _, _ = run_evaluation(model, ds, csv_name,
                                                  num_workers=num_workers, force=force,
                                                  desc=f"Backbone {bb}/{ds}")
                print(f"  {bb:25s}  {ds:10s}  SRCC={srcc:.4f}  PLCC={plcc:.4f}")
            summary[bb][ds] = {"srcc": srcc, "plcc": plcc,
                                **run_config(bb, feat_layer, wt_layer)}

    save_json(summary, "phase2_backbone_comparison.json")
    print(f"  Summary → {out_path('phase2_backbone_comparison.json')}")
    return summary

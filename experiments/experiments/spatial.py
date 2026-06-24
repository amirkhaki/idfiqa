"""Spatial causal patch selection experiment."""
import csv
from ..config import CFG
from ..factories import make_spatial_model
from ..utils import out_path, save_json, already_done, compute_metrics
from ..evaluation import run_evaluation
from .helpers import run_slug, run_config


def experiment_spatial(datasets, num_workers, force, device):
    print("\n=== Phase 1b: Spatial Causal Patch Selection ===")
    feat_layer = CFG.get_feature_layer()
    base_slug = run_slug(CFG.backbone, feat_layer)
    slug = f"{base_slug}_spatial_ps{CFG.patch_size_spatial}_{CFG.patch_score_method}"
    cfg_dict = run_config(CFG.backbone, feat_layer)
    cfg_dict["patch_size_spatial"] = CFG.patch_size_spatial
    cfg_dict["patch_score_method"] = CFG.patch_score_method
    cfg_dict["max_intensity"] = CFG.max_intensity
    cfg_dict["n_steps"] = CFG.n_steps
    print(f"  backbone={CFG.backbone}  feat={feat_layer}"
          f"  ps={CFG.patch_size_spatial}  method={CFG.patch_score_method}"
          f"  max_intensity={CFG.max_intensity}  n_steps={CFG.n_steps}")
    summary_file = f"phase1b_spatial_{slug}_summary.json"
    results = {}

    for ds in datasets:
        model = make_spatial_model(device, feature_layer=feat_layer).eval()
        csv_name = f"spatial_{slug}_{ds}.csv"

        if already_done(csv_name, force):
            with open(out_path(csv_name), newline="") as f:
                rows = list(csv.DictReader(f))
            srcc, plcc = compute_metrics([float(r["score"]) for r in rows],
                                         [float(r["mos_label"]) for r in rows])
            print(f"  {ds:10s}  SRCC={srcc:.4f}  PLCC={plcc:.4f}  (cached)")
        else:
            srcc, plcc, _, _ = run_evaluation(model, ds, csv_name,
                                              num_workers=num_workers, force=force,
                                              desc=f"Spatial/{ds}")
            print(f"  {ds:10s}  SRCC={srcc:.4f}  PLCC={plcc:.4f}")
        results[ds] = {"srcc": srcc, "plcc": plcc, **cfg_dict}

    save_json(results, summary_file)
    print(f"  Summary → {out_path(summary_file)}")
    return results

"""Patch-weighted experiment."""
from ..config import CFG
from ..factories import make_patch_model
from ..utils import out_path, save_json, already_done, compute_metrics
from ..evaluation import run_evaluation
from .helpers import run_slug, run_config


def experiment_patch_weighted(datasets, num_workers, force, device):
    print("\n=== Phase 1: Patch-Weighted IDFIQA ===")
    feat_layer = CFG.get_feature_layer()
    wt_layer = CFG.get_weight_layer()
    slug = run_slug(CFG.backbone, feat_layer, wt_layer)
    cfg_dict = run_config(CFG.backbone, feat_layer, wt_layer)
    print(f"  backbone={CFG.backbone}  feat={feat_layer}  wt={wt_layer}"
          f"  pf={CFG.percent_features}  ws={CFG.window_size}"
          f"  ps={CFG.patch_size}  agg={CFG.aggregation}")
    summary_file = f"phase1_patch_{slug}_summary.json"
    results = {}

    for ds in datasets:
        model = make_patch_model(device,
                                 feature_layer=feat_layer,
                                 weight_layer=wt_layer).eval()
        csv_name = f"patch_weighted_{slug}_{ds}.csv"

        if already_done(csv_name, force):
            with open(out_path(csv_name), newline="") as f:
                rows = list(csv.DictReader(f))
            srcc, plcc = compute_metrics([float(r["score"]) for r in rows],
                                         [float(r["mos_label"]) for r in rows])
            print(f"  {ds:10s}  SRCC={srcc:.4f}  PLCC={plcc:.4f}  (cached)")
        else:
            srcc, plcc, _, _ = run_evaluation(model, ds, csv_name,
                                              num_workers=num_workers, force=force,
                                              desc=f"PatchWeighted/{ds}")
            print(f"  {ds:10s}  SRCC={srcc:.4f}  PLCC={plcc:.4f}")
        results[ds] = {"srcc": srcc, "plcc": plcc, **cfg_dict}

    save_json(results, summary_file)
    print(f"  Summary → {out_path(summary_file)}")
    return results

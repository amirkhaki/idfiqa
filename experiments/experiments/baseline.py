"""Baseline experiment."""
import csv

from ..config import CFG
from ..factories import make_baseline_model
from ..utils import out_path, save_json, already_done, compute_metrics
from ..evaluation import run_evaluation
from .helpers import run_slug, run_config


def experiment_baseline(datasets, num_workers, force, device):
    print("\n=== Phase 1: No-Patch Baseline (ICCKE) ===")
    feat_layer = CFG.get_feature_layer()
    slug = run_slug(CFG.backbone, feat_layer)
    cfg_dict = run_config(CFG.backbone, feat_layer)
    print(f"  backbone={CFG.backbone}  feat={feat_layer}"
          f"  pf={CFG.percent_features}  ws={CFG.window_size}")
    summary_file = f"phase1_baseline_{slug}_summary.json"
    results = {}

    for ds in datasets:
        model = make_baseline_model(device, feature_layer=feat_layer).eval()
        csv_name = f"baseline_{slug}_{ds}.csv"

        if already_done(csv_name, force):
            with open(out_path(csv_name), newline="") as f:
                rows = list(csv.DictReader(f))
            srcc, plcc = compute_metrics([float(r["score"]) for r in rows],
                                         [float(r["mos_label"]) for r in rows])
            print(f"  {ds:10s}  SRCC={srcc:.4f}  PLCC={plcc:.4f}  (cached)")
        else:
            srcc, plcc, _, _ = run_evaluation(model, ds, csv_name,
                                              num_workers=num_workers, force=force,
                                              desc=f"Baseline/{ds}")
            print(f"  {ds:10s}  SRCC={srcc:.4f}  PLCC={plcc:.4f}")
        results[ds] = {"srcc": srcc, "plcc": plcc, **cfg_dict}

    save_json(results, summary_file)
    print(f"  Summary → {out_path(summary_file)}")
    return results


def compute_delta_table(baseline_results, patch_results):
    print("\n=== Phase 1: Delta Table (patch - baseline) ===")
    delta = {}
    rows = [["dataset", "baseline_srcc", "baseline_plcc",
             "patch_srcc", "patch_plcc", "delta_srcc", "delta_plcc"]]
    for ds in baseline_results:
        if ds not in patch_results:
            continue
        bs = baseline_results[ds]["srcc"]
        bp = baseline_results[ds]["plcc"]
        ps = patch_results[ds]["srcc"]
        pp = patch_results[ds]["plcc"]
        delta[ds] = {"delta_srcc": ps - bs, "delta_plcc": pp - bp}
        rows.append([ds, bs, bp, ps, pp, ps - bs, pp - bp])
        print(f"  {ds:10s}  ΔSRCC={ps-bs:+.4f}  ΔPLCC={pp-bp:+.4f}")
    save_json(delta, "phase1_delta_table.json")
    with open(out_path("phase1_delta_table.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  Delta → {out_path('phase1_delta_table.json')}")
    return delta

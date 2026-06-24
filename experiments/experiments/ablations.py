"""Ablation experiments."""
import itertools

from ..config import CFG
from ..factories import make_patch_model
from ..utils import out_path, save_json, already_done, compute_metrics
from ..evaluation import run_evaluation
from .helpers import run_slug, run_config


def experiment_ablation_weight_source(datasets, num_workers, force, device):
    print("\n=== Phase 2 Ablation: Weight Map Source Layer ===")
    feat_layer = CFG.get_feature_layer()
    candidates = CFG.candidate_layers()
    print(f"  backbone={CFG.backbone}  feat={feat_layer}  sweeping {len(candidates)} weight layers"
          f"  pf={CFG.percent_features}  ws={CFG.window_size}  ps={CFG.patch_size}  agg={CFG.aggregation}")
    summary = {}

    for ds in datasets:
        summary[ds] = {}
        for label, wlayer in candidates.items():
            model = make_patch_model(device,
                                     feature_layer=feat_layer,
                                     weight_layer=wlayer).eval()
            slug = run_slug(CFG.backbone, feat_layer, wlayer)
            csv_name = f"abl_wsrc_{slug}_{ds}.csv"
            if already_done(csv_name, force):
                with open(out_path(csv_name), newline="") as f:
                    rows = list(csv.DictReader(f))
                srcc, plcc = compute_metrics([float(r["score"]) for r in rows],
                                             [float(r["mos_label"]) for r in rows])
                print(f"  wt={label:50s}  {ds:10s}  SRCC={srcc:.4f}  PLCC={plcc:.4f}  (cached)")
            else:
                srcc, plcc, _, _ = run_evaluation(model, ds, csv_name,
                                                  num_workers=num_workers, force=force,
                                                  desc=f"WtSrc {label}/{ds}")
                print(f"  wt={label:50s}  {ds:10s}  SRCC={srcc:.4f}  PLCC={plcc:.4f}")
            summary[ds][label] = {"layer": wlayer, "srcc": srcc, "plcc": plcc,
                                   **run_config(CFG.backbone, feat_layer, wlayer)}

    base_slug = run_slug(CFG.backbone, feat_layer)
    out_name = f"phase2_ablation_weight_source_{base_slug}.json"
    save_json(summary, out_name)
    print(f"  Summary → {out_path(out_name)}")
    return summary


def experiment_ablation_aggregation(datasets, num_workers, force, device):
    print("\n=== Phase 2 Ablation: Aggregation Function ===")
    feat_layer = CFG.get_feature_layer()
    wt_layer = CFG.get_weight_layer()
    print(f"  backbone={CFG.backbone}  feat={feat_layer}  wt={wt_layer}"
          f"  pf={CFG.percent_features}  ws={CFG.window_size}  ps={CFG.patch_size}"
          f"  sweeping aggregations: {CFG.ablation_aggregations}")
    summary = {}

    for agg in CFG.ablation_aggregations:
        summary[agg] = {}
        for ds in datasets:
            model = make_patch_model(device,
                                     feature_layer=feat_layer,
                                     weight_layer=wt_layer,
                                     aggregation=agg).eval()
            slug = run_slug(CFG.backbone, feat_layer, wt_layer, aggregation=agg)
            csv_name = f"abl_agg_{slug}_{ds}.csv"
            if already_done(csv_name, force):
                with open(out_path(csv_name), newline="") as f:
                    rows = list(csv.DictReader(f))
                srcc, plcc = compute_metrics([float(r["score"]) for r in rows],
                                             [float(r["mos_label"]) for r in rows])
                print(f"  agg={agg:14s}  {ds:10s}  SRCC={srcc:.4f}  PLCC={plcc:.4f}  (cached)")
            else:
                srcc, plcc, _, _ = run_evaluation(model, ds, csv_name,
                                                  num_workers=num_workers, force=force,
                                                  desc=f"Agg {agg}/{ds}")
                print(f"  agg={agg:14s}  {ds:10s}  SRCC={srcc:.4f}  PLCC={plcc:.4f}")
            summary[agg][ds] = {"srcc": srcc, "plcc": plcc,
                                **run_config(CFG.backbone, feat_layer, wt_layer, aggregation=agg)}

    base_slug = run_slug(CFG.backbone, feat_layer, wt_layer)
    out_name = f"phase2_ablation_aggregation_{base_slug}.json"
    save_json(summary, out_name)
    print(f"  Summary → {out_path(out_name)}")
    return summary


def experiment_ablation_patch_window(datasets, num_workers, force, device):
    print("\n=== Phase 2 Ablation: Patch Size × Window Size ===")
    feat_layer = CFG.get_feature_layer()
    wt_layer = CFG.get_weight_layer()
    print(f"  backbone={CFG.backbone}  feat={feat_layer}  wt={wt_layer}"
          f"  pf={CFG.percent_features}  agg={CFG.aggregation}"
          f"  sweeping patch_sizes={CFG.ablation_patch_sizes}  window_sizes={CFG.ablation_window_sizes}")
    summary = {}
    combos = list(itertools.product(CFG.ablation_patch_sizes,
                                    CFG.ablation_window_sizes))
    for ps, ws in combos:
        key = f"ps{ps}_ws{ws}"
        summary[key] = {}
        for ds in datasets:
            model = make_patch_model(device,
                                     feature_layer=feat_layer,
                                     weight_layer=wt_layer,
                                     ps=ps, ws=ws).eval()
            slug = run_slug(CFG.backbone, feat_layer, wt_layer,
                            window_size=ws, patch_size=ps)
            csv_name = f"abl_psws_{slug}_{ds}.csv"
            if already_done(csv_name, force):
                with open(out_path(csv_name), newline="") as f:
                    rows = list(csv.DictReader(f))
                srcc, plcc = compute_metrics([float(r["score"]) for r in rows],
                                             [float(r["mos_label"]) for r in rows])
                print(f"  {key:12s}  {ds:10s}  SRCC={srcc:.4f}  PLCC={plcc:.4f}  (cached)")
            else:
                srcc, plcc, _, _ = run_evaluation(model, ds, csv_name,
                                                  num_workers=num_workers, force=force,
                                                  desc=f"PS/WS {key}/{ds}")
                print(f"  {key:12s}  {ds:10s}  SRCC={srcc:.4f}  PLCC={plcc:.4f}")
            summary[key][ds] = {"srcc": srcc, "plcc": plcc,
                                **run_config(CFG.backbone, feat_layer, wt_layer,
                                             window_size=ws, patch_size=ps)}

    base_slug = run_slug(CFG.backbone, feat_layer, wt_layer)
    out_name = f"phase2_ablation_patch_window_{base_slug}.json"
    save_json(summary, out_name)
    print(f"  Summary → {out_path(out_name)}")
    return summary


def experiment_ablation_percent_features(datasets, num_workers, force, device):
    print("\n=== Phase 2 Ablation: Variance-Guided Channel Threshold ===")
    feat_layer = CFG.get_feature_layer()
    wt_layer = CFG.get_weight_layer()
    print(f"  backbone={CFG.backbone}  feat={feat_layer}  wt={wt_layer}"
          f"  ws={CFG.window_size}  ps={CFG.patch_size}  agg={CFG.aggregation}"
          f"  sweeping pf_values={CFG.ablation_pf_values}")
    summary = {}

    for pf in CFG.ablation_pf_values:
        key = f"pf{pf}"
        summary[key] = {}
        for ds in datasets:
            model = make_patch_model(device,
                                     feature_layer=feat_layer,
                                     weight_layer=wt_layer,
                                     pf=pf).eval()
            slug = run_slug(CFG.backbone, feat_layer, wt_layer,
                            percent_features=pf)
            csv_name = f"abl_pf_{slug}_{ds}.csv"
            if already_done(csv_name, force):
                with open(out_path(csv_name), newline="") as f:
                    rows = list(csv.DictReader(f))
                srcc, plcc = compute_metrics([float(r["score"]) for r in rows],
                                             [float(r["mos_label"]) for r in rows])
                print(f"  pf={pf:.1f}  {ds:10s}  SRCC={srcc:.4f}  PLCC={plcc:.4f}  (cached)")
            else:
                srcc, plcc, _, _ = run_evaluation(model, ds, csv_name,
                                                  num_workers=num_workers, force=force,
                                                  desc=f"PF {pf}/{ds}")
                print(f"  pf={pf:.1f}  {ds:10s}  SRCC={srcc:.4f}  PLCC={plcc:.4f}")
            summary[key][ds] = {"srcc": srcc, "plcc": plcc,
                                **run_config(CFG.backbone, feat_layer, wt_layer,
                                             percent_features=pf)}

    base_slug = run_slug(CFG.backbone, feat_layer, wt_layer)
    out_name = f"phase2_ablation_percent_features_{base_slug}.json"
    save_json(summary, out_name)
    print(f"  Summary → {out_path(out_name)}")
    return summary

"""Layer search experiment."""
import os
import json

from ..config import CFG
from ..factories import make_baseline_model, make_patch_model
from ..utils import out_path, save_json
from ..evaluation import run_evaluation
from .helpers import run_slug

_LAYER_SEARCH_JSON = "layer_search_results.json"


def experiment_layer_search(datasets, num_workers, force, device):
    print("\n=== Layer Search ===")
    print(f"  Backbone: {CFG.backbone}")

    candidates = CFG.candidate_layers()
    candidate_items = list(candidates.items())
    print(f"  Discovered {len(candidate_items)} candidate layers from backbone graph")

    json_path = out_path(_LAYER_SEARCH_JSON)
    if os.path.exists(json_path) and not force:
        with open(json_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    for ds in datasets:
        key = f"{CFG.backbone}/{ds}"
        print(f"\n  Dataset: {ds}  backbone: {CFG.backbone}")

        if key not in all_results:
            all_results[key] = {
                "stage1_feature_layer": {},
                "stage2_weight_layer": {},
                "best": {}
            }
        ds_results = all_results[key]

        print(f"  Stage 1 — feature layer search ({len(candidate_items)} candidates)")
        stage1 = ds_results["stage1_feature_layer"]

        for label, layer_str in candidate_items:
            if label in stage1 and not force:
                srcc = stage1[label]["srcc"]
                plcc = stage1[label]["plcc"]
                print(f"    [cached]  feat={label:50s}  SRCC={srcc:.4f}  PLCC={plcc:.4f}")
                continue

            safe_label = label.replace(".", "_")
            csv_name = f"lsearch_{CFG.backbone}_{ds}_feat_{safe_label}.csv"
            try:
                model = make_baseline_model(device,
                                            backbone=CFG.backbone,
                                            feature_layer=layer_str).eval()
                srcc, plcc, _, _ = run_evaluation(
                    model, ds, csv_name,
                    num_workers=num_workers, force=force,
                    desc=f"LayerSearch feat={label}/{ds}")
            except Exception as exc:
                print(f"    SKIP  feat={label:50s}  ({type(exc).__name__}: {exc})")
                continue

            print(f"    feat={label:50s}  SRCC={srcc:.4f}  PLCC={plcc:.4f}")
            stage1[label] = {"layer": layer_str, "srcc": srcc, "plcc": plcc}

            save_json(all_results, _LAYER_SEARCH_JSON)

        if not stage1:
            print("  WARNING: no valid feature layers found; skipping this dataset.")
            continue

        best_feat_label = max(stage1, key=lambda lb: abs(stage1[lb]["srcc"]))
        best_feat_layer = stage1[best_feat_label]["layer"]
        print(f"  → Best feature layer: {best_feat_label} ({best_feat_layer})"
              f"  SRCC={stage1[best_feat_label]['srcc']:.4f}")

        print(f"  Stage 2 — weight layer search ({len(candidate_items)} candidates)"
              f"  [feature={best_feat_label}]")
        stage2 = ds_results["stage2_weight_layer"]

        for label, layer_str in candidate_items:
            if label in stage2 and not force:
                srcc = stage2[label]["srcc"]
                plcc = stage2[label]["plcc"]
                print(f"    [cached]  wt={label:50s}  SRCC={srcc:.4f}  PLCC={plcc:.4f}")
                continue

            safe_label = label.replace(".", "_")
            safe_feat_label = best_feat_label.replace(".", "_")
            csv_name = (f"lsearch_{CFG.backbone}_{ds}_wt_{safe_label}"
                        f"_feat_{safe_feat_label}.csv")
            try:
                model = make_patch_model(device,
                                         backbone=CFG.backbone,
                                         feature_layer=best_feat_layer,
                                         weight_layer=layer_str).eval()
                srcc, plcc, _, _ = run_evaluation(
                    model, ds, csv_name,
                    num_workers=num_workers, force=force,
                    desc=f"LayerSearch wt={label}/{ds}")
            except Exception as exc:
                print(f"    SKIP  wt={label:50s}  ({type(exc).__name__}: {exc})")
                continue

            print(f"    wt={label:50s}  SRCC={srcc:.4f}  PLCC={plcc:.4f}")
            stage2[label] = {"layer": layer_str, "srcc": srcc, "plcc": plcc}

            save_json(all_results, _LAYER_SEARCH_JSON)

        if not stage2:
            best_wt_label = best_feat_label
            best_wt_layer = best_feat_layer
        else:
            best_wt_label = max(stage2, key=lambda lb: stage2[lb]["srcc"])
            best_wt_layer = stage2[best_wt_label]["layer"]

        print(f"  → Best weight layer:  {best_wt_label} ({best_wt_layer})"
              f"  SRCC={stage2[best_wt_label]['srcc']:.4f}" if stage2 else
              f"  → Best weight layer:  {best_wt_label} (same as feature, no candidates)")

        ds_results["best"] = {
            "feature_layer": best_feat_layer,
            "feature_label": best_feat_label,
            "weight_layer": best_wt_layer,
            "weight_label": best_wt_label,
        }
        save_json(all_results, _LAYER_SEARCH_JSON)

    print(f"\n  Full layer search results → {out_path(_LAYER_SEARCH_JSON)}")
    return all_results

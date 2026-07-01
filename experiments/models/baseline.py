"""Variance-guided channel selection baseline."""
import csv
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CFG
from ..extractors import make_single_extractor
from ..utils import out_path, save_json, already_done, compute_metrics
from ..evaluation import run_evaluation
from ..experiments.helpers import run_slug, run_config
from ..registry import DefaultExperiment, register_experiment


class IDFIQA_Baseline(nn.Module):
    """
    No-patch IDFIQA (ICCKE baseline).
    Global gram matrices + windowed SSIM-like score.
    Uses percent_features_to_keep for variance-guided channel selection.
    """

    def __init__(self, feature_extractor, normalize,
                 feature_node_key="features",
                 device=None,
                 percent_features_to_keep=0.6,
                 window_size=4,
                 xi=1e-8):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor.to(self.device).eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.normalize = normalize
        self.feature_node_key = feature_node_key
        self.pf = percent_features_to_keep
        self.ws = window_size
        self.xi = xi

    def _features(self, img):
        out = self.feature_extractor(self.normalize(img.to(self.device)))
        return out[self.feature_node_key] if isinstance(out, dict) else out

    @staticmethod
    def _gram(feat):
        n, c, h, w = feat.shape
        f = feat.view(n, c, h * w)
        return torch.bmm(f, f.transpose(1, 2)) / (h * w)

    def _select_channels(self, feat_ref, feat_dist):
        n, c, h, w = feat_ref.shape
        k = max(1, int(c * self.pf))
        var = torch.var(feat_ref, dim=(2, 3), unbiased=False)
        _, idx = torch.topk(var, k, dim=1)
        idx_r = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        s_ref = torch.gather(feat_ref, 1, idx_r)
        _, _, hd, wd = feat_dist.shape
        idx_d = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, hd, wd)
        s_dist = torch.gather(feat_dist, 1, idx_d)
        return s_ref, s_dist

    def forward(self, ref, dist):
        fr = self._features(ref)
        fd = self._features(dist)
        fr, fd = self._select_channels(fr, fd)
        gr = self._gram(fr)
        gd = self._gram(fd)
        gr_u = F.unfold(gr.unsqueeze(1), kernel_size=self.ws, stride=1).transpose(1, 2)
        gd_u = F.unfold(gd.unsqueeze(1), kernel_size=self.ws, stride=1).transpose(1, 2)
        vr = torch.var(gr_u, dim=2, unbiased=False)
        vd = torch.var(gd_u, dim=2, unbiased=False)
        mr = torch.mean(gr_u, dim=2, keepdim=True)
        md = torch.mean(gd_u, dim=2, keepdim=True)
        cov = torch.mean((gr_u - mr) * (gd_u - md), dim=2)
        local = (2 * cov + self.xi) / (vr + vd + self.xi)
        return local.mean(dim=1)


def _build_baseline_model(device, backbone=None, feature_layer=None, pf=None, ws=None):
    backbone = backbone or CFG.backbone
    feature_layer = feature_layer or CFG.get_feature_layer(backbone)
    pf = pf if pf is not None else CFG.percent_features
    ws = ws if ws is not None else CFG.window_size
    ext, norm, key = make_single_extractor(backbone, feature_layer)
    return IDFIQA_Baseline(ext, norm, feature_node_key=key,
                           device=device, percent_features_to_keep=pf, window_size=ws)


@register_experiment
class BaselineExperiment(DefaultExperiment):
    name = "baseline"
    description = "Variance-guided channel selection baseline (ICCKE)"
    summary_prefix = "baseline"

    def add_arguments(self, parser):
        parser.add_argument("--backbone", type=str, default=CFG.backbone)
        parser.add_argument("--feature-layer", type=str, default=None)
        parser.add_argument("--percent-features", type=float, default=CFG.percent_features)
        parser.add_argument("--window-size", type=int, default=CFG.window_size)
        parser.add_argument("--backbones", nargs="+", default=None)
        sub = parser.add_subparsers(dest="action")
        sub.add_parser("layer_search", help="Search best feature layer")
        sub.add_parser("backbone_comparison", help="Compare backbones")

    def build_model(self, device, args):
        return _build_baseline_model(device, backbone=args.backbone,
                                     feature_layer=args.feature_layer,
                                     pf=args.percent_features,
                                     ws=args.window_size)

    def run(self, args, datasets, num_workers, force, device):
        if args.action == "layer_search":
            self._layer_search(args, datasets, num_workers, force, device)
        elif args.action == "backbone_comparison":
            self._backbone_comparison(args, datasets, num_workers, force, device)
        else:
            super().run(args, datasets, num_workers, force, device)

    def _layer_search(self, args, datasets, num_workers, force, device):
        from .patch import _build_patch_model
        backbone = args.backbone
        print("\n=== Layer Search ===")
        print(f"  Backbone: {backbone}")
        candidates = CFG.candidate_layers(backbone) if backbone != CFG.backbone else CFG.candidate_layers()
        candidate_items = list(candidates.items())
        print(f"  Discovered {len(candidate_items)} candidate layers from backbone graph")

        json_path = out_path("layer_search_results.json")
        if os.path.exists(json_path) and not force:
            with open(json_path) as f:
                all_results = json.load(f)
        else:
            all_results = {}

        for ds in datasets:
            key = f"{backbone}/{ds}"
            print(f"\n  Dataset: {ds}  backbone: {backbone}")

            if key not in all_results:
                all_results[key] = {
                    "stage1_feature_layer": {},
                    "stage2_weight_layer": {},
                    "best": {}
                }
            ds_results = all_results[key]

            print(f"  Stage 1 -- feature layer search ({len(candidate_items)} candidates)")
            stage1 = ds_results["stage1_feature_layer"]

            for label, layer_str in candidate_items:
                if label in stage1 and not force:
                    srcc = stage1[label]["srcc"]
                    plcc = stage1[label]["plcc"]
                    print(f"    [cached]  feat={label:50s}  SRCC={srcc:.4f}  PLCC={plcc:.4f}")
                    continue

                safe_label = label.replace(".", "_")
                csv_name = f"lsearch_{backbone}_{ds}_feat_{safe_label}.csv"
                try:
                    model = _build_baseline_model(device, backbone=backbone,
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
                save_json(all_results, "layer_search_results.json")

            if not stage1:
                print("  WARNING: no valid feature layers found; skipping this dataset.")
                continue

            best_feat_label = max(stage1, key=lambda lb: abs(stage1[lb]["srcc"]))
            best_feat_layer = stage1[best_feat_label]["layer"]
            print(f"  -> Best feature layer: {best_feat_label} ({best_feat_layer})"
                  f"  SRCC={stage1[best_feat_label]['srcc']:.4f}")

            print(f"  Stage 2 -- weight layer search ({len(candidate_items)} candidates)"
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
                csv_name = (f"lsearch_{backbone}_{ds}_wt_{safe_label}"
                            f"_feat_{safe_feat_label}.csv")
                try:
                    model = _build_patch_model(device, backbone=backbone,
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
                save_json(all_results, "layer_search_results.json")

            if not stage2:
                best_wt_label = best_feat_label
                best_wt_layer = best_feat_layer
            else:
                best_wt_label = max(stage2, key=lambda lb: stage2[lb]["srcc"])
                best_wt_layer = stage2[best_wt_label]["layer"]

            print(f"  -> Best weight layer:  {best_wt_label} ({best_wt_layer})"
                  f"  SRCC={stage2[best_wt_label]['srcc']:.4f}" if stage2 else
                  f"  -> Best weight layer:  {best_wt_label} (same as feature, no candidates)")

            ds_results["best"] = {
                "feature_layer": best_feat_layer,
                "feature_label": best_feat_label,
                "weight_layer": best_wt_layer,
                "weight_label": best_wt_label,
            }
            save_json(all_results, "layer_search_results.json")

        print(f"\n  Full layer search results -> {out_path('layer_search_results.json')}")
        return all_results

    def _backbone_comparison(self, args, datasets, num_workers, force, device):
        from ..config import BACKBONE_REGISTRY
        from ..extractors import make_dual_extractor
        from .patch import WeightedPatchIDFIQA
        backbones = args.backbones if args.backbones else CFG.comparison_backbones
        print("\n=== Backbone Comparison ===")
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

        save_json(summary, "backbone_comparison.json")
        print(f"  Summary -> {out_path('backbone_comparison.json')}")
        return summary

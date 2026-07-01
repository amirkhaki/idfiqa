"""Patch-based weighted IDFIQA model."""
import csv
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CFG
from ..extractors import make_dual_extractor
from ..utils import out_path, save_json, already_done, compute_metrics
from ..evaluation import run_evaluation
from ..helpers import run_slug, run_config
from ..registry import DefaultExperiment, register_experiment


class WeightedPatchIDFIQA(nn.Module):
    """
    Patch-based weighted IDFIQA (new method).
    The extractor must return a dict with 'features' and 'weights' keys.
    Aggregation: 'max' | 'mean' | 'softmax_<temp>' | 'uniform'
    """

    def __init__(self, feature_extractor, normalize,
                 device=None,
                 percent_features_to_keep=0.6,
                 window_size=4,
                 patch_size=8,
                 aggregation="max",
                 xi=1e-8):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor.to(self.device).eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.normalize = normalize
        self.pf = percent_features_to_keep
        self.ws = window_size
        self.ps = patch_size
        self.aggregation = aggregation
        self.xi = xi

    def _extract(self, img):
        from ..config import CFG
        out = self.feature_extractor(self.normalize(img.to(self.device)))
        feat = out["features"]
        wt = out["weights"]
        if CFG.diagnose and not getattr(self, "_diag_printed", False):
            self._diag_printed = True
            same_obj = feat is wt
            same_ptr = feat.data_ptr() == wt.data_ptr()
            shape_msg = (f"  diff_norm={(feat-wt).norm():.6g}"
                         if feat.shape == wt.shape else "  shapes differ")
            print(f"  [diag] feat={tuple(feat.shape)} norm={feat.norm():.4g}"
                  f"  wt={tuple(wt.shape)} norm={wt.norm():.4g}"
                  f"  same_obj={same_obj}  same_ptr={same_ptr}{shape_msg}")
        return feat, wt

    @staticmethod
    def _gram(feat):
        n, c, h, w = feat.shape
        f = feat.view(n, c, h * w)
        return torch.bmm(f, f.transpose(1, 2)) / (h * w)

    def _patch_score(self, pr, pd):
        n, c, ph, pw = pr.shape
        k = max(1, int(c * self.pf))
        var = torch.var(pr, dim=(2, 3), unbiased=False)
        _, idx = torch.topk(var, k, dim=1)
        idx_e = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, ph, pw)
        pr = torch.gather(pr, 1, idx_e)
        pd = torch.gather(pd, 1, idx_e)
        gr = self._gram(pr)
        gd = self._gram(pd)
        gr_u = F.unfold(gr.unsqueeze(1), kernel_size=self.ws, stride=1).transpose(1, 2)
        gd_u = F.unfold(gd.unsqueeze(1), kernel_size=self.ws, stride=1).transpose(1, 2)
        vr = torch.var(gr_u, dim=2, unbiased=False)
        vd = torch.var(gd_u, dim=2, unbiased=False)
        mr = gr_u.mean(dim=2, keepdim=True)
        md = gd_u.mean(dim=2, keepdim=True)
        cov = ((gr_u - mr) * (gd_u - md)).mean(dim=2)
        return ((2 * cov + self.xi) / (vr + vd + self.xi)).mean(dim=1)

    def _all_patch_scores(self, fr, fd, nh, nw, ph, pw):
        B, C, H, W = fr.shape
        P = nh * nw

        pr_flat = F.unfold(fr, kernel_size=(ph, pw), stride=(ph, pw))
        pd_flat = F.unfold(fd, kernel_size=(ph, pw), stride=(ph, pw))

        pr = pr_flat.permute(0, 2, 1).reshape(B * P, C, ph, pw)
        pd = pd_flat.permute(0, 2, 1).reshape(B * P, C, ph, pw)

        k = max(1, int(C * self.pf))
        var = torch.var(pr, dim=(2, 3), unbiased=False)
        _, idx = torch.topk(var, k, dim=1)
        idx_e = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, ph, pw)
        pr = torch.gather(pr, 1, idx_e)
        pd = torch.gather(pd, 1, idx_e)

        def gram_batched(x):
            bp, k, h, w = x.shape
            f = x.view(bp, k, h * w)
            return torch.bmm(f, f.transpose(1, 2)) / (h * w)

        gr = gram_batched(pr)
        gd = gram_batched(pd)

        gr_u = F.unfold(gr.unsqueeze(1), kernel_size=self.ws, stride=1).transpose(1, 2)
        gd_u = F.unfold(gd.unsqueeze(1), kernel_size=self.ws, stride=1).transpose(1, 2)
        vr = torch.var(gr_u, dim=2, unbiased=False)
        vd = torch.var(gd_u, dim=2, unbiased=False)
        mr = gr_u.mean(dim=2, keepdim=True)
        md = gd_u.mean(dim=2, keepdim=True)
        cov = ((gr_u - mr) * (gd_u - md)).mean(dim=2)
        scores_flat = ((2 * cov + self.xi) / (vr + vd + self.xi)).mean(dim=1)
        return scores_flat.view(B, P)

    def forward(self, ref, dist):
        fr, _ = self._extract(ref)
        fd, _ = self._extract(dist)

        n, c, h, w = fr.shape

        nh = max(1, h // self.ps)
        nw = max(1, w // self.ps)
        ph = h // nh
        pw = w // nw

        fr = fr[:, :, :nh * ph, :nw * pw]
        fd = fd[:, :, :nh * ph, :nw * pw]

        if ph < self.ws or pw < self.ws:
            return self._patch_score(fr, fd)

        scores = self._all_patch_scores(fr, fd, nh, nw, ph, pw)
        return torch.mean(scores, dim=1)

    def _patch_weight_scalar(self, wmap_patches):
        if self.aggregation == "uniform":
            return torch.ones(wmap_patches.shape[:2],
                              device=wmap_patches.device,
                              dtype=wmap_patches.dtype)
        elif self.aggregation == "max":
            return wmap_patches.max(dim=2).values
        elif self.aggregation.startswith("softmax"):
            temp = float(self.aggregation.split("_")[1]) if "_" in self.aggregation else 1.0
            patch_weights = torch.softmax(wmap_patches / temp, dim=2)
            return (patch_weights * wmap_patches).sum(dim=2)
        else:
            return wmap_patches.mean(dim=2)

    def _normalise_weights(self, raw_weights):
        if self.aggregation == "uniform":
            return torch.ones_like(raw_weights) / raw_weights.shape[1]
        elif self.aggregation == "max":
            idx = raw_weights.argmax(dim=1, keepdim=True)
            return torch.zeros_like(raw_weights).scatter_(1, idx, 1.0)
        elif self.aggregation.startswith("softmax"):
            temp = float(self.aggregation.split("_")[1]) if "_" in self.aggregation else 1.0
            return torch.softmax(raw_weights / temp, dim=1)
        else:
            return raw_weights / (raw_weights.sum(dim=1, keepdim=True) + 1e-8)


def _build_patch_model(device, backbone=None, feature_layer=None, weight_layer=None,
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


@register_experiment
class PatchExperiment(DefaultExperiment):
    name = "patch"
    description = "Patch-weighted IDFIQA with ablation studies"
    summary_prefix = "patch"

    def add_arguments(self, parser):
        parser.add_argument("--backbone", type=str, default=CFG.backbone)
        parser.add_argument("--feature-layer", type=str, default=None)
        parser.add_argument("--weight-layer", type=str, default=None)
        parser.add_argument("--percent-features", type=float, default=CFG.percent_features)
        parser.add_argument("--window-size", type=int, default=CFG.window_size)
        parser.add_argument("--patch-size", type=int, default=CFG.patch_size)
        parser.add_argument("--aggregation", type=str, default=CFG.aggregation)
        sub = parser.add_subparsers(dest="action")
        sub.add_parser("ablation_weight_source", help="Sweep weight map source layer")
        sub.add_parser("ablation_aggregation", help="Sweep aggregation functions")
        sub.add_parser("ablation_patch_window", help="Sweep patch size x window size")
        sub.add_parser("ablation_percent_features", help="Sweep channel threshold")

    def slug_args(self, args):
        backbone = args.backbone
        feat_layer = args.feature_layer or CFG.get_feature_layer(backbone)
        wt_layer = args.weight_layer or CFG.get_weight_layer(backbone)
        return {"backbone": backbone, "feature_layer": feat_layer, "wt_layer": wt_layer}

    def build_model(self, device, args):
        return _build_patch_model(device, backbone=args.backbone,
                                  feature_layer=args.feature_layer,
                                  weight_layer=args.weight_layer,
                                  pf=args.percent_features,
                                  ws=args.window_size,
                                  ps=args.patch_size,
                                  aggregation=args.aggregation)

    def run(self, args, datasets, num_workers, force, device):
        action = args.action
        if action == "ablation_weight_source":
            self._ablation_weight_source(args, datasets, num_workers, force, device)
        elif action == "ablation_aggregation":
            self._ablation_aggregation(args, datasets, num_workers, force, device)
        elif action == "ablation_patch_window":
            self._ablation_patch_window(args, datasets, num_workers, force, device)
        elif action == "ablation_percent_features":
            self._ablation_percent_features(args, datasets, num_workers, force, device)
        else:
            super().run(args, datasets, num_workers, force, device)

    def _ablation_weight_source(self, args, datasets, num_workers, force, device):
        backbone = args.backbone
        feat_layer = args.feature_layer or CFG.get_feature_layer(backbone)
        candidates = CFG.candidate_layers(backbone) if backbone != CFG.backbone else CFG.candidate_layers()
        print(f"  backbone={backbone}  feat={feat_layer}  sweeping {len(candidates)} weight layers"
              f"  pf={args.percent_features}  ws={args.window_size}  ps={args.patch_size}  agg={args.aggregation}")
        summary = {}
        for ds in datasets:
            summary[ds] = {}
            for label, wlayer in candidates.items():
                model = _build_patch_model(device, backbone=backbone,
                                           feature_layer=feat_layer,
                                           weight_layer=wlayer,
                                           pf=args.percent_features,
                                           ws=args.window_size,
                                           ps=args.patch_size,
                                           aggregation=args.aggregation).eval()
                slug = run_slug(backbone, feat_layer, wlayer)
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
                                       **run_config(backbone, feat_layer, wlayer)}
        base_slug = run_slug(backbone, feat_layer)
        out_name = f"ablation_weight_source_{base_slug}.json"
        save_json(summary, out_name)
        print(f"  Summary -> {out_path(out_name)}")
        return summary

    def _ablation_aggregation(self, args, datasets, num_workers, force, device):
        backbone = args.backbone
        feat_layer = args.feature_layer or CFG.get_feature_layer(backbone)
        wt_layer = args.weight_layer or CFG.get_weight_layer(backbone)
        print(f"  backbone={backbone}  feat={feat_layer}  wt={wt_layer}"
              f"  pf={args.percent_features}  ws={args.window_size}  ps={args.patch_size}"
              f"  sweeping aggregations: {CFG.ablation_aggregations}")
        summary = {}
        for agg in CFG.ablation_aggregations:
            summary[agg] = {}
            for ds in datasets:
                model = _build_patch_model(device, backbone=backbone,
                                           feature_layer=feat_layer,
                                           weight_layer=wt_layer,
                                           pf=args.percent_features,
                                           ws=args.window_size,
                                           ps=args.patch_size,
                                           aggregation=agg).eval()
                slug = run_slug(backbone, feat_layer, wt_layer, aggregation=agg)
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
                                    **run_config(backbone, feat_layer, wt_layer, aggregation=agg)}
        base_slug = run_slug(backbone, feat_layer, wt_layer)
        out_name = f"ablation_aggregation_{base_slug}.json"
        save_json(summary, out_name)
        print(f"  Summary -> {out_path(out_name)}")
        return summary

    def _ablation_patch_window(self, args, datasets, num_workers, force, device):
        backbone = args.backbone
        feat_layer = args.feature_layer or CFG.get_feature_layer(backbone)
        wt_layer = args.weight_layer or CFG.get_weight_layer(backbone)
        print(f"  backbone={backbone}  feat={feat_layer}  wt={wt_layer}"
              f"  pf={args.percent_features}  agg={args.aggregation}"
              f"  sweeping patch_sizes={CFG.ablation_patch_sizes}  window_sizes={CFG.ablation_window_sizes}")
        summary = {}
        combos = list(itertools.product(CFG.ablation_patch_sizes, CFG.ablation_window_sizes))
        for ps, ws in combos:
            key = f"ps{ps}_ws{ws}"
            summary[key] = {}
            for ds in datasets:
                model = _build_patch_model(device, backbone=backbone,
                                           feature_layer=feat_layer,
                                           weight_layer=wt_layer,
                                           pf=args.percent_features,
                                           ws=ws, ps=ps,
                                           aggregation=args.aggregation).eval()
                slug = run_slug(backbone, feat_layer, wt_layer,
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
                                    **run_config(backbone, feat_layer, wt_layer,
                                                 window_size=ws, patch_size=ps)}
        base_slug = run_slug(backbone, feat_layer, wt_layer)
        out_name = f"ablation_patch_window_{base_slug}.json"
        save_json(summary, out_name)
        print(f"  Summary -> {out_path(out_name)}")
        return summary

    def _ablation_percent_features(self, args, datasets, num_workers, force, device):
        backbone = args.backbone
        feat_layer = args.feature_layer or CFG.get_feature_layer(backbone)
        wt_layer = args.weight_layer or CFG.get_weight_layer(backbone)
        print(f"  backbone={backbone}  feat={feat_layer}  wt={wt_layer}"
              f"  ws={args.window_size}  ps={args.patch_size}  agg={args.aggregation}"
              f"  sweeping pf_values={CFG.ablation_pf_values}")
        summary = {}
        for pf in CFG.ablation_pf_values:
            key = f"pf{pf}"
            summary[key] = {}
            for ds in datasets:
                model = _build_patch_model(device, backbone=backbone,
                                           feature_layer=feat_layer,
                                           weight_layer=wt_layer,
                                           pf=pf,
                                           ws=args.window_size,
                                           ps=args.patch_size,
                                           aggregation=args.aggregation).eval()
                slug = run_slug(backbone, feat_layer, wt_layer,
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
                                    **run_config(backbone, feat_layer, wt_layer,
                                                 percent_features=pf)}
        base_slug = run_slug(backbone, feat_layer, wt_layer)
        out_name = f"ablation_percent_features_{base_slug}.json"
        save_json(summary, out_name)
        print(f"  Summary -> {out_path(out_name)}")
        return summary

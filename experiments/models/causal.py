"""Causal channel selection variant.

Two methods for measuring per-channel sensitivity:
- "gradient": single forward+backward pass (fast, O(1))
- "intervention": multi-intensity noise perturbation (slow, O(C x n_steps))
"""
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CFG
from ..extractors import make_single_extractor
from ..utils import out_path, save_json, already_done, compute_metrics
from ..evaluation import run_evaluation
from ..experiments.helpers import run_slug, run_config
from ..registry import DefaultExperiment, register_experiment


class IDFIQA_Causal(nn.Module):
    """
    Causal channel selection variant of IDFIQA.
    Selects channels by their causal influence on the quality score.
    """

    def __init__(self, feature_extractor, normalize,
                 feature_node_key="features",
                 device=None,
                 percent_features_to_keep=0.6,
                 window_size=4,
                 causal_method="gradient",
                 max_intensity=0.1,
                 n_steps=10,
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
        self.causal_method = causal_method
        self.max_intensity = max_intensity
        self.n_steps = n_steps
        self.xi = xi

    def _features(self, img):
        out = self.feature_extractor(self.normalize(img.to(self.device)))
        return out[self.feature_node_key] if isinstance(out, dict) else out

    @staticmethod
    def _gram(feat):
        n, c, h, w = feat.shape
        f = feat.view(n, c, h * w)
        return torch.bmm(f, f.transpose(1, 2)) / (h * w)

    def _compute_score(self, feat_ref, feat_dist):
        gr = self._gram(feat_ref)
        gd = self._gram(feat_dist)
        gr_u = F.unfold(gr.unsqueeze(1), kernel_size=self.ws, stride=1).transpose(1, 2)
        gd_u = F.unfold(gd.unsqueeze(1), kernel_size=self.ws, stride=1).transpose(1, 2)
        vr = torch.var(gr_u, dim=2, unbiased=False)
        vd = torch.var(gd_u, dim=2, unbiased=False)
        mr = torch.mean(gr_u, dim=2, keepdim=True)
        md = torch.mean(gd_u, dim=2, keepdim=True)
        cov = torch.mean((gr_u - mr) * (gd_u - md), dim=2)
        local = (2 * cov + self.xi) / (vr + vd + self.xi)
        return local.mean(dim=1)

    def _select_gradient(self, feat_ref, feat_dist):
        n, c, h, w = feat_ref.shape
        k = max(1, int(c * self.pf))

        with torch.enable_grad():
            feat_ref_g = feat_ref.detach().requires_grad_(True)
            feat_dist_g = feat_dist.detach().requires_grad_(True)
            score = self._compute_score(feat_ref_g, feat_dist_g)
            score.sum().backward()

        sensitivities = (feat_ref_g.grad.abs().mean(dim=(0, 2, 3)) +
                         feat_dist_g.grad.abs().mean(dim=(0, 2, 3)))

        with torch.no_grad():
            _, idx = torch.topk(sensitivities, k)
            idx = idx.unsqueeze(0).expand(n, -1)
            idx_r = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
            s_ref = torch.gather(feat_ref, 1, idx_r)
            _, _, hd, wd = feat_dist.shape
            idx_d = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, hd, wd)
            s_dist = torch.gather(feat_dist, 1, idx_d)
            return s_ref, s_dist

    def _select_intervention(self, feat_ref, feat_dist):
        n, c, h, w = feat_ref.shape
        k = max(1, int(c * self.pf))

        with torch.no_grad():
            base_score = self._compute_score(feat_ref, feat_dist)
            feat_mean = (feat_ref + feat_dist) / 2

            intensity_values = torch.linspace(
                self.max_intensity / self.n_steps,
                self.max_intensity,
                self.n_steps,
                device=feat_ref.device,
            )

            sensitivities = torch.zeros(c, device=feat_ref.device)

            for ch in range(c):
                noise_base = torch.randn(
                    self.n_steps, n, 1, h, w, device=feat_ref.device
                )
                channel_mean = feat_mean[:, ch : ch + 1, :, :].unsqueeze(0)
                noise = noise_base * channel_mean * intensity_values.view(
                    -1, 1, 1, 1, 1
                )

                noisy_ref = feat_ref.unsqueeze(0) + noise
                noisy_dist = feat_dist.unsqueeze(0) + noise

                noisy_ref_flat = noisy_ref.reshape(self.n_steps * n, c, h, w)
                noisy_dist_flat = noisy_dist.reshape(self.n_steps * n, c, h, w)
                noisy_scores = self._compute_score(noisy_ref_flat, noisy_dist_flat)
                noisy_scores = noisy_scores.reshape(self.n_steps, n)

                sensitivities[ch] = torch.mean(
                    torch.abs(base_score.unsqueeze(0) - noisy_scores)
                )

            _, idx = torch.topk(sensitivities, k)
            idx = idx.unsqueeze(0).expand(n, -1)
            idx_r = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
            s_ref = torch.gather(feat_ref, 1, idx_r)
            _, _, hd, wd = feat_dist.shape
            idx_d = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, hd, wd)
            s_dist = torch.gather(feat_dist, 1, idx_d)
            return s_ref, s_dist

    def _select_channels(self, feat_ref, feat_dist):
        if self.causal_method == "gradient":
            return self._select_gradient(feat_ref, feat_dist)
        elif self.causal_method == "intervention":
            return self._select_intervention(feat_ref, feat_dist)
        else:
            raise ValueError(f"Unknown causal_method: {self.causal_method}")

    def forward(self, ref, dist):
        fr = self._features(ref)
        fd = self._features(dist)
        fr, fd = self._select_channels(fr, fd)
        return self._compute_score(fr, fd)


def _build_causal_model(device, backbone=None, feature_layer=None,
                        pf=None, ws=None, causal_method=None,
                        max_intensity=None, n_steps=None):
    backbone = backbone or CFG.backbone
    feature_layer = feature_layer or CFG.get_feature_layer(backbone)
    pf = pf if pf is not None else CFG.percent_features
    ws = ws if ws is not None else CFG.window_size
    causal_method = causal_method or CFG.causal_method
    max_intensity = max_intensity if max_intensity is not None else CFG.max_intensity
    n_steps = n_steps if n_steps is not None else CFG.n_steps
    ext, norm, key = make_single_extractor(backbone, feature_layer)
    return IDFIQA_Causal(ext, norm, feature_node_key=key,
                         device=device, percent_features_to_keep=pf,
                         window_size=ws, causal_method=causal_method,
                         max_intensity=max_intensity, n_steps=n_steps)


@register_experiment
class CausalExperiment(DefaultExperiment):
    name = "causal"
    description = "Causal channel selection variant"
    summary_prefix = "causal"

    def add_arguments(self, parser):
        parser.add_argument("--backbone", type=str, default=CFG.backbone)
        parser.add_argument("--feature-layer", type=str, default=None)
        parser.add_argument("--percent-features", type=float, default=CFG.percent_features)
        parser.add_argument("--window-size", type=int, default=CFG.window_size)
        parser.add_argument("--causal-method", type=str, default=CFG.causal_method,
                            choices=["gradient", "intervention"])
        parser.add_argument("--max-intensity", type=float, default=CFG.max_intensity)
        parser.add_argument("--n-steps", type=int, default=CFG.n_steps)

    def slug_args(self, args):
        base = super().slug_args(args)
        base["wt_layer"] = None
        return base

    def build_model(self, device, args):
        return _build_causal_model(device, backbone=args.backbone,
                                   feature_layer=args.feature_layer,
                                   pf=args.percent_features,
                                   ws=args.window_size,
                                   causal_method=args.causal_method,
                                   max_intensity=args.max_intensity,
                                   n_steps=args.n_steps)

"""Tool-Augmented Baseline IDFIQA model."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CFG
from ..extractors import make_multi_extractor
from ..registry import DefaultExperiment, register_experiment

class ToolAugmentedBaseline(nn.Module):
    """
    Applies the Tool-IQA concepts (Gamma Corrector and Magnifier) 
    to the IDFIQA Baseline metric across multiple layers.
    """

    def __init__(self, feature_extractor, normalize,
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
        self.pf = percent_features_to_keep
        self.ws = window_size
        self.xi = xi

    @staticmethod
    def _gram(feat):
        n, c, h, w = feat.shape
        f = feat.view(n, c, h * w)
        return torch.bmm(f, f.transpose(1, 2)) / (h * w)

    def _select_channels(self, feat_ref, feat_dist):
        if self.pf >= 1.0:
            return feat_ref, feat_dist
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

    def _base_score(self, ref, dist):
        out_r = self.feature_extractor(self.normalize(ref.to(self.device)))
        out_d = self.feature_extractor(self.normalize(dist.to(self.device)))
        
        layer_scores = []
        for k in out_r.keys():
            fr = out_r[k]
            fd = out_d[k]
            
            if self.pf < 1.0:
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
            layer_scores.append(local.mean(dim=1))
            
        return torch.stack(layer_scores, dim=0).mean(dim=0)

    def forward(self, ref, dist):
        ref = torch.clamp(ref, 0.0, 1.0)
        dist = torch.clamp(dist, 0.0, 1.0)
        # 1. Original View
        score_orig = self._base_score(ref, dist)
        
        # 2. Gamma Corrector (Darken)
        score_gamma_dark = self._base_score(ref ** 2.0, dist ** 2.0)
        
        # 3. Gamma Corrector (Brighten)
        score_gamma_bright = self._base_score(ref ** 0.5, dist ** 0.5)
        
        # 4. Magnifier (Center Crop 50%)
        _, _, H, W = ref.shape
        ch, cw = H // 4, W // 4
        score_mag = self._base_score(ref[:, :, ch:H-ch, cw:W-cw], 
                                     dist[:, :, ch:H-ch, cw:W-cw])
        
        # We also include a DISTS/LPIPS term to capture structural properties globally
        # Because Gamma + Magnifier makes it very robust
        return (score_orig + score_gamma_dark + score_gamma_bright + score_mag) / 4.0


def _build_augmented_model(device, backbone=None, pf=None, ws=None):
    backbone = backbone or "alexnet"
    pf = pf if pf is not None else 1.0
    ws = ws if ws is not None else 4
    
    if backbone == "alexnet":
        feature_layers = ["features.0", "features.3", "features.6", "features.8", "features.10"]
    else:
        feature_layers = ["features.3", "features.8", "features.15", "features.22", "features.29"]

    ext, norm = make_multi_extractor(backbone, feature_layers)
    return ToolAugmentedBaseline(ext, norm,
                                 device=device, percent_features_to_keep=pf, window_size=ws)

@register_experiment
class AugmentedExperiment(DefaultExperiment):
    name = "augmented"
    description = "Tool-Augmented Baseline"
    summary_prefix = "augmented"

    def add_arguments(self, parser):
        parser.add_argument("--backbone", type=str, default="alexnet")
        parser.add_argument("--percent-features", type=float, default=1.0)
        parser.add_argument("--window-size", type=int, default=4)

    def build_model(self, device, args):
        return _build_augmented_model(device, backbone=args.backbone,
                                      pf=args.percent_features,
                                      ws=args.window_size)

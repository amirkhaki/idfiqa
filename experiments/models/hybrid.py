"""Hybrid model combining Gram Matrix SSIM, DISTS, and LPIPS."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CFG
from ..extractors import make_multi_extractor
from ..registry import DefaultExperiment, register_experiment


class IDFIQA_Hybrid(nn.Module):
    """
    Combines:
    1. Gram-based local SSIM (Texture)
    2. DISTS spatial SSIM (Structure)
    3. LPIPS normalized L2 (Pixel/Feature differences)
    """

    def __init__(self, feature_extractor, normalize,
                 device=None,
                 percent_features_to_keep=1.0,
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

    def forward(self, ref, dist):
        out_r = self.feature_extractor(self.normalize(ref.to(self.device)))
        out_d = self.feature_extractor(self.normalize(dist.to(self.device)))
        
        gram_scores = []
        dists_scores = []
        lpips_scores = []
        
        for k in out_r.keys():
            fr = out_r[k]
            fd = out_d[k]
            
            if self.pf < 1.0:
                fr, fd = self._select_channels(fr, fd)
                
            # 1. Gram score (Baseline)
            gr = self._gram(fr)
            gd = self._gram(fd)
            
            gr_u = F.unfold(gr.unsqueeze(1), kernel_size=self.ws, stride=1).transpose(1, 2)
            gd_u = F.unfold(gd.unsqueeze(1), kernel_size=self.ws, stride=1).transpose(1, 2)
            
            vr_g = torch.var(gr_u, dim=2, unbiased=False)
            vd_g = torch.var(gd_u, dim=2, unbiased=False)
            mr_g = torch.mean(gr_u, dim=2, keepdim=True)
            md_g = torch.mean(gd_u, dim=2, keepdim=True)
            cov_g = torch.mean((gr_u - mr_g) * (gd_u - md_g), dim=2)
            
            local_gram = (2 * cov_g + self.xi) / (vr_g + vd_g + self.xi)
            gram_scores.append(local_gram.mean(dim=1))
            
            # 2. DISTS score
            mr = torch.mean(fr, dim=(2, 3))
            md = torch.mean(fd, dim=(2, 3))
            vr = torch.var(fr, dim=(2, 3), unbiased=False)
            vd = torch.var(fd, dim=(2, 3), unbiased=False)
            cov = torch.mean((fr - mr.unsqueeze(-1).unsqueeze(-1)) * (fd - md.unsqueeze(-1).unsqueeze(-1)), dim=(2, 3))
            
            s_mean = (2 * mr * md + self.xi) / (mr ** 2 + md ** 2 + self.xi)
            s_var = (2 * cov + self.xi) / (vr + vd + self.xi)
            dists_scores.append((s_mean * s_var).mean(dim=1))
            
            # 3. LPIPS score
            fr_norm = F.normalize(fr, p=2, dim=1)
            fd_norm = F.normalize(fd, p=2, dim=1)
            lpips_scores.append(1.0 - ((fr_norm - fd_norm)**2).mean(dim=(1, 2, 3)))
            
        gram_score = torch.stack(gram_scores, dim=0).mean(dim=0)
        dists_score = torch.stack(dists_scores, dim=0).mean(dim=0)
        lpips_score = torch.stack(lpips_scores, dim=0).mean(dim=0)
        
        # We can weigh them equally for now. Or prioritize spatial vs texture.
        # Let's do 1/3 each.
        return (gram_score + dists_score + lpips_score) / 3.0


def _build_hybrid_model(device, backbone=None, pf=None, ws=None):
    backbone = backbone or "vgg16"
    pf = pf if pf is not None else 1.0
    ws = ws if ws is not None else 4
    
    nodes = list(CFG.candidate_layers(backbone).values())
    if len(nodes) > 5:
        idx = torch.linspace(0, len(nodes)-1, 5).long()
        feature_layers = [nodes[i.item()] for i in idx]
    else:
        feature_layers = nodes

    ext, norm = make_multi_extractor(backbone, feature_layers)
    return IDFIQA_Hybrid(ext, norm,
                         device=device, percent_features_to_keep=pf, window_size=ws)


@register_experiment
class HybridExperiment(DefaultExperiment):
    name = "hybrid"
    description = "Hybrid Gram + DISTS + LPIPS model"
    summary_prefix = "hybrid"

    def add_arguments(self, parser):
        parser.add_argument("--backbone", type=str, default="vgg16")
        parser.add_argument("--percent-features", type=float, default=1.0)
        parser.add_argument("--window-size", type=int, default=4)

    def build_model(self, device, args):
        return _build_hybrid_model(device, backbone=args.backbone,
                                   pf=args.percent_features,
                                   ws=args.window_size)

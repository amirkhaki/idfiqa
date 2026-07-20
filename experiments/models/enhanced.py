"""Enhanced spatial SSIM IDFIQA model."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CFG, get_all_feature_nodes
from ..extractors import make_multi_extractor
from ..registry import DefaultExperiment, register_experiment


class IDFIQA_Enhanced(nn.Module):
    """
    Enhanced IDFIQA using multi-layer Gram local similarity and Cosine similarity.
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

    def _gram(self, feat):
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
            
            C = gr.shape[-1]
            if C < self.ws:
                continue
                
            # Baseline Gram local
            gr_u = F.unfold(gr.unsqueeze(1), kernel_size=self.ws, stride=1).transpose(1, 2)
            gd_u = F.unfold(gd.unsqueeze(1), kernel_size=self.ws, stride=1).transpose(1, 2)
            
            vr = torch.var(gr_u, dim=2, unbiased=False)
            vd = torch.var(gd_u, dim=2, unbiased=False)
            mr = torch.mean(gr_u, dim=2, keepdim=True)
            md = torch.mean(gd_u, dim=2, keepdim=True)
            cov = torch.mean((gr_u - mr) * (gd_u - md), dim=2)
            
            local = (2 * cov + self.xi) / (vr + vd + self.xi)
            score_gram = local.mean(dim=1)
            
            # Spatial cosine similarity
            fr_norm = F.normalize(fr, p=2, dim=1)
            fd_norm = F.normalize(fd, p=2, dim=1)
            score_spatial = (fr_norm * fd_norm).sum(dim=1).mean(dim=(1, 2))
            
            # Hybrid
            layer_scores.append(0.5 * score_gram + 0.5 * score_spatial)
            
        # Average across all selected layers
        return torch.stack(layer_scores, dim=0).mean(dim=0)


def _build_enhanced_model(device, backbone=None, pf=None, ws=None):
    backbone = backbone or CFG.backbone
    pf = pf if pf is not None else CFG.percent_features
    ws = ws if ws is not None else 4
    
    # Use evenly spaced layers from the backbone
    all_nodes = list(get_all_feature_nodes(backbone).keys())
    # Take 5 layers uniformly distributed across depth
    if len(all_nodes) >= 5:
        step = len(all_nodes) / 5.0
        feature_layers = [all_nodes[int(i * step)] for i in range(5)]
    else:
        feature_layers = all_nodes

    ext, norm = make_multi_extractor(backbone, feature_layers)
    return IDFIQA_Enhanced(ext, norm,
                           device=device, percent_features_to_keep=pf, window_size=ws)


@register_experiment
class EnhancedSSIMExperiment(DefaultExperiment):
    name = "enhanced"
    description = "Enhanced Multi-layer Hybrid"
    summary_prefix = "enhanced"

    def add_arguments(self, parser):
        parser.add_argument("--backbone", type=str, default=CFG.backbone)
        parser.add_argument("--percent-features", type=float, default=CFG.percent_features)
        parser.add_argument("--window-size", type=int, default=4)

    def build_model(self, device, args):
        return _build_enhanced_model(device, backbone=args.backbone,
                                     pf=args.percent_features,
                                     ws=args.window_size)

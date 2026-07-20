"""Enhanced spatial SSIM IDFIQA model."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CFG, get_all_feature_nodes
from ..extractors import make_multi_extractor
from ..registry import DefaultExperiment, register_experiment


class IDFIQA_Enhanced(nn.Module):
    """
    Enhanced IDFIQA using multi-layer Spatial SSIM.
    Uses avg_pool2d to avoid memory issues and computes SSIM over multiple layers.
    """

    def __init__(self, feature_extractor, normalize,
                 device=None,
                 percent_features_to_keep=1.0,
                 window_size=7,
                 c1=1e-6,
                 c2=1e-6):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor.to(self.device).eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.normalize = normalize
        self.pf = percent_features_to_keep
        self.ws = window_size
        self.c1 = c1
        self.c2 = c2

    def _select_channels(self, feat_ref, feat_dist):
        n, c, h, w = feat_ref.shape
        k = max(1, int(c * self.pf))
        # Variance across spatial dims
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
                
            B, C, H, W = fr.shape
            ws = min(self.ws, H, W)
            if ws % 2 == 0:
                ws -= 1 # Ensure odd window size if possible, though pool2d supports even
            ws = max(1, ws)
            
            # Using avg_pool2d to compute local mean and variance
            mr = F.avg_pool2d(fr, kernel_size=ws, stride=1)
            md = F.avg_pool2d(fd, kernel_size=ws, stride=1)
            
            mr_sq = mr.pow(2)
            md_sq = md.pow(2)
            mr_md = mr * md
            
            vr = F.avg_pool2d(fr.pow(2), kernel_size=ws, stride=1) - mr_sq
            vd = F.avg_pool2d(fd.pow(2), kernel_size=ws, stride=1) - md_sq
            cov = F.avg_pool2d(fr * fd, kernel_size=ws, stride=1) - mr_md
            
            vr = torch.clamp(vr, min=0.0)
            vd = torch.clamp(vd, min=0.0)
            
            L = (2 * mr_md + self.c1) / (mr_sq + md_sq + self.c1)
            CS = (2 * cov + self.c2) / (vr + vd + self.c2)
            
            ssim = L * CS # (B, C, H', W')
            layer_scores.append(ssim.mean(dim=(1, 2, 3)))
            
        # Average across all selected layers
        return torch.stack(layer_scores, dim=0).mean(dim=0)


def _build_enhanced_model(device, backbone=None, pf=None, ws=None):
    backbone = backbone or CFG.backbone
    pf = pf if pf is not None else CFG.percent_features
    ws = ws if ws is not None else 7
    
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
    description = "Enhanced Multi-layer Spatial SSIM"
    summary_prefix = "enhanced"

    def add_arguments(self, parser):
        parser.add_argument("--backbone", type=str, default=CFG.backbone)
        parser.add_argument("--percent-features", type=float, default=CFG.percent_features)
        parser.add_argument("--window-size", type=int, default=7)

    def build_model(self, device, args):
        return _build_enhanced_model(device, backbone=args.backbone,
                                     pf=args.percent_features,
                                     ws=args.window_size)

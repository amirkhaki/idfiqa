"""Enhanced unweighted DISTS IDFIQA model."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CFG, get_all_feature_nodes
from ..extractors import make_multi_extractor
from ..registry import DefaultExperiment, register_experiment


class IDFIQA_Enhanced(nn.Module):
    """
    Enhanced IDFIQA using unweighted DISTS-like global structural similarity.
    """

    def __init__(self, feature_extractor, normalize,
                 device=None,
                 percent_features_to_keep=1.0,
                 xi=1e-6):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor.to(self.device).eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.normalize = normalize
        self.pf = percent_features_to_keep
        self.xi = xi

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
        
        layer_scores = []
        for k in out_r.keys():
            fr = out_r[k]
            fd = out_d[k]
            
            fr, fd = self._select_channels(fr, fd)
                
            # Global spatial mean and variance per channel
            mu_r = fr.mean(dim=(2, 3), keepdim=True)
            mu_d = fd.mean(dim=(2, 3), keepdim=True)
            
            var_r = ((fr - mu_r) ** 2).mean(dim=(2, 3), keepdim=True)
            var_d = ((fd - mu_d) ** 2).mean(dim=(2, 3), keepdim=True)
            
            cov_rd = ((fr - mu_r) * (fd - mu_d)).mean(dim=(2, 3), keepdim=True)
            
            # Structure and texture similarity
            s_mean = (2 * mu_r * mu_d + self.xi) / (mu_r ** 2 + mu_d ** 2 + self.xi)
            s_var = (2 * cov_rd + self.xi) / (var_r + var_d + self.xi)
            
            # Combine similarities (mean over spatial dimension is already done, just mean over channels)
            layer_sim = (s_mean.mean(dim=(1, 2, 3)) + s_var.mean(dim=(1, 2, 3))) / 2.0
            layer_scores.append(layer_sim)
            
        # Average across all selected layers
        return torch.stack(layer_scores, dim=0).mean(dim=0)


def _build_enhanced_model(device, backbone=None, pf=None):
    backbone = backbone or CFG.backbone
    pf = pf if pf is not None else 1.0
    
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
                           device=device, percent_features_to_keep=pf)


@register_experiment
class EnhancedSSIMExperiment(DefaultExperiment):
    name = "enhanced"
    description = "Enhanced Multi-layer Unweighted DISTS"
    summary_prefix = "enhanced"

    def add_arguments(self, parser):
        parser.add_argument("--backbone", type=str, default=CFG.backbone)
        parser.add_argument("--percent-features", type=float, default=1.0)

    def build_model(self, device, args):
        return _build_enhanced_model(device, backbone=args.backbone, pf=args.percent_features)

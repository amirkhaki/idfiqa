"""Enhanced spatial SSIM IDFIQA model."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CFG
from ..extractors import make_multi_extractor
from ..registry import DefaultExperiment, register_experiment


class IDFIQA_Enhanced(nn.Module):
    """
    Enhanced IDFIQA using Spatial SSIM on raw feature maps.
    This avoids Gram matrices, which are easily fooled by GAN distortions in PIPAL.
    """

    def __init__(self, feature_extractor, normalize,
                 device=None,
                 percent_features_to_keep=1.0,
                 window_size=7,
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
            
            if self.pf < 1.0:
                fr, fd = self._select_channels(fr, fd)
            
            N, C, H, W = fr.shape
            if H < self.ws or W < self.ws:
                # If feature map is too small, just do global SSIM
                mr = torch.mean(fr, dim=(2, 3))
                md = torch.mean(fd, dim=(2, 3))
                vr = torch.var(fr, dim=(2, 3), unbiased=False)
                vd = torch.var(fd, dim=(2, 3), unbiased=False)
                cov = torch.mean((fr - mr.unsqueeze(-1).unsqueeze(-1)) * (fd - md.unsqueeze(-1).unsqueeze(-1)), dim=(2, 3))
                
                s_var = (2 * cov + self.xi) / (vr + vd + self.xi)
                s_mean = (2 * mr * md + self.xi) / (mr ** 2 + md ** 2 + self.xi)
                score = (s_mean * s_var).mean(dim=1)
                layer_scores.append(score)
                continue
                
            # Local spatial SSIM on the features
            fr_u = F.unfold(fr, kernel_size=self.ws, stride=1).view(N, C, self.ws*self.ws, -1)
            fd_u = F.unfold(fd, kernel_size=self.ws, stride=1).view(N, C, self.ws*self.ws, -1)
            
            vr = torch.var(fr_u, dim=2, unbiased=False)
            vd = torch.var(fd_u, dim=2, unbiased=False)
            mr = torch.mean(fr_u, dim=2)
            md = torch.mean(fd_u, dim=2)
            cov = torch.mean((fr_u - mr.unsqueeze(2)) * (fd_u - md.unsqueeze(2)), dim=2)
            
            s_var = (2 * cov + self.xi) / (vr + vd + self.xi)
            s_mean = (2 * mr * md + self.xi) / (mr ** 2 + md ** 2 + self.xi)
            
            local = s_mean * s_var
            # Average over patches and channels
            score = local.mean(dim=(1, 2))
            
            layer_scores.append(score)
            
        # Average across all selected layers
        return torch.stack(layer_scores, dim=0).mean(dim=0)


def _build_enhanced_model(device, backbone=None, pf=None, ws=None):
    backbone = backbone or CFG.backbone
    pf = pf if pf is not None else 1.0  # Use all features
    ws = ws if ws is not None else 7    # Use larger window size for spatial SSIM
    
    # Use standard 5 evenly spaced layers for VGG
    feature_layers = ["features.3", "features.8", "features.15", "features.22", "features.29"]

    ext, norm = make_multi_extractor(backbone, feature_layers)
    return IDFIQA_Enhanced(ext, norm,
                           device=device, percent_features_to_keep=pf, window_size=ws)


@register_experiment
class EnhancedSSIMExperiment(DefaultExperiment):
    name = "enhanced"
    description = "Enhanced Spatial Feature SSIM"
    summary_prefix = "enhanced"

    def add_arguments(self, parser):
        parser.add_argument("--backbone", type=str, default=CFG.backbone)
        parser.add_argument("--percent-features", type=float, default=1.0)
        parser.add_argument("--window-size", type=int, default=7)

    def build_model(self, device, args):
        return _build_enhanced_model(device, backbone=args.backbone,
                                     pf=args.percent_features,
                                     ws=args.window_size)

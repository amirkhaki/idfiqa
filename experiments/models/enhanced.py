"""Enhanced spatial SSIM IDFIQA model."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CFG
from ..extractors import make_single_extractor
from ..registry import DefaultExperiment, register_experiment


class IDFIQA_Enhanced(nn.Module):
    """
    Enhanced IDFIQA.
    Computes a full SSIM on the spatial feature maps instead of Gram matrices.
    """

    def __init__(self, feature_extractor, normalize,
                 feature_node_key="features",
                 device=None,
                 percent_features_to_keep=0.6,
                 window_size=11,
                 c1=1e-6,
                 c2=1e-6):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor.to(self.device).eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.normalize = normalize
        self.feature_node_key = feature_node_key
        self.pf = percent_features_to_keep
        self.ws = window_size
        self.c1 = c1
        self.c2 = c2

    def _features(self, img):
        out = self.feature_extractor(self.normalize(img.to(self.device)))
        return out[self.feature_node_key] if isinstance(out, dict) else out

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
        
        if self.pf < 1.0:
            fr, fd = self._select_channels(fr, fd)
            
        B, C, H, W = fr.shape
        
        # For very small feature maps, average pooling might be needed or reduce ws
        ws = min(self.ws, H, W)
        
        fr_u = F.unfold(fr, kernel_size=ws, stride=1)  # (B, C * ws^2, L_patches)
        fd_u = F.unfold(fd, kernel_size=ws, stride=1)
        
        L_patches = fr_u.shape[-1]
        
        fr_u = fr_u.view(B, C, ws * ws, L_patches)
        fd_u = fd_u.view(B, C, ws * ws, L_patches)
        
        mr = fr_u.mean(dim=2)  # (B, C, L_patches)
        md = fd_u.mean(dim=2)
        
        vr = fr_u.var(dim=2, unbiased=False)
        vd = fd_u.var(dim=2, unbiased=False)
        
        cov = ((fr_u - mr.unsqueeze(2)) * (fd_u - md.unsqueeze(2))).mean(dim=2)
        
        L = (2 * mr * md + self.c1) / (mr.pow(2) + md.pow(2) + self.c1)
        CS = (2 * cov + self.c2) / (vr + vd + self.c2)
        
        ssim = L * CS
        
        return ssim.mean(dim=(1, 2))


def _build_enhanced_model(device, backbone=None, feature_layer=None, pf=None, ws=None):
    backbone = backbone or CFG.backbone
    feature_layer = feature_layer or CFG.get_feature_layer(backbone)
    pf = pf if pf is not None else CFG.percent_features
    ws = ws if ws is not None else 11
    ext, norm, key = make_single_extractor(backbone, feature_layer)
    return IDFIQA_Enhanced(ext, norm, feature_node_key=key,
                           device=device, percent_features_to_keep=pf, window_size=ws)


@register_experiment
class EnhancedSSIMExperiment(DefaultExperiment):
    name = "enhanced"
    description = "Enhanced SSIM on spatial feature maps"
    summary_prefix = "enhanced"

    def add_arguments(self, parser):
        parser.add_argument("--backbone", type=str, default=CFG.backbone)
        parser.add_argument("--feature-layer", type=str, default=None)
        parser.add_argument("--percent-features", type=float, default=CFG.percent_features)
        parser.add_argument("--window-size", type=int, default=11)

    def build_model(self, device, args):
        return _build_enhanced_model(device, backbone=args.backbone,
                                     feature_layer=args.feature_layer,
                                     pf=args.percent_features,
                                     ws=args.window_size)

"""Enhanced IDFIQA model combining DISTS and LPIPS concepts."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CFG
from ..extractors import make_multi_extractor
from ..registry import DefaultExperiment, register_experiment

class IDFIQA_Enhanced(nn.Module):
    """
    Enhanced IDFIQA using unweighted DISTS (Global Spatial SSIM) 
    and unweighted LPIPS (Normalized L2 Distance).
    """

    def __init__(self, feature_extractor, normalize,
                 device=None,
                 percent_features_to_keep=1.0,
                 window_size=None,
                 xi=1e-8):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor.to(self.device).eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.normalize = normalize
        self.pf = percent_features_to_keep
        self.xi = xi

    def forward(self, ref, dist):
        out_r = self.feature_extractor(self.normalize(ref.to(self.device)))
        out_d = self.feature_extractor(self.normalize(dist.to(self.device)))
        
        layer_scores = []
        for k in out_r.keys():
            fr = out_r[k]
            fd = out_d[k]
            
            # ConvNeXt can output (N, C) for the classifier layer, we must handle it.
            if fr.dim() == 2:
                fr = fr.unsqueeze(-1).unsqueeze(-1)
                fd = fd.unsqueeze(-1).unsqueeze(-1)
                
            n, c = fr.shape[:2]
            fr_flat = fr.view(n, c, -1)
            fd_flat = fd.view(n, c, -1)
            
            # Unweighted DISTS
            mr = torch.mean(fr_flat, dim=2, keepdim=True)
            md = torch.mean(fd_flat, dim=2, keepdim=True)
            vr = torch.var(fr_flat, dim=2, unbiased=False)
            vd = torch.var(fd_flat, dim=2, unbiased=False)
            cov = torch.mean((fr_flat - mr) * (fd_flat - md), dim=2)
            
            mr = mr.squeeze(2)
            md = md.squeeze(2)
            
            s_mean = (2 * mr * md + self.xi) / (mr ** 2 + md ** 2 + self.xi)
            s_var = (2 * cov + self.xi) / (vr + vd + self.xi)
            dists_score = (s_mean * s_var).mean(dim=1)
            
            # Unweighted LPIPS
            fr_norm = F.normalize(fr_flat, p=2, dim=1)
            fd_norm = F.normalize(fd_flat, p=2, dim=1)
            lpips_dist = ((fr_norm - fd_norm)**2).mean(dim=(1, 2))
            lpips_score = 1.0 - lpips_dist
            
            # Combine both concepts
            score = 0.5 * dists_score + 0.5 * lpips_score
            layer_scores.append(score)
            
        # Average across all selected layers
        return torch.stack(layer_scores, dim=0).mean(dim=0)


def _build_enhanced_model(device, backbone=None, pf=None, ws=None):
    backbone = backbone or "vgg16"
    pf = pf if pf is not None else 1.0
    
    if "vgg" in backbone:
        feature_layers = ["features.3", "features.8", "features.15", "features.22", "features.29"]
    elif "convnext" in backbone:
        feature_layers = ["features.1", "features.3", "features.5", "features.7"]
    elif "resnet" in backbone:
        feature_layers = ["layer1", "layer2", "layer3", "layer4"]
    elif "efficientnet" in backbone:
        feature_layers = ["features.1", "features.3", "features.5", "features.7"]
    else:
        feature_layers = [CFG.get_feature_layer(backbone)]

    ext, norm = make_multi_extractor(backbone, feature_layers)
    return IDFIQA_Enhanced(ext, norm,
                           device=device, percent_features_to_keep=pf, window_size=ws)


@register_experiment
class EnhancedSSIMExperiment(DefaultExperiment):
    name = "enhanced"
    description = "Enhanced Unweighted DISTS + LPIPS"
    summary_prefix = "enhanced"

    def add_arguments(self, parser):
        parser.add_argument("--backbone", type=str, default="vgg16")
        parser.add_argument("--percent-features", type=float, default=1.0)

    def build_model(self, device, args):
        return _build_enhanced_model(device, backbone=args.backbone,
                                     pf=args.percent_features)

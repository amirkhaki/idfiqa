"""Local DISTS Experiment."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CFG
from ..extractors import make_multi_extractor
from ..registry import DefaultExperiment, register_experiment

class LocalDISTS(nn.Module):
    """
    Computes Local Spatial SSIM on Deep Features instead of Global SSIM.
    This helps in capturing localized GAN artifacts and spatial distortions,
    which are very common in PIPAL.
    """

    def __init__(self, feature_extractor, normalize,
                 device=None,
                 window_size=11,
                 xi=1e-8):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor.to(self.device).eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.normalize = normalize
        self.ws = window_size
        self.xi = xi

    def forward(self, ref, dist):
        out_r = self.feature_extractor(self.normalize(ref.to(self.device)))
        out_d = self.feature_extractor(self.normalize(dist.to(self.device)))
        
        layer_scores = []
        for k in out_r.keys():
            fr = out_r[k]
            fd = out_d[k]
            
            # Pad to keep spatial dimensions same if needed, though for average pool it's fine
            pad = self.ws // 2
            
            # Local DISTS (Spatial SSIM on Features)
            mr = F.avg_pool2d(fr, kernel_size=self.ws, stride=1, padding=pad)
            md = F.avg_pool2d(fd, kernel_size=self.ws, stride=1, padding=pad)
            
            mr_sq = mr ** 2
            md_sq = md ** 2
            mr_md = mr * md
            
            vr = F.avg_pool2d(fr ** 2, kernel_size=self.ws, stride=1, padding=pad) - mr_sq
            vd = F.avg_pool2d(fd ** 2, kernel_size=self.ws, stride=1, padding=pad) - md_sq
            cov = F.avg_pool2d(fr * fd, kernel_size=self.ws, stride=1, padding=pad) - mr_md
            
            # Some feature values might cause negative variance due to floating point, so clamp it
            vr = torch.clamp(vr, min=0.0)
            vd = torch.clamp(vd, min=0.0)
            
            s_mean = (2 * mr_md + self.xi) / (mr_sq + md_sq + self.xi)
            s_var = (2 * cov + self.xi) / (vr + vd + self.xi)
            
            local_ssim = s_mean * s_var
            
            # Average over spatial and channel
            dists_score = local_ssim.mean(dim=(1, 2, 3))
            
            # Local Unweighted LPIPS
            fr_norm = F.normalize(fr, p=2, dim=1)
            fd_norm = F.normalize(fd, p=2, dim=1)
            lpips = 1.0 - ((fr_norm - fd_norm)**2).mean(dim=1) # (N, H, W)
            lpips_score = lpips.mean(dim=(1, 2))
            
            # Combine
            layer_scores.append(0.5 * dists_score + 0.5 * lpips_score)
            
        return torch.stack(layer_scores, dim=0).mean(dim=0)


def _build_local_dists(device, backbone=None, ws=None):
    backbone = backbone or "vgg16"
    ws = ws if ws is not None else 11
    
    nodes = list(CFG.candidate_layers(backbone).values())
    if len(nodes) > 5:
        idx = torch.linspace(0, len(nodes)-1, 5).long()
        feature_layers = [nodes[i.item()] for i in idx]
    else:
        feature_layers = nodes

    ext, norm = make_multi_extractor(backbone, feature_layers)
    return LocalDISTS(ext, norm, device=device, window_size=ws)


@register_experiment
class LocalDISTSExperiment(DefaultExperiment):
    name = "local_dists"
    description = "Local DISTS + LPIPS on Features"
    summary_prefix = "local_dists"

    def add_arguments(self, parser):
        parser.add_argument("--backbone", type=str, default="vgg16")
        parser.add_argument("--window-size", type=int, default=11)

    def slug_args(self, args):
        base = super().slug_args(args)
        base["wt_layer"] = None
        base["patch_size"] = None
        return base

    def build_model(self, device, args):
        return _build_local_dists(device, backbone=args.backbone, ws=args.window_size)

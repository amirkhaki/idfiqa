"""Local Feature SSIM + LPIPS model."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CFG
from ..extractors import make_multi_extractor
from ..registry import DefaultExperiment, register_experiment

class IDFIQA_LocalDISTS(nn.Module):
    def __init__(self, feature_extractor, normalize,
                 device=None,
                 window_size=3,
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
            
            if fr.dim() == 2:
                fr = fr.unsqueeze(-1).unsqueeze(-1)
                fd = fd.unsqueeze(-1).unsqueeze(-1)
                
            # Local DISTS
            pad = self.ws // 2
            
            # Using AvgPool instead of unfold for memory efficiency
            pool = nn.AvgPool2d(kernel_size=self.ws, stride=1, padding=pad)
            
            mr = pool(fr)
            md = pool(fd)
            
            vr = pool(fr**2) - mr**2
            vd = pool(fd**2) - md**2
            cov = pool(fr * fd) - mr * md
            
            # Ensure non-negative variance
            vr = torch.clamp(vr, min=0.0)
            vd = torch.clamp(vd, min=0.0)
            
            s_mean = (2 * mr * md + self.xi) / (mr ** 2 + md ** 2 + self.xi)
            s_var = (2 * cov + self.xi) / (vr + vd + self.xi)
            
            dists_map = s_mean * s_var
            
            # Global average of local SSIMs for each channel, then mean across channels
            dists_score = dists_map.mean(dim=(2, 3)).mean(dim=1)
            
            # Global LPIPS
            fr_norm = F.normalize(fr, p=2, dim=1)
            fd_norm = F.normalize(fd, p=2, dim=1)
            lpips_score = 1.0 - ((fr_norm - fd_norm)**2).mean(dim=(1, 2, 3))
            
            score = 0.5 * dists_score + 0.5 * lpips_score
            layer_scores.append(score)
            
        return torch.stack(layer_scores, dim=0).mean(dim=0)

def _build_localdists_model(device, backbone=None, ws=None):
    backbone = backbone or "vgg16"
    ws = ws if ws is not None else 3
    
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
    return IDFIQA_LocalDISTS(ext, norm,
                             device=device, window_size=ws)

@register_experiment
class LocalDISTSExperiment(DefaultExperiment):
    name = "localdists"
    description = "Local DISTS feature SSIM + LPIPS"
    summary_prefix = "localdists"

    def add_arguments(self, parser):
        parser.add_argument("--backbone", type=str, default="vgg16")
        parser.add_argument("--window-size", type=int, default=3)

    def build_model(self, device, args):
        return _build_localdists_model(device, backbone=args.backbone, ws=args.window_size)

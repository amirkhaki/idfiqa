"""Patch-based aggregation variant of baseline."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CFG
from ..extractors import make_multi_extractor
from ..registry import DefaultExperiment, register_experiment

class IDFIQA_Patch(nn.Module):
    """
    Computes DISTS + LPIPS on overlapping or non-overlapping patches,
    and aggregates them to form the final score.
    """

    def __init__(self, feature_extractor, normalize,
                 device=None,
                 percent_features_to_keep=1.0,
                 patch_size=32,
                 aggregation="mean",
                 xi=1e-8):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor.to(self.device).eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.normalize = normalize
        self.pf = percent_features_to_keep
        self.ps = patch_size
        self.agg = aggregation
        self.xi = xi

    def _patchify(self, tensor, size):
        # tensor is (N, C, H, W)
        # We want to extract patches of size (size x size) from the input IMAGE,
        # but here tensor is the input image!
        # wait, if we extract patches from the input image, we can just feed them through the network.
        # But this increases batch size significantly.
        # Let's extract patches from the image.
        N, C, H, W = tensor.shape
        # Just use unfold to get patches
        patches = tensor.unfold(2, size, size).unfold(3, size, size)
        # patches: (N, C, H//size, W//size, size, size)
        patches = patches.contiguous().view(N, C, -1, size, size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous() # (N, num_patches, C, size, size)
        return patches

    def _base_score(self, fr, fd):
        # fr, fd: (B, C, H, W)
        mr = torch.mean(fr, dim=(2, 3))
        md = torch.mean(fd, dim=(2, 3))
        vr = torch.var(fr, dim=(2, 3), unbiased=False)
        vd = torch.var(fd, dim=(2, 3), unbiased=False)
        cov = torch.mean((fr - mr.unsqueeze(-1).unsqueeze(-1)) * (fd - md.unsqueeze(-1).unsqueeze(-1)), dim=(2, 3))
        
        s_mean = (2 * mr * md + self.xi) / (mr ** 2 + md ** 2 + self.xi)
        s_var = (2 * cov + self.xi) / (vr + vd + self.xi)
        dists_score = (s_mean * s_var).mean(dim=1)
        
        fr_norm = F.normalize(fr, p=2, dim=1)
        fd_norm = F.normalize(fd, p=2, dim=1)
        lpips_score = 1.0 - ((fr_norm - fd_norm)**2).mean(dim=(1, 2, 3))
        
        return 0.5 * dists_score + 0.5 * lpips_score

    def forward(self, ref, dist):
        # Pad images to be divisible by patch_size
        H, W = ref.shape[2], ref.shape[3]
        pad_h = (self.ps - H % self.ps) % self.ps
        pad_w = (self.ps - W % self.ps) % self.ps
        
        if pad_h > 0 or pad_w > 0:
            ref = F.pad(ref, (0, pad_w, 0, pad_h), mode='reflect')
            dist = F.pad(dist, (0, pad_w, 0, pad_h), mode='reflect')
            
        ref_patches = self._patchify(ref, self.ps)
        dist_patches = self._patchify(dist, self.ps)
        
        N, num_patches, C, H_p, W_p = ref_patches.shape
        
        ref_flat = ref_patches.view(N * num_patches, C, H_p, W_p)
        dist_flat = dist_patches.view(N * num_patches, C, H_p, W_p)
        
        # Batch size might be too large for feature extractor, so we process it in chunks
        batch_size = 64
        scores_list = []
        
        for i in range(0, N * num_patches, batch_size):
            r_b = ref_flat[i:i+batch_size]
            d_b = dist_flat[i:i+batch_size]
            
            out_r = self.feature_extractor(self.normalize(r_b.to(self.device)))
            out_d = self.feature_extractor(self.normalize(d_b.to(self.device)))
            
            layer_scores = []
            for k in out_r.keys():
                layer_scores.append(self._base_score(out_r[k], out_d[k]))
                
            # Mean over layers
            patch_score = torch.stack(layer_scores, dim=0).mean(dim=0)
            scores_list.append(patch_score)
            
        all_patch_scores = torch.cat(scores_list, dim=0).view(N, num_patches)
        
        if self.agg == "mean":
            final_scores = all_patch_scores.mean(dim=1)
        elif self.agg == "max":
            final_scores = all_patch_scores.max(dim=1)[0]
        elif self.agg.startswith("softmax_"):
            # softmax aggregation
            T = float(self.agg.split("_")[1])
            weights = F.softmax(all_patch_scores / T, dim=1)
            final_scores = (all_patch_scores * weights).sum(dim=1)
        else:
            final_scores = all_patch_scores.mean(dim=1)
            
        return final_scores

def _build_patch_model(device, backbone=None, pf=None, ps=None, agg=None):
    backbone = backbone or "vgg16"
    ps = ps if ps is not None else 64
    agg = agg or "mean"
    
    nodes = list(CFG.candidate_layers(backbone).values())
    if len(nodes) > 5:
        idx = torch.linspace(0, len(nodes)-1, 5).long()
        feature_layers = [nodes[i.item()] for i in idx]
    else:
        feature_layers = nodes

    ext, norm = make_multi_extractor(backbone, feature_layers)
    return IDFIQA_Patch(ext, norm,
                        device=device, percent_features_to_keep=pf,
                        patch_size=ps, aggregation=agg)

@register_experiment
class PatchExperiment(DefaultExperiment):
    name = "patch"
    description = "Patch-based aggregation variant"
    summary_prefix = "patch"

    def add_arguments(self, parser):
        parser.add_argument("--backbone", type=str, default="vgg16")
        parser.add_argument("--percent-features", type=float, default=1.0)
        parser.add_argument("--patch-size", type=int, default=64)
        parser.add_argument("--aggregation", type=str, default="mean")

    def slug_args(self, args):
        base = super().slug_args(args)
        base["wt_layer"] = None
        base["patch_size"] = args.patch_size
        base["aggregation"] = args.aggregation
        return base

    def build_model(self, device, args):
        return _build_patch_model(device, backbone=args.backbone,
                                  pf=args.percent_features,
                                  ps=args.patch_size,
                                  agg=args.aggregation)

"""Hybrid model combining Gram Matrix SSIM, Local DISTS, and Local LPIPS."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CFG
from ..extractors import make_multi_extractor
from ..registry import DefaultExperiment, register_experiment


class IDFIQA_Hybrid(nn.Module):
    """
    Combines:
    1. Gram-based local SSIM (Texture) - computed on variance-selected channels
    2. Local DISTS spatial SSIM (Structure) - computed on variance-selected channels
    3. Local LPIPS normalized L2 (Pixel/Feature differences) - computed on variance-selected channels
    """

    def __init__(self, feature_extractor, normalize,
                 device=None,
                 percent_features_to_keep=0.6,
                 window_size=4,
                 alpha=1.0,
                 beta=1.0,
                 gamma=1.0,
                 xi=1e-8,
                 spatial_pooling="mean"):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor.to(self.device).eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.normalize = normalize
        self.pf = percent_features_to_keep
        self.ws = window_size
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.xi = xi
        self.spatial_pooling = spatial_pooling

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
            
            # Ensure spatial dimensions exist
            if fr.dim() == 2:
                fr = fr.unsqueeze(-1).unsqueeze(-1)
                fd = fd.unsqueeze(-1).unsqueeze(-1)
                
            # Variance-guided channel selection
            if self.pf < 1.0:
                fr, fd = self._select_channels(fr, fd)
                
            # 1. Local Gram SSIM (Texture)
            if fr.shape[2] >= self.ws and fr.shape[3] >= self.ws:
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
            else:
                n, c = fr.shape[:2]
                fr_flat = fr.view(n, c, -1)
                fd_flat = fd.view(n, c, -1)
                gr = torch.bmm(fr_flat, fr_flat.transpose(1, 2)) / fr_flat.shape[2]
                gd = torch.bmm(fd_flat, fd_flat.transpose(1, 2)) / fd_flat.shape[2]
                vr_g = torch.var(gr, dim=(1, 2), unbiased=False)
                vd_g = torch.var(gd, dim=(1, 2), unbiased=False)
                mr_g = torch.mean(gr, dim=(1, 2))
                md_g = torch.mean(gd, dim=(1, 2))
                cov_g = torch.mean((gr - mr_g.unsqueeze(-1).unsqueeze(-1)) * (gd - md_g.unsqueeze(-1).unsqueeze(-1)), dim=(1, 2))
                s_mean_g = (2 * mr_g * md_g + self.xi) / (mr_g ** 2 + md_g ** 2 + self.xi)
                s_var_g = (2 * cov_g + self.xi) / (vr_g + vd_g + self.xi)
                gram_scores.append(s_mean_g * s_var_g)
            
            # 2. Global DISTS per-channel
            if self.spatial_pooling == "global":
                mr = torch.mean(fr, dim=(2, 3), keepdim=True)
                md = torch.mean(fd, dim=(2, 3), keepdim=True)
                vr = torch.var(fr, dim=(2, 3), unbiased=False, keepdim=True)
                vd = torch.var(fd, dim=(2, 3), unbiased=False, keepdim=True)
                cov = torch.mean((fr - mr) * (fd - md), dim=(2, 3), keepdim=True)
                
                s_mean = (2 * mr * md + self.xi) / (mr ** 2 + md ** 2 + self.xi)
                s_var = (2 * cov + self.xi) / (vr + vd + self.xi)
                
                # Combine structure and texture similarity per channel, then average across channels
                dists_map = s_mean * s_var
                dists_scores.append(dists_map.mean(dim=(1, 2, 3)))
            else:
                pad = self.ws // 2
                pool = nn.AvgPool2d(kernel_size=self.ws, stride=1, padding=pad)
                
                mr = pool(fr)
                md = pool(fd)
                vr = torch.clamp(pool(fr**2) - mr**2, min=0.0)
                vd = torch.clamp(pool(fd**2) - md**2, min=0.0)
                cov = pool(fr * fd) - mr * md
                
                s_mean = (2 * mr * md + self.xi) / (mr ** 2 + md ** 2 + self.xi)
                s_var = (2 * cov + self.xi) / (vr + vd + self.xi)
                dists_map = s_mean * s_var
                n, c = dists_map.shape[:2]
                
                if self.spatial_pooling == "min":
                    dists_scores.append(dists_map.view(n, c, -1).min(dim=2)[0].mean(dim=1))
                elif self.spatial_pooling == "worst10":
                    dists_flat = dists_map.view(n, c, -1)
                    k = max(1, int(dists_flat.shape[2] * 0.1))
                    worst_k = torch.topk(dists_flat, k, dim=2, largest=False)[0]
                    dists_scores.append(worst_k.mean(dim=2).mean(dim=1))
                else:
                    dists_scores.append(dists_map.mean(dim=(2, 3)).mean(dim=1))
            
            # 3. LPIPS
            if self.spatial_pooling == "global":
                # Global MSE of channel-normalized features
                fr_norm = F.normalize(fr, p=2, dim=1)
                fd_norm = F.normalize(fd, p=2, dim=1)
                # 1 - Global MSE
                lpips_map = 1.0 - torch.mean((fr_norm - fd_norm)**2, dim=(2, 3))
                lpips_scores.append(lpips_map.mean(dim=1))
            else:
                fr_norm = F.normalize(fr, p=2, dim=1)
                fd_norm = F.normalize(fd, p=2, dim=1)
                lpips_map = 1.0 - ((fr_norm - fd_norm)**2)
                n, c = lpips_map.shape[:2]
                if self.spatial_pooling == "min":
                    lpips_scores.append(lpips_map.view(n, c, -1).min(dim=2)[0].mean(dim=1))
                elif self.spatial_pooling == "worst10":
                    lpips_flat = lpips_map.view(n, c, -1)
                    k = max(1, int(lpips_flat.shape[2] * 0.1))
                    worst_k = torch.topk(lpips_flat, k, dim=2, largest=False)[0]
                    lpips_scores.append(worst_k.mean(dim=2).mean(dim=1))
                else:
                    lpips_scores.append(lpips_map.mean(dim=(1, 2, 3)))
            
        gram_score = torch.stack(gram_scores, dim=0).mean(dim=0)
        dists_score = torch.stack(dists_scores, dim=0).mean(dim=0)
        lpips_score = torch.stack(lpips_scores, dim=0).mean(dim=0)
        
        total_weight = self.alpha + self.beta + self.gamma
        return (self.alpha * gram_score + self.beta * dists_score + self.gamma * lpips_score) / total_weight


def _build_hybrid_model(device, backbone=None, pf=None, ws=None, alpha=1.0, beta=1.0, gamma=1.0, spatial_pooling="mean"):
    backbone = backbone or "vgg16"
    pf = pf if pf is not None else 0.6
    ws = ws if ws is not None else 4

    if "vgg" in backbone:
        feature_layers = ["features.3", "features.8", "features.15", "features.22", "features.29"]
    elif "convnext" in backbone:
        feature_layers = ["features.1", "features.3", "features.5", "features.7"]
    elif "resnet" in backbone:
        feature_layers = ["layer1", "layer2", "layer3", "layer4"]
    elif "alexnet" in backbone:
        feature_layers = ["features.2", "features.5", "features.7", "features.9", "features.12"]
    else:
        feature_layers = [CFG.get_feature_layer(backbone)]

    ext, norm = make_multi_extractor(backbone, feature_layers)
    return IDFIQA_Hybrid(ext, norm,
                         device=device, percent_features_to_keep=pf, window_size=ws,
                         alpha=alpha, beta=beta, gamma=gamma, spatial_pooling=spatial_pooling)


@register_experiment
class HybridExperiment(DefaultExperiment):
    name = "hybrid"
    description = "Unified Hybrid Gram + Local DISTS + Local LPIPS model"
    summary_prefix = "hybrid"

    def add_arguments(self, parser):
        parser.add_argument("--backbone", type=str, default="vgg16")
        parser.add_argument("--percent-features", type=float, default=0.6)
        parser.add_argument("--window-size", type=int, default=4)
        parser.add_argument("--alpha", type=float, default=1.0, help="Weight for Gram SSIM")
        parser.add_argument("--beta", type=float, default=1.0, help="Weight for Local DISTS")
        parser.add_argument("--gamma", type=float, default=1.0, help="Weight for Local LPIPS")
        parser.add_argument("--spatial-pooling", type=str, default="mean", choices=["mean", "min", "worst10", "global"], help="Spatial pooling method for DISTS/LPIPS")

    def slug_args(self, args):
        base = super().slug_args(args)
        base["percent_features"] = args.percent_features
        base["window_size"] = args.window_size
        base["sp"] = args.spatial_pooling
        return base

    def build_model(self, device, args):
        return _build_hybrid_model(device, backbone=args.backbone,
                                   pf=args.percent_features,
                                   ws=args.window_size,
                                   alpha=args.alpha,
                                   beta=args.beta,
                                   gamma=args.gamma,
                                   spatial_pooling=args.spatial_pooling)

class IDFIQA_Ensemble(nn.Module):
    def __init__(self, m1, m2):
        super().__init__()
        self.m1 = m1
        self.m2 = m2
        
    def forward(self, ref, dist):
        return (self.m1(ref, dist) + self.m2(ref, dist)) / 2.0

@register_experiment
class EnsembleExperiment(DefaultExperiment):
    name = "ensemble"
    description = "Ensemble of VGG16 and ConvNeXt-Tiny Hybrid models"
    summary_prefix = "ensemble"

    def add_arguments(self, parser):
        parser.add_argument("--percent-features", type=float, default=0.6)
        parser.add_argument("--alpha", type=float, default=2.0)
        parser.add_argument("--spatial-pooling", type=str, default="mean")

    def slug_args(self, args):
        base = {"backbone": "vgg16_convnext", "feature_layer": "multi"}
        base["percent_features"] = args.percent_features
        base["sp"] = args.spatial_pooling
        return base

    def build_model(self, device, args):
        m1 = _build_hybrid_model(device, backbone="vgg16", pf=args.percent_features, ws=4, alpha=args.alpha, beta=1.0, gamma=1.0, spatial_pooling=args.spatial_pooling)
        m2 = _build_hybrid_model(device, backbone="convnext_tiny", pf=args.percent_features, ws=4, alpha=args.alpha, beta=1.0, gamma=1.0, spatial_pooling=args.spatial_pooling)
        return IDFIQA_Ensemble(m1, m2)

class IDFIQA_Multiscale(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        
    def forward(self, ref, dist):
        # Scale 1: Original
        s1 = self.base(ref, dist)
        
        # Scale 2: 1/2
        ref_2 = F.interpolate(ref, scale_factor=0.5, mode='bilinear', align_corners=False)
        dist_2 = F.interpolate(dist, scale_factor=0.5, mode='bilinear', align_corners=False)
        s2 = self.base(ref_2, dist_2)
        
        # Scale 3: 1/4
        ref_4 = F.interpolate(ref, scale_factor=0.25, mode='bilinear', align_corners=False)
        dist_4 = F.interpolate(dist, scale_factor=0.25, mode='bilinear', align_corners=False)
        s3 = self.base(ref_4, dist_4)
        
        return (s1 + s2 + s3) / 3.0

@register_experiment
class MultiscaleExperiment(DefaultExperiment):
    name = "multiscale"
    description = "Multi-Scale VGG16 Hybrid (1x, 0.5x, 0.25x)"
    summary_prefix = "multiscale"

    def add_arguments(self, parser):
        parser.add_argument("--percent-features", type=float, default=0.6)
        parser.add_argument("--alpha", type=float, default=2.0)
        parser.add_argument("--spatial-pooling", type=str, default="mean")

    def slug_args(self, args):
        base = {"backbone": "vgg16", "feature_layer": "multi"}
        base["percent_features"] = args.percent_features
        base["sp"] = args.spatial_pooling
        return base

    def build_model(self, device, args):
        base_model = _build_hybrid_model(device, backbone="vgg16", pf=args.percent_features, ws=4, alpha=args.alpha, beta=1.0, gamma=1.0, spatial_pooling=args.spatial_pooling)
        return IDFIQA_Multiscale(base_model)


class IDFIQA_KRSA(nn.Module):
    def __init__(self, extractor, norm, device, alpha=1.0, beta=1.0):
        super().__init__()
        self.feature_extractor = extractor
        self.norm = norm
        self.device = device
        self.alpha = alpha
        self.beta = beta

    def forward(self, ref, dist):
        ref = self.norm(ref)
        dist = self.norm(dist)

        feats_r = self.feature_extractor(ref)
        feats_d = self.feature_extractor(dist)

        s1_scores = []
        s2_scores = []

        for fr, fd in zip(feats_r.values(), feats_d.values()):
            n, c, h, w = fr.shape
            fr_flat = fr.view(n, c, -1)
            fd_flat = fd.view(n, c, -1)

            # Self-Similarity MAE (Gram matrix)
            gram_r = torch.bmm(fr_flat, fr_flat.transpose(1, 2)) / (h * w)
            gram_d = torch.bmm(fd_flat, fd_flat.transpose(1, 2)) / (h * w)
            s1 = torch.mean(torch.abs(gram_r - gram_d), dim=(1, 2))

            # Pairwise MAE
            s2 = torch.mean(torch.abs(fr - fd), dim=(1, 2, 3))
            
            s1_scores.append(s1)
            s2_scores.append(s2)

        s1_total = torch.stack(s1_scores, dim=0).mean(dim=0)
        s2_total = torch.stack(s2_scores, dim=0).mean(dim=0)

        # Logarithmic summation
        score = torch.log(self.alpha * s1_total + self.beta * s2_total + 1.0)
        return score

@register_experiment
class KRSAExperiment(DefaultExperiment):
    name = "krsa"
    description = "KRSA-style L1 Gram + L1 Feature Metric"
    summary_prefix = "krsa"

    def add_arguments(self, parser):
        parser.add_argument("--backbone", type=str, default="vgg16")
        parser.add_argument("--alpha", type=float, default=1.0)
        parser.add_argument("--beta", type=float, default=1.0)

    def slug_args(self, args):
        base = {"backbone": args.backbone, "feature_layer": f"krsa_a{args.alpha}_b{args.beta}"}
        return base

    def build_model(self, device, args):
        if "vgg" in args.backbone:
            feature_layers = ["features.3", "features.8", "features.15", "features.22", "features.29"]
        elif "convnext" in args.backbone:
            feature_layers = ["features.1", "features.3", "features.5", "features.7"]
        elif "resnet" in args.backbone:
            feature_layers = ["layer1", "layer2", "layer3", "layer4"]
        else:
            feature_layers = ["features.2", "features.5", "features.7", "features.9", "features.12"]
            
        extractor, norm = make_multi_extractor(args.backbone, feature_layers)
        model = IDFIQA_KRSA(extractor, norm, device, alpha=args.alpha, beta=args.beta)
        model = model.to(device)
        model.eval()
        return model


@register_experiment
class ShallowHybridExperiment(DefaultExperiment):
    name = "shallow_hybrid"
    description = "Hybrid model using only shallow layers (features.3, features.8)"
    summary_prefix = "shallow_hybrid"

    def add_arguments(self, parser):
        parser.add_argument("--percent-features", type=float, default=0.6)
        parser.add_argument("--alpha", type=float, default=2.0)
        parser.add_argument("--spatial-pooling", type=str, default="mean")

    def slug_args(self, args):
        base = {"backbone": "vgg16", "feature_layer": "shallow"}
        base["percent_features"] = args.percent_features
        base["sp"] = args.spatial_pooling
        return base

    def build_model(self, device, args):
        feature_layers = ["features.3", "features.8"]
        ext, norm = make_multi_extractor("vgg16", feature_layers)
        model = IDFIQA_Hybrid(ext, norm,
                             device=device, percent_features_to_keep=args.percent_features, window_size=4,
                             alpha=args.alpha, beta=1.0, gamma=1.0, spatial_pooling=args.spatial_pooling)
        model = model.to(device)
        model.eval()
        return model

class IDFIQA_WSD(nn.Module):
    def __init__(self, feature_extractor, normalize, device):
        super().__init__()
        self.feature_extractor = feature_extractor.to(device).eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.normalize = normalize
        self.device = device
        
    def forward(self, ref, dist):
        out_r = self.feature_extractor(self.normalize(ref.to(self.device)))
        out_d = self.feature_extractor(self.normalize(dist.to(self.device)))
        
        wsd_scores = []
        for fr, fd in zip(out_r.values(), out_d.values()):
            n, c, h, w = fr.shape
            fr_flat = fr.view(n, c, -1)
            fd_flat = fd.view(n, c, -1)
            
            # 1D Wasserstein distance per channel
            fr_sorted, _ = torch.sort(fr_flat, dim=2)
            fd_sorted, _ = torch.sort(fd_flat, dim=2)
            
            # Absolute difference of sorted values
            diff = torch.abs(fr_sorted - fd_sorted)
            wsd = torch.mean(diff, dim=2) # Shape: (n, c)
            
            wsd_scores.append(wsd.mean(dim=1))
            
        # Sum over all layers
        total_wsd = sum(wsd_scores)
        
        return total_wsd

@register_experiment
class WSDExperiment(DefaultExperiment):
    name = "wsd"
    description = "Wasserstein Distance on shallow features"
    summary_prefix = "wsd"

    def add_arguments(self, parser):
        pass

    def slug_args(self, args):
        return {"backbone": "vgg16", "feature_layer": "shallow_wsd"}

    def build_model(self, device, args):
        feature_layers = ["features.3", "features.8"]
        ext, norm = make_multi_extractor("vgg16", feature_layers)
        model = IDFIQA_WSD(ext, norm, device)
        model = model.to(device)
        model.eval()
        return model


class IDFIQA_CausalHybrid(nn.Module):
    """
    Causal Hybrid: channel-importance-weighted feature comparison.

    Key ideas from Shen (CVPR 2025):
    - Channels that change significantly between ref and dist are
      'causally relevant' to perceptual quality.
    - Weight each channel's contribution by its sensitivity to distortion.
    - Combine Gram SSIM (texture) + DISTS SSIM (structure) + channel-WSD.
    - Per-layer normalization to prevent deep layers from dominating.
    """

    def __init__(self, feature_extractor, normalize, device,
                 alpha=1.0, beta=1.0, gamma=0.5, xi=1e-8,
                 causal_temp=1.0):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor.to(self.device).eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.normalize = normalize
        self.alpha = alpha  # Gram SSIM weight
        self.beta = beta    # DISTS SSIM weight
        self.gamma = gamma  # WSD weight
        self.xi = xi
        self.causal_temp = causal_temp

    def _channel_importance(self, fr, fd):
        """Compute per-channel importance based on feature difference magnitude."""
        # Mean absolute difference per channel
        diff = torch.mean(torch.abs(fr - fd), dim=(2, 3))  # (n, c)
        # Softmax to get normalized weights
        weights = F.softmax(diff / self.causal_temp, dim=1)  # (n, c)
        return weights

    def _gram_ssim(self, fr, fd):
        """Gram-matrix based SSIM."""
        n, c, h, w = fr.shape
        fr_flat = fr.view(n, c, h * w)
        fd_flat = fd.view(n, c, h * w)
        gr = torch.bmm(fr_flat, fr_flat.transpose(1, 2)) / (h * w)
        gd = torch.bmm(fd_flat, fd_flat.transpose(1, 2)) / (h * w)

        # Flatten gram matrices for SSIM computation
        gr_flat = gr.view(n, -1)
        gd_flat = gd.view(n, -1)

        mr = gr_flat.mean(dim=1)
        md = gd_flat.mean(dim=1)
        vr = gr_flat.var(dim=1, unbiased=False)
        vd = gd_flat.var(dim=1, unbiased=False)
        cov = ((gr_flat - mr.unsqueeze(1)) * (gd_flat - md.unsqueeze(1))).mean(dim=1)

        s_mean = (2 * mr * md + self.xi) / (mr ** 2 + md ** 2 + self.xi)
        s_var = (2 * cov + self.xi) / (vr + vd + self.xi)
        return s_mean * s_var  # (n,)

    def _dists_ssim(self, fr, fd, weights):
        """Channel-weighted DISTS SSIM."""
        # Per-channel mean and variance
        mr = torch.mean(fr, dim=(2, 3))  # (n, c)
        md = torch.mean(fd, dim=(2, 3))
        vr = torch.var(fr, dim=(2, 3), unbiased=False)
        vd = torch.var(fd, dim=(2, 3), unbiased=False)
        cov = torch.mean((fr - mr.unsqueeze(-1).unsqueeze(-1)) *
                         (fd - md.unsqueeze(-1).unsqueeze(-1)), dim=(2, 3))

        s_mean = (2 * mr * md + self.xi) / (mr ** 2 + md ** 2 + self.xi)
        s_var = (2 * cov + self.xi) / (vr + vd + self.xi)
        per_channel = s_mean * s_var  # (n, c)

        # Weighted average across channels
        return (per_channel * weights).sum(dim=1)  # (n,)

    def _channel_wsd(self, fr, fd, weights):
        """Channel-weighted 1D Wasserstein distance."""
        n, c, h, w = fr.shape
        fr_flat = fr.view(n, c, -1)
        fd_flat = fd.view(n, c, -1)

        fr_sorted, _ = torch.sort(fr_flat, dim=2)
        fd_sorted, _ = torch.sort(fd_flat, dim=2)

        wsd_per_ch = torch.mean(torch.abs(fr_sorted - fd_sorted), dim=2)  # (n, c)
        # Weighted sum, then negate (higher distance = lower quality)
        return -(wsd_per_ch * weights).sum(dim=1)  # (n,)

    def forward(self, ref, dist):
        out_r = self.feature_extractor(self.normalize(ref.to(self.device)))
        out_d = self.feature_extractor(self.normalize(dist.to(self.device)))

        gram_scores = []
        dists_scores = []
        wsd_scores = []

        for fr, fd in zip(out_r.values(), out_d.values()):
            if fr.dim() == 2:
                fr = fr.unsqueeze(-1).unsqueeze(-1)
                fd = fd.unsqueeze(-1).unsqueeze(-1)

            # Causal channel importance
            weights = self._channel_importance(fr, fd)

            gram_scores.append(self._gram_ssim(fr, fd))
            dists_scores.append(self._dists_ssim(fr, fd, weights))
            wsd_scores.append(self._channel_wsd(fr, fd, weights))

        # Per-layer normalize then average
        n_layers = len(gram_scores)
        gram_score = sum(gram_scores) / n_layers
        dists_score = sum(dists_scores) / n_layers
        wsd_score = sum(wsd_scores) / n_layers

        total = self.alpha + self.beta + self.gamma
        return (self.alpha * gram_score + self.beta * dists_score + self.gamma * wsd_score) / total


@register_experiment
class CausalHybridExperiment(DefaultExperiment):
    name = "causal_hybrid"
    description = "Causal channel-importance weighted Gram SSIM + DISTS + WSD"
    summary_prefix = "causal_hybrid"

    def add_arguments(self, parser):
        parser.add_argument("--alpha", type=float, default=1.0)
        parser.add_argument("--beta", type=float, default=1.0)
        parser.add_argument("--gamma", type=float, default=0.5)
        parser.add_argument("--causal-temp", type=float, default=1.0)
        parser.add_argument("--layers", type=str, default="all",
                            choices=["all", "shallow", "mid"])

    def slug_args(self, args):
        return {"backbone": "vgg16",
                "feature_layer": f"causal_{args.layers}_a{args.alpha}_b{args.beta}_g{args.gamma}_t{args.causal_temp}"}

    def build_model(self, device, args):
        if args.layers == "shallow":
            feature_layers = ["features.3", "features.8"]
        elif args.layers == "mid":
            feature_layers = ["features.3", "features.8", "features.15"]
        else:
            feature_layers = ["features.3", "features.8", "features.15", "features.22", "features.29"]

        ext, norm = make_multi_extractor("vgg16", feature_layers)
        model = IDFIQA_CausalHybrid(ext, norm, device,
                                     alpha=args.alpha, beta=args.beta,
                                     gamma=args.gamma, causal_temp=args.causal_temp)
        model = model.to(device)
        model.eval()
        return model

import torchvision.transforms.functional as TF

class IDFIQA_ShenCausalHybrid(nn.Module):
    def __init__(self, extractor, norm, device, pf=0.6, ws=4, temp=1.0):
        super().__init__()
        self.extractor = extractor.to(device)
        self.norm = norm.to(device)
        self.device = device
        self.pf = pf
        self.ws = ws
        self.xi = 1e-6

    def forward(self, img_ref, img_dist):
        img_ref = self.norm(img_ref.to(self.device))
        img_dist = self.norm(img_dist.to(self.device))
        
        # Abductive counterfactual: Blur the reference image
        img_blurred = TF.gaussian_blur(img_ref, kernel_size=[11, 11], sigma=[5.0, 5.0])
        
        feat_ref = self.extractor(img_ref)
        feat_dist = self.extractor(img_dist)
        feat_blur = self.extractor(img_blurred)

        scores = []
        for (fr, fd, fb) in zip(feat_ref.values(), feat_dist.values(), feat_blur.values()):
            n, c, h, w = fr.size()
            
            # Compute causal sensitivity: difference between original and blurred
            # High difference = highly sensitive (causal shallow features like edges)
            # Low difference = invariant (spurious deep semantics)
            sensitivity = torch.mean(torch.abs(fr - fb), dim=(2, 3)) # shape: [n, c]
            
            k = max(1, int(c * self.pf))
            # We want the most sensitive channels (highest difference)
            _, idx = torch.topk(sensitivity, k, dim=1) # shape: [n, k]
            
            # Gather top-k channels
            fr_top = torch.gather(fr, 1, idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w))
            fd_top = torch.gather(fd, 1, idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w))
            
            pad = self.ws // 2
            pool = nn.AvgPool2d(kernel_size=self.ws, stride=1, padding=pad)
            
            mr = pool(fr_top)
            md = pool(fd_top)
            vr = torch.clamp(pool(fr_top**2) - mr**2, min=0.0)
            vd = torch.clamp(pool(fd_top**2) - md**2, min=0.0)
            cov = pool(fr_top * fd_top) - mr * md
            
            s_mean = (2 * mr * md + self.xi) / (mr ** 2 + md ** 2 + self.xi)
            s_var = (2 * cov + self.xi) / (vr + vd + self.xi)
            dists_map = s_mean * s_var
            
            val = dists_map.mean(dim=(1, 2, 3))
            scores.append(val)
        
        return torch.stack(scores, dim=1).mean(dim=1)

@register_experiment
class ShenCausalExperiment(DefaultExperiment):
    name = "shen_causal"
    description = "Hybrid with Abductive Counterfactual (Shen et al 2025) Channel Weighting"
    summary_prefix = "shen_causal"

    def slug_args(self, args):
        return {"backbone": args.backbone, "feature_layer": f"causal_topk"}

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument("--backbone", type=str, default="vgg16")
        parser.add_argument("--pf", type=float, default=0.2)

    def build_model(self, device, args):
        feature_layers = ["features.3", "features.8", "features.15", "features.22", "features.29"]
        ext, norm = make_multi_extractor(args.backbone, feature_layers)
        model = IDFIQA_ShenCausalHybrid(ext, norm, device, pf=args.pf)
        model = model.to(device)
        model.eval()
        return model

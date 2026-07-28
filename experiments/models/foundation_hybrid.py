"""
Foundation Hybrid IQA Model (Training-Free / Zero-Shot).
Combines DINOv2-Base 2D Spatial Patch Features (DISTS SSIM + Patch Cosine + CLS)
with Multi-CNN Features (AlexNet + ConvNeXt-Tiny / VGG16),
enhanced with quantile worst-k patch weighting and multi-scale pyramid aggregation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CFG
from ..extractors import make_multi_extractor
from ..registry import DefaultExperiment, register_experiment
from ..helpers import run_slug, run_config


class DINOv2SpatialExtractor(nn.Module):
    """
    Extracts multi-layer 2D spatial feature maps and CLS tokens from DINOv2.
    """

    def __init__(self, model_name="dinov2_vitb14", device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model = self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # Layers 2, 5, 8, 11 capture multi-level structural details
        self.layer_indices = [2, 5, 8, 11]
        self.layer_weights = [0.35, 0.30, 0.20, 0.15]

    @torch.no_grad()
    def forward(self, x):
        # Input x shape: (B, 3, H, W)
        B, C, H, W = x.shape
        pad_h = (14 - H % 14) % 14
        pad_w = (14 - W % 14) % 14
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        H_pad, W_pad = x.shape[2], x.shape[3]
        h_patches, w_patches = H_pad // 14, W_pad // 14

        out_raw = self.model.get_intermediate_layers(
            x.to(self.device), n=self.layer_indices, return_class_token=True
        )

        spatial_maps = []
        cls_tokens = []

        for patch_tokens, cls_tok in out_raw:
            C_feat = patch_tokens.shape[-1]
            feat_2d = patch_tokens.permute(0, 2, 1).view(B, C_feat, h_patches, w_patches)
            spatial_maps.append(feat_2d)
            cls_tokens.append(cls_tok)

        return spatial_maps, cls_tokens


class IDFIQA_FoundationHybrid(nn.Module):
    """
    Training-Free Foundation Hybrid Model combining:
    1. DINOv2-Base multi-layer 2D spatial DISTS SSIM + Patch Cosine + CLS Similarity.
    2. Primary CNN (AlexNet) multi-layer DISTS + Gram SSIM.
    3. Secondary CNN (ConvNeXt-Tiny) multi-layer DISTS + Gram SSIM.
    4. Multi-scale pyramid processing (1.0x, 0.75x, 0.50x, 0.35x).
    """

    def __init__(self, dino_model_name="dinov2_vitb14",
                 cnn_backbone="alexnet",
                 secondary_cnn="convnext_tiny",
                 device=None,
                 w_dino=0.50,         # Weight for DINOv2
                 w_primary_cnn=0.30,  # Weight for AlexNet
                 w_secondary_cnn=0.20,# Weight for ConvNeXt-Tiny
                 worst_k_ratio=0.15,
                 ws=4,
                 pf=0.6,
                 xi=1e-6,
                 multiscale=True):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.w_dino = w_dino
        self.w_primary_cnn = w_primary_cnn
        self.w_secondary_cnn = w_secondary_cnn
        self.worst_k_ratio = worst_k_ratio
        self.ws = ws
        self.pf = pf
        self.xi = xi
        self.multiscale = multiscale

        # 1. DINOv2 Extractor
        self.dino = None
        try:
            self.dino = DINOv2SpatialExtractor(dino_model_name, device=self.device)
            print(f"[FoundationHybrid] Loaded DINOv2 ({dino_model_name})")
        except Exception as e:
            print(f"[FoundationHybrid] DINOv2 Warning: {e}")

        # 2. Primary CNN Extractor (AlexNet)
        self.primary_backbone = cnn_backbone
        if "alexnet" in cnn_backbone:
            p_layers = ["features.2", "features.5", "features.7", "features.9", "features.12"]
        elif "vgg" in cnn_backbone:
            p_layers = ["features.3", "features.8", "features.15", "features.22", "features.29"]
        else:
            p_layers = ["layer1", "layer2", "layer3", "layer4"]

        self.p_ext, self.p_norm = make_multi_extractor(cnn_backbone, p_layers)
        self.p_ext = self.p_ext.to(self.device).eval()
        for p in self.p_ext.parameters():
            p.requires_grad = False

        # 3. Secondary CNN Extractor (ConvNeXt-Tiny)
        self.secondary_backbone = secondary_cnn
        if secondary_cnn and secondary_cnn != "none":
            if "convnext" in secondary_cnn:
                s_layers = ["features.1", "features.3", "features.5", "features.7"]
            elif "vgg" in secondary_cnn:
                s_layers = ["features.3", "features.8", "features.15", "features.22", "features.29"]
            else:
                s_layers = ["layer1", "layer2", "layer3", "layer4"]

            self.s_ext, self.s_norm = make_multi_extractor(secondary_cnn, s_layers)
            self.s_ext = self.s_ext.to(self.device).eval()
            for p in self.s_ext.parameters():
                p.requires_grad = False
        else:
            self.s_ext = None

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

    def _compute_dino_score(self, ref, dist):
        if self.dino is None:
            return None

        maps_r, cls_r = self.dino(ref)
        maps_d, cls_d = self.dino(dist)

        weighted_layer_scores = []
        for (fr, fd, cr, cd), lw in zip(zip(maps_r, maps_d, cls_r, cls_d), self.dino.layer_weights):
            # 1. DISTS SSIM on 2D spatial feature map
            mr = torch.mean(fr, dim=(2, 3), keepdim=True)
            md = torch.mean(fd, dim=(2, 3), keepdim=True)
            vr = torch.var(fr, dim=(2, 3), unbiased=False, keepdim=True)
            vd = torch.var(fd, dim=(2, 3), unbiased=False, keepdim=True)
            cov = torch.mean((fr - mr) * (fd - md), dim=(2, 3), keepdim=True)

            s_mean = (2 * mr * md + self.xi) / (mr ** 2 + md ** 2 + self.xi)
            s_var = (2 * cov + self.xi) / (vr + vd + self.xi)
            dists_score = (s_mean * s_var).mean(dim=(1, 2, 3))

            # 2. Patch Cosine Similarity + Quantile Worst-K
            fr_norm = F.normalize(fr, p=2, dim=1)
            fd_norm = F.normalize(fd, p=2, dim=1)
            cos_sim_map = (fr_norm * fd_norm).sum(dim=1)  # (B, H', W')
            cos_flat = cos_sim_map.view(ref.shape[0], -1)

            mean_cos = cos_flat.mean(dim=1)
            k_val = max(1, int(cos_flat.shape[1] * self.worst_k_ratio))
            worst_cos = torch.topk(cos_flat, k_val, dim=1, largest=False)[0].mean(dim=1)
            patch_cos_score = 0.6 * mean_cos + 0.4 * worst_cos

            # 3. CLS Cosine Similarity
            cr_norm = F.normalize(cr, p=2, dim=1)
            cd_norm = F.normalize(cd, p=2, dim=1)
            cls_score = (cr_norm * cd_norm).sum(dim=1)

            layer_score = 0.45 * dists_score + 0.45 * patch_cos_score + 0.1 * cls_score
            weighted_layer_scores.append(lw * layer_score)

        total_weight = sum(self.dino.layer_weights)
        return torch.stack(weighted_layer_scores, dim=0).sum(dim=0) / total_weight

    def _compute_cnn_score(self, ext, norm, ref, dist):
        out_r = ext(norm(ref.to(self.device)))
        out_d = ext(norm(dist.to(self.device)))

        dists_scores = []
        gram_scores = []

        for k in out_r.keys():
            fr = out_r[k]
            fd = out_d[k]
            if fr.dim() == 2:
                fr = fr.unsqueeze(-1).unsqueeze(-1)
                fd = fd.unsqueeze(-1).unsqueeze(-1)

            # Local DISTS
            pad = self.ws // 2
            pool = nn.AvgPool2d(kernel_size=self.ws, stride=1, padding=pad)
            mr = pool(fr)
            md = pool(fd)
            vr = torch.clamp(pool(fr ** 2) - mr ** 2, min=0.0)
            vd = torch.clamp(pool(fd ** 2) - md ** 2, min=0.0)
            cov = pool(fr * fd) - mr * md
            s_mean = (2 * mr * md + self.xi) / (mr ** 2 + md ** 2 + self.xi)
            s_var = (2 * cov + self.xi) / (vr + vd + self.xi)
            dists_map = s_mean * s_var
            dists_scores.append(dists_map.mean(dim=(1, 2, 3)))

            # Gram SSIM
            if self.pf < 1.0:
                fr_sel, fd_sel = self._select_channels(fr, fd)
            else:
                fr_sel, fd_sel = fr, fd

            if fr_sel.shape[2] >= self.ws and fr_sel.shape[3] >= self.ws:
                gr = self._gram(fr_sel)
                gd = self._gram(fd_sel)
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
                n, c = fr_sel.shape[:2]
                fr_flat = fr_sel.view(n, c, -1)
                fd_flat = fd_sel.view(n, c, -1)
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

        dists_score = torch.stack(dists_scores, dim=0).mean(dim=0)
        gram_score = torch.stack(gram_scores, dim=0).mean(dim=0)
        return 0.6 * dists_score + 0.4 * gram_score

    def _single_scale_forward(self, ref, dist):
        with torch.no_grad():
            dino_score = self._compute_dino_score(ref, dist)
            primary_cnn_score = self._compute_cnn_score(self.p_ext, self.p_norm, ref, dist)

            if self.s_ext is not None:
                secondary_cnn_score = self._compute_cnn_score(self.s_ext, self.s_norm, ref, dist)
            else:
                secondary_cnn_score = None

            if dino_score is not None and secondary_cnn_score is not None:
                total_w = self.w_dino + self.w_primary_cnn + self.w_secondary_cnn
                return (self.w_dino * dino_score +
                        self.w_primary_cnn * primary_cnn_score +
                        self.w_secondary_cnn * secondary_cnn_score) / total_w
            elif dino_score is not None:
                total_w = self.w_dino + self.w_primary_cnn
                return (self.w_dino * dino_score + self.w_primary_cnn * primary_cnn_score) / total_w
            else:
                return primary_cnn_score

    def forward(self, ref, dist):
        s1 = self._single_scale_forward(ref, dist)

        if not self.multiscale:
            return s1

        # Multi-scale 0.75x, 0.50x, 0.35x
        ref_75 = F.interpolate(ref, scale_factor=0.75, mode="bilinear", align_corners=False)
        dist_75 = F.interpolate(dist, scale_factor=0.75, mode="bilinear", align_corners=False)
        s75 = self._single_scale_forward(ref_75, dist_75)

        ref_50 = F.interpolate(ref, scale_factor=0.50, mode="bilinear", align_corners=False)
        dist_50 = F.interpolate(dist, scale_factor=0.50, mode="bilinear", align_corners=False)
        s50 = self._single_scale_forward(ref_50, dist_50)

        ref_35 = F.interpolate(ref, scale_factor=0.35, mode="bilinear", align_corners=False)
        dist_35 = F.interpolate(dist, scale_factor=0.35, mode="bilinear", align_corners=False)
        s35 = self._single_scale_forward(ref_35, dist_35)

        return 0.45 * s1 + 0.30 * s75 + 0.15 * s50 + 0.10 * s35


def _build_foundation_hybrid(device, dino_model="dinov2_vitb14", cnn_backbone="alexnet",
                             secondary_cnn="convnext_tiny",
                             w_dino=0.50, w_primary_cnn=0.30, w_secondary_cnn=0.20,
                             multiscale=True):
    return IDFIQA_FoundationHybrid(
        dino_model_name=dino_model,
        cnn_backbone=cnn_backbone,
        secondary_cnn=secondary_cnn,
        device=device,
        w_dino=w_dino,
        w_primary_cnn=w_primary_cnn,
        w_secondary_cnn=w_secondary_cnn,
        multiscale=multiscale,
    )


@register_experiment
class FoundationHybridExperiment(DefaultExperiment):
    name = "foundation_hybrid"
    description = "Training-Free Foundation Hybrid (DINOv2 + AlexNet + ConvNeXt)"
    summary_prefix = "foundation_hybrid"

    def add_arguments(self, parser):
        parser.add_argument("--dino-model", type=str, default="dinov2_vitb14",
                            choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"])
        parser.add_argument("--cnn-backbone", type=str, default="alexnet",
                            choices=["vgg16", "convnext_base", "convnext_tiny", "alexnet", "resnet50"])
        parser.add_argument("--secondary-cnn", type=str, default="convnext_tiny",
                            choices=["convnext_tiny", "vgg16", "none"])
        parser.add_argument("--w-dino", type=float, default=0.50, help="Weight for DINOv2")
        parser.add_argument("--w-primary-cnn", type=float, default=0.30, help="Weight for Primary CNN (AlexNet)")
        parser.add_argument("--w-secondary-cnn", type=float, default=0.20, help="Weight for Secondary CNN")
        parser.add_argument("--no-multiscale", action="store_true", help="Disable multi-scale pyramid")

    def slug_args(self, args):
        ms_str = "single" if getattr(args, "no_multiscale", False) else "ms"
        dino_name = getattr(args, "dino_model", "dinov2_vitb14")
        cnn_name = getattr(args, "cnn_backbone", "alexnet")
        sec_name = getattr(args, "secondary_cnn", "convnext_tiny")
        return {
            "backbone": f"{dino_name}_{cnn_name}_{sec_name}",
            "feature_layer": f"fh_wd{args.w_dino}_wp{args.w_primary_cnn}_ws{args.w_secondary_cnn}_{ms_str}"
        }

    def build_model(self, device, args):
        ms = not getattr(args, "no_multiscale", False)
        return _build_foundation_hybrid(
            device=device,
            dino_model=args.dino_model,
            cnn_backbone=args.cnn_backbone,
            secondary_cnn=args.secondary_cnn,
            w_dino=args.w_dino,
            w_primary_cnn=args.w_primary_cnn,
            w_secondary_cnn=args.w_secondary_cnn,
            multiscale=ms
        )

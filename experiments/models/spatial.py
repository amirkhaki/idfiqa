"""Spatial causal patch selection variant.

Measures per-region importance via gradient of the quality score
w.r.t. input image patches.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class IDFIQA_SpatialCausal(nn.Module):
    """
    Spatial causal variant of IDFIQA.
    Measures per-patch sensitivity via gradient of the quality score
    w.r.t. input features, then computes sensitivity-weighted quality score.
    """

    def __init__(self, feature_extractor, normalize,
                 feature_node_key="features",
                 device=None,
                 percent_features_to_keep=0.6,
                 window_size=4,
                 patch_size=32,
                 xi=1e-8):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor.to(self.device).eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.normalize = normalize
        self.feature_node_key = feature_node_key
        self.pf = percent_features_to_keep
        self.ws = window_size
        self.ps = patch_size
        self.xi = xi

    def _features(self, img):
        out = self.feature_extractor(self.normalize(img.to(self.device)))
        return out[self.feature_node_key] if isinstance(out, dict) else out

    @staticmethod
    def _gram(feat):
        n, c, h, w = feat.shape
        f = feat.view(n, c, h * w)
        return torch.bmm(f, f.transpose(1, 2)) / (h * w)

    def _compute_score(self, feat_ref, feat_dist):
        gr = self._gram(feat_ref)
        gd = self._gram(feat_dist)
        gr_u = F.unfold(gr.unsqueeze(1), kernel_size=self.ws, stride=1).transpose(1, 2)
        gd_u = F.unfold(gd.unsqueeze(1), kernel_size=self.ws, stride=1).transpose(1, 2)
        vr = torch.var(gr_u, dim=2, unbiased=False)
        vd = torch.var(gd_u, dim=2, unbiased=False)
        mr = torch.mean(gr_u, dim=2, keepdim=True)
        md = torch.mean(gd_u, dim=2, keepdim=True)
        cov = torch.mean((gr_u - mr) * (gd_u - md), dim=2)
        local = (2 * cov + self.xi) / (vr + vd + self.xi)
        return local.mean(dim=1)

    def _get_patch_scores(self, feat_ref, feat_dist, grid_h, grid_w):
        n, c, h, w = feat_ref.shape
        fsh = max(1, h // grid_h)
        fsw = max(1, w // grid_w)
        scores = torch.zeros(grid_h, grid_w, device=feat_ref.device)
        for i in range(grid_h):
            for j in range(grid_w):
                y1, y2 = i * fsh, min((i + 1) * fsh, h)
                x1, x2 = j * fsw, min((j + 1) * fsw, w)
                pr = feat_ref[:, :, y1:y2, x1:x2]
                pd = feat_dist[:, :, y1:y2, x1:x2]
                scores[i, j] = torch.norm(pr.flatten(1) - pd.flatten(1), dim=1).mean()
        return scores

    def forward(self, ref, dist, return_map=False):
        feat_ref = self._features(ref)
        feat_dist = self._features(dist)

        n, c, h, w = feat_ref.shape
        grid_h, grid_w = h // max(1, h // (ref.shape[2] // self.ps)), w // max(1, w // (ref.shape[3] // self.ps))
        base_scores = self._get_patch_scores(feat_ref, feat_dist, grid_h, grid_w)

        with torch.enable_grad():
            feat_ref_g = feat_ref.detach().requires_grad_(True)
            feat_dist_g = feat_dist.detach().requires_grad_(True)
            score = self._compute_score(feat_ref_g, feat_dist_g)
            score.sum().backward()

        grad_ref = feat_ref_g.grad.abs()
        grad_dist = feat_dist_g.grad.abs()
        importance = (grad_ref + grad_dist).mean(dim=1)

        fsh = h // grid_h
        fsw = w // grid_w
        sensitivity = torch.zeros(grid_h, grid_w, device=ref.device)
        for i in range(grid_h):
            for j in range(grid_w):
                y1, y2 = i * fsh, min((i + 1) * fsh, h)
                x1, x2 = j * fsw, min((j + 1) * fsw, w)
                sensitivity[i, j] = importance[:, y1:y2, x1:x2].mean()

        final = (sensitivity * base_scores).sum() / (sensitivity.sum() + 1e-8)

        if return_map:
            return final, sensitivity
        return final

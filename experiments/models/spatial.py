"""Spatial causal patch selection variant.

Perturbs input image patches to measure per-region importance
for quality assessment.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class IDFIQA_SpatialCausal(nn.Module):
    """
    Spatial causal variant of IDFIQA.
    Measures per-patch sensitivity by perturbing input regions,
    then computes sensitivity-weighted quality score.
    """

    def __init__(self, feature_extractor, normalize,
                 feature_node_key="features",
                 device=None,
                 percent_features_to_keep=0.6,
                 window_size=4,
                 patch_size=32,
                 patch_score_method="full",
                 max_intensity=0.1,
                 n_steps=10,
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
        self.patch_score_method = patch_score_method
        self.max_intensity = max_intensity
        self.n_steps = n_steps
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

    def _crop_patch(self, img, i, j):
        _, _, H, W = img.shape
        y1, y2 = i * self.ps, min((i + 1) * self.ps, H)
        x1, x2 = j * self.ps, min((j + 1) * self.ps, W)
        return img[:, :, y1:y2, x1:x2]

    def _patch_score_full(self, ref, dist, i, j):
        patch_ref = self._crop_patch(ref, i, j)
        patch_dist = self._crop_patch(dist, i, j)
        feat_ref = self._features(patch_ref)
        feat_dist = self._features(patch_dist)
        return self._compute_score(feat_ref, feat_dist)

    def _patch_score_l2(self, feat_ref, feat_dist, i, j):
        n, c, h, w = feat_ref.shape
        fsh = max(1, self.ps // 16)
        y1, y2 = i * fsh, min((i + 1) * fsh, h)
        x1, x2 = j * fsh, min((j + 1) * fsh, w)
        pr = feat_ref[:, :, y1:y2, x1:x2].flatten(1)
        pd = feat_dist[:, :, y1:y2, x1:x2].flatten(1)
        return torch.norm(pr - pd, dim=1)

    def _build_perturbed_batch(self, ref, dist, intensity):
        n, c, h, w = ref.shape
        grid_h, grid_w = h // self.ps, w // self.ps

        ref_list = []
        dist_list = []
        coords = []

        for i in range(grid_h):
            for j in range(grid_w):
                y1, y2 = i * self.ps, min((i + 1) * self.ps, h)
                x1, x2 = j * self.ps, min((j + 1) * self.ps, w)
                ph, pw = y2 - y1, x2 - x1

                patch_mean = ((ref[:, :, y1:y2, x1:x2] +
                               dist[:, :, y1:y2, x1:x2]) / 2).mean(dim=[2, 3], keepdim=True)
                noise = torch.randn(n, c, ph, pw, device=ref.device) * patch_mean * intensity

                r = ref.clone()
                d = dist.clone()
                r[:, :, y1:y2, x1:x2] += noise
                d[:, :, y1:y2, x1:x2] += noise

                ref_list.append(r)
                dist_list.append(d)
                coords.append((i, j))

        ref_batch = torch.cat(ref_list, dim=0)
        dist_batch = torch.cat(dist_list, dim=0)
        return ref_batch, dist_batch, coords

    def _get_patch_scores(self, ref, dist, feat_ref, feat_dist, grid_h, grid_w):
        scores = torch.zeros(grid_h, grid_w, device=ref.device)
        for i in range(grid_h):
            for j in range(grid_w):
                if self.patch_score_method == "full":
                    scores[i, j] = self._patch_score_full(ref, dist, i, j).mean()
                else:
                    scores[i, j] = self._patch_score_l2(feat_ref, feat_dist, i, j).mean()
        return scores

    def forward(self, ref, dist, return_map=False):
        n, c, h, w = ref.shape
        feat_ref = self._features(ref)
        feat_dist = self._features(dist)

        grid_h, grid_w = h // self.ps, w // self.ps
        base_scores = self._get_patch_scores(ref, dist, feat_ref, feat_dist, grid_h, grid_w)

        with torch.no_grad():
            intensity_values = torch.linspace(
                self.max_intensity / self.n_steps,
                self.max_intensity,
                self.n_steps,
                device=ref.device,
            )

            sensitivity = torch.zeros(grid_h, grid_w, device=ref.device)

            for s in range(self.n_steps):
                ref_batch, dist_batch, coords = self._build_perturbed_batch(
                    ref, dist, intensity_values[s]
                )

                if self.patch_score_method == "full":
                    feat_ref_batch = self._features(ref_batch)
                    feat_dist_batch = self._features(dist_batch)
                    for idx, (i, j) in enumerate(coords):
                        fr = feat_ref_batch[idx:idx+1]
                        fd = feat_dist_batch[idx:idx+1]
                        score = self._compute_score(fr, fd)
                        sensitivity[i, j] += torch.abs(base_scores[i, j] - score.mean())
                else:
                    feat_ref_batch = self._features(ref_batch)
                    feat_dist_batch = self._features(dist_batch)
                    for idx, (i, j) in enumerate(coords):
                        fr = feat_ref_batch[idx:idx+1]
                        fd = feat_dist_batch[idx:idx+1]
                        score = self._patch_score_l2(fr, fd, i, j)
                        sensitivity[i, j] += torch.abs(base_scores[i, j] - score.mean())

            sensitivity /= self.n_steps

        final = (sensitivity * base_scores).sum() / (sensitivity.sum() + 1e-8)

        if return_map:
            return final, sensitivity
        return final

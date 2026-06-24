"""Causal channel selection variant."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class IDFIQA_Causal(nn.Module):
    """
    Causal channel selection variant of IDFIQA.
    For each channel, adds noise and measures how much the final score changes.
    Channels with larger score delta are deemed more important.
    """

    def __init__(self, feature_extractor, normalize,
                 feature_node_key="features",
                 device=None,
                 percent_features_to_keep=0.6,
                 window_size=4,
                 noise_std=0.1,
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
        self.noise_std = noise_std
        self.xi = xi

    def _features(self, img):
        out = self.feature_extractor(self.normalize(img.to(self.device)))
        return out[self.feature_node_key] if isinstance(out, dict) else out

    @staticmethod
    def _gram(feat):
        n, c, h, w = feat.shape
        f = feat.view(n, c, h * w)
        return torch.bmm(f, f.transpose(1, 2)) / (h * w)

    def _score_from_selected(self, feat_ref, feat_dist, channel_mask):
        n, c, h, w = feat_ref.shape
        idx = torch.where(channel_mask)[0]
        if idx.numel() == 0:
            return torch.zeros(n, device=feat_ref.device)
        idx_r = idx.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(n, -1, h, w)
        s_ref = torch.gather(feat_ref, 1, idx_r)
        _, _, hd, wd = feat_dist.shape
        idx_d = idx.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(n, -1, hd, wd)
        s_dist = torch.gather(feat_dist, 1, idx_d)
        gr = self._gram(s_ref)
        gd = self._gram(s_dist)
        gr_u = F.unfold(gr.unsqueeze(1), kernel_size=self.ws, stride=1).transpose(1, 2)
        gd_u = F.unfold(gd.unsqueeze(1), kernel_size=self.ws, stride=1).transpose(1, 2)
        vr = torch.var(gr_u, dim=2, unbiased=False)
        vd = torch.var(gd_u, dim=2, unbiased=False)
        mr = torch.mean(gr_u, dim=2, keepdim=True)
        md = torch.mean(gd_u, dim=2, keepdim=True)
        cov = torch.mean((gr_u - mr) * (gd_u - md), dim=2)
        local = (2 * cov + self.xi) / (vr + vd + self.xi)
        return local.mean(dim=1)

    def _select_channels(self, feat_ref, feat_dist):
        n, c, h, w = feat_ref.shape
        k = max(1, int(c * self.pf))

        with torch.no_grad():
            base_score = self._score_from_selected(
                feat_ref, feat_dist,
                torch.ones(c, dtype=torch.bool, device=feat_ref.device)
            )

            sensitivities = torch.zeros(c, device=feat_ref.device)
            for ch in range(c):
                noise = torch.zeros_like(feat_ref)
                noise[:, ch] = torch.randn(n, 1, h, w, device=feat_ref.device) * self.noise_std
                noisy_ref = feat_ref + noise
                noisy_score = self._score_from_selected(
                    noisy_ref, feat_dist,
                    torch.ones(c, dtype=torch.bool, device=feat_ref.device)
                )
                sensitivities[ch] = torch.mean(torch.abs(base_score - noisy_score))

        _, idx = torch.topk(sensitivities, k)
        idx_r = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        s_ref = torch.gather(feat_ref, 1, idx_r)
        _, _, hd, wd = feat_dist.shape
        idx_d = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, hd, wd)
        s_dist = torch.gather(feat_dist, 1, idx_d)
        return s_ref, s_dist

    def forward(self, ref, dist):
        fr = self._features(ref)
        fd = self._features(dist)
        fr, fd = self._select_channels(fr, fd)
        gr = self._gram(fr)
        gd = self._gram(fd)
        gr_u = F.unfold(gr.unsqueeze(1), kernel_size=self.ws, stride=1).transpose(1, 2)
        gd_u = F.unfold(gd.unsqueeze(1), kernel_size=self.ws, stride=1).transpose(1, 2)
        vr = torch.var(gr_u, dim=2, unbiased=False)
        vd = torch.var(gd_u, dim=2, unbiased=False)
        mr = torch.mean(gr_u, dim=2, keepdim=True)
        md = torch.mean(gd_u, dim=2, keepdim=True)
        cov = torch.mean((gr_u - mr) * (gd_u - md), dim=2)
        local = (2 * cov + self.xi) / (vr + vd + self.xi)
        return local.mean(dim=1)

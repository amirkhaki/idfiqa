"""Causal channel selection variant.

Adopts the multi-intensity intervention approach from DeepCausalQuality:
- Tests multiple noise intensity levels
- Perturbs both reference and distorted features
- Scales noise by feature magnitude
- Selects channels by accumulated sensitivity
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class IDFIQA_Causal(nn.Module):
    """
    Causal channel selection variant of IDFIQA.
    Uses multi-intensity noise intervention (inspired by DeepCausalQuality)
    to measure per-channel sensitivity for channel selection.
    """

    def __init__(self, feature_extractor, normalize,
                 feature_node_key="features",
                 device=None,
                 percent_features_to_keep=0.6,
                 window_size=4,
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
        """Quality score from gram matrices + windowed SSIM-like local similarity."""
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

    def _select_channels(self, feat_ref, feat_dist):
        n, c, h, w = feat_ref.shape
        k = max(1, int(c * self.pf))

        with torch.no_grad():
            base_score = self._compute_score(feat_ref, feat_dist)

            feat_mean = (feat_ref + feat_dist) / 2

            intensity_values = torch.linspace(
                self.max_intensity / self.n_steps,
                self.max_intensity,
                self.n_steps,
                device=feat_ref.device,
            )

            sensitivities = torch.zeros(c, device=feat_ref.device)

            for ch in range(c):
                noise_base = torch.randn(
                    self.n_steps, n, 1, h, w, device=feat_ref.device
                )
                channel_mean = feat_mean[:, ch : ch + 1, :, :].unsqueeze(0)
                noise = noise_base * channel_mean * intensity_values.view(
                    -1, 1, 1, 1, 1
                )

                noisy_ref = feat_ref.unsqueeze(0) + noise
                noisy_dist = feat_dist.unsqueeze(0) + noise

                noisy_ref_flat = noisy_ref.reshape(self.n_steps * n, c, h, w)
                noisy_dist_flat = noisy_dist.reshape(self.n_steps * n, c, h, w)
                noisy_scores = self._compute_score(noisy_ref_flat, noisy_dist_flat)
                noisy_scores = noisy_scores.reshape(self.n_steps, n)

                sensitivities[ch] = torch.mean(
                    torch.abs(base_score.unsqueeze(0) - noisy_scores)
                )

        _, idx = torch.topk(sensitivities, k)
        idx = idx.unsqueeze(0).expand(n, -1)
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
        return self._compute_score(fr, fd)

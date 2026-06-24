"""Patch-based weighted IDFIQA model."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedPatchIDFIQA(nn.Module):
    """
    Patch-based weighted IDFIQA (new method).
    The extractor must return a dict with 'features' and 'weights' keys.
    Aggregation: 'max' | 'mean' | 'softmax_<temp>' | 'uniform'
    """

    def __init__(self, feature_extractor, normalize,
                 device=None,
                 percent_features_to_keep=0.6,
                 window_size=4,
                 patch_size=8,
                 aggregation="max",
                 xi=1e-8):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor.to(self.device).eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.normalize = normalize
        self.pf = percent_features_to_keep
        self.ws = window_size
        self.ps = patch_size
        self.aggregation = aggregation
        self.xi = xi

    def _extract(self, img):
        from ..config import CFG
        out = self.feature_extractor(self.normalize(img.to(self.device)))
        feat = out["features"]
        wt = out["weights"]
        if CFG.diagnose and not getattr(self, "_diag_printed", False):
            self._diag_printed = True
            same_obj = feat is wt
            same_ptr = feat.data_ptr() == wt.data_ptr()
            shape_msg = (f"  diff_norm={(feat-wt).norm():.6g}"
                         if feat.shape == wt.shape else "  shapes differ")
            print(f"  [diag] feat={tuple(feat.shape)} norm={feat.norm():.4g}"
                  f"  wt={tuple(wt.shape)} norm={wt.norm():.4g}"
                  f"  same_obj={same_obj}  same_ptr={same_ptr}{shape_msg}")
        return feat, wt

    @staticmethod
    def _gram(feat):
        n, c, h, w = feat.shape
        f = feat.view(n, c, h * w)
        return torch.bmm(f, f.transpose(1, 2)) / (h * w)

    def _patch_score(self, pr, pd):
        n, c, ph, pw = pr.shape
        k = max(1, int(c * self.pf))
        var = torch.var(pr, dim=(2, 3), unbiased=False)
        _, idx = torch.topk(var, k, dim=1)
        idx_e = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, ph, pw)
        pr = torch.gather(pr, 1, idx_e)
        pd = torch.gather(pd, 1, idx_e)
        gr = self._gram(pr)
        gd = self._gram(pd)
        gr_u = F.unfold(gr.unsqueeze(1), kernel_size=self.ws, stride=1).transpose(1, 2)
        gd_u = F.unfold(gd.unsqueeze(1), kernel_size=self.ws, stride=1).transpose(1, 2)
        vr = torch.var(gr_u, dim=2, unbiased=False)
        vd = torch.var(gd_u, dim=2, unbiased=False)
        mr = gr_u.mean(dim=2, keepdim=True)
        md = gd_u.mean(dim=2, keepdim=True)
        cov = ((gr_u - mr) * (gd_u - md)).mean(dim=2)
        return ((2 * cov + self.xi) / (vr + vd + self.xi)).mean(dim=1)

    def _all_patch_scores(self, fr, fd, nh, nw, ph, pw):
        B, C, H, W = fr.shape
        P = nh * nw

        pr_flat = F.unfold(fr, kernel_size=(ph, pw), stride=(ph, pw))
        pd_flat = F.unfold(fd, kernel_size=(ph, pw), stride=(ph, pw))

        pr = pr_flat.permute(0, 2, 1).reshape(B * P, C, ph, pw)
        pd = pd_flat.permute(0, 2, 1).reshape(B * P, C, ph, pw)

        k = max(1, int(C * self.pf))
        var = torch.var(pr, dim=(2, 3), unbiased=False)
        _, idx = torch.topk(var, k, dim=1)
        idx_e = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, ph, pw)
        pr = torch.gather(pr, 1, idx_e)
        pd = torch.gather(pd, 1, idx_e)

        def gram_batched(x):
            bp, k, h, w = x.shape
            f = x.view(bp, k, h * w)
            return torch.bmm(f, f.transpose(1, 2)) / (h * w)

        gr = gram_batched(pr)
        gd = gram_batched(pd)

        gr_u = F.unfold(gr.unsqueeze(1), kernel_size=self.ws, stride=1).transpose(1, 2)
        gd_u = F.unfold(gd.unsqueeze(1), kernel_size=self.ws, stride=1).transpose(1, 2)
        vr = torch.var(gr_u, dim=2, unbiased=False)
        vd = torch.var(gd_u, dim=2, unbiased=False)
        mr = gr_u.mean(dim=2, keepdim=True)
        md = gd_u.mean(dim=2, keepdim=True)
        cov = ((gr_u - mr) * (gd_u - md)).mean(dim=2)
        scores_flat = ((2 * cov + self.xi) / (vr + vd + self.xi)).mean(dim=1)
        return scores_flat.view(B, P)

    def forward(self, ref, dist):
        fr, _ = self._extract(ref)
        fd, _ = self._extract(dist)

        n, c, h, w = fr.shape

        nh = max(1, h // self.ps)
        nw = max(1, w // self.ps)
        ph = h // nh
        pw = w // nw

        fr = fr[:, :, :nh * ph, :nw * pw]
        fd = fd[:, :, :nh * ph, :nw * pw]

        if ph < self.ws or pw < self.ws:
            return self._patch_score(fr, fd)

        scores = self._all_patch_scores(fr, fd, nh, nw, ph, pw)
        return torch.mean(scores, dim=1)

    def _patch_weight_scalar(self, wmap_patches):
        if self.aggregation == "uniform":
            return torch.ones(wmap_patches.shape[:2],
                              device=wmap_patches.device,
                              dtype=wmap_patches.dtype)
        elif self.aggregation == "max":
            return wmap_patches.max(dim=2).values
        elif self.aggregation.startswith("softmax"):
            temp = float(self.aggregation.split("_")[1]) if "_" in self.aggregation else 1.0
            patch_weights = torch.softmax(wmap_patches / temp, dim=2)
            return (patch_weights * wmap_patches).sum(dim=2)
        else:
            return wmap_patches.mean(dim=2)

    def _normalise_weights(self, raw_weights):
        if self.aggregation == "uniform":
            return torch.ones_like(raw_weights) / raw_weights.shape[1]
        elif self.aggregation == "max":
            idx = raw_weights.argmax(dim=1, keepdim=True)
            return torch.zeros_like(raw_weights).scatter_(1, idx, 1.0)
        elif self.aggregation.startswith("softmax"):
            temp = float(self.aggregation.split("_")[1]) if "_" in self.aggregation else 1.0
            return torch.softmax(raw_weights / temp, dim=1)
        else:
            return raw_weights / (raw_weights.sum(dim=1, keepdim=True) + 1e-8)

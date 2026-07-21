# Spatial Causal Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use compose:subagent (recommended) or compose:execute to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a spatial causal model that identifies important image regions by perturbing input patches and measuring quality score changes.

**Architecture:** New model `IDFIQA_SpatialCausal` that perturbs 32×32 input patches, re-extracts features, and computes per-patch sensitivity. Supports two per-patch score methods: "full" (standalone image processing) and "l2" (feature cropping). Final score is sensitivity-weighted average of per-patch scores.

**Tech Stack:** PyTorch, torchvision, matplotlib (for visualization)

---

### Task 1: Create the spatial causal model

**Covers:** Core model implementation

**Files:**
- Create: `experiments/models/spatial.py`

- [ ] **Step 1: Write the spatial causal model**

```python
"""Spatial causal channel selection variant.

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
        fsh = self.ps // 16
        fsh = max(1, fsh)
        y1, y2 = i * fsh, min((i + 1) * fsh, h)
        x1, x2 = j * fsh, min((j + 1) * fsh, w)
        pr = feat_ref[:, :, y1:y2, x1:x2].flatten(1)
        pd = feat_dist[:, :, y1:y2, x1:x2].flatten(1)
        return torch.norm(pr - pd, dim=1)

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

        base_scores = self._get_patch_scores(
            ref, dist, feat_ref, feat_dist,
            h // self.ps, w // self.ps
        )

        with torch.no_grad():
            intensity_values = torch.linspace(
                self.max_intensity / self.n_steps,
                self.max_intensity,
                self.n_steps,
                device=ref.device,
            )

            sensitivity = torch.zeros(h // self.ps, w // self.ps, device=ref.device)

            for i in range(h // self.ps):
                for j in range(w // self.ps):
                    y1, y2 = i * self.ps, min((i + 1) * self.ps, h)
                    x1, x2 = j * self.ps, min((j + 1) * self.ps, w)

                    noise_base = torch.randn(
                        self.n_steps, n, 1, self.ps, self.ps, device=ref.device
                    )
                    patch_mean = ((ref[:, :, y1:y2, x1:x2] +
                                   dist[:, :, y1:y2, x1:x2]) / 2).mean(dim=[2, 3], keepdim=True)
                    noise = noise_base * patch_mean * intensity_values.view(-1, 1, 1, 1, 1)

                    ref_patch = ref[:, :, y1:y2, x1:x2].unsqueeze(0) + noise
                    dist_patch = dist[:, :, y1:y2, x1:x2].unsqueeze(0) + noise

                    ref_perturbed = ref.clone().unsqueeze(0).expand(self.n_steps, -1, -1, -1, -1)
                    dist_perturbed = dist.clone().unsqueeze(0).expand(self.n_steps, -1, -1, -1, -1)
                    ref_perturbed[:, :, y1:y2, x1:x2] = ref_patch
                    dist_perturbed[:, :, y1:y2, x1:x2] = dist_patch

                    ref_flat = ref_perturbed.reshape(self.n_steps * n, c, h, w)
                    dist_flat = dist_perturbed.reshape(self.n_steps * n, c, h, w)

                    if self.patch_score_method == "full":
                        perturbed = torch.stack([
                            self._patch_score_full(ref_flat[b:b+1], dist_flat[b:b+1], i, j)
                            for b in range(self.n_steps * n)
                        ])
                    else:
                        feat_ref_flat = self._features(ref_flat)
                        feat_dist_flat = self._features(dist_flat)
                        perturbed = torch.stack([
                            self._patch_score_l2(feat_ref_flat[b:b+1], feat_dist_flat[b:b+1], i, j)
                            for b in range(self.n_steps * n)
                        ])

                    perturbed = perturbed.reshape(self.n_steps, n)
                    base = base_scores[i, j].expand_as(perturbed[:, :1])
                    sensitivity[i, j] = torch.mean(torch.abs(base - perturbed))

        final = (sensitivity * base_scores).sum() / (sensitivity.sum() + 1e-8)

        if return_map:
            return final, sensitivity
        return final
```

- [ ] **Step 2: Verify syntax**

Run: `python3 -m py_compile experiments/models/spatial.py`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add experiments/models/spatial.py
git commit -m "feat: add spatial causal model"
```

---

### Task 2: Register model in package and factory

**Covers:** Model registration, factory integration

**Files:**
- Modify: `experiments/models/__init__.py`
- Modify: `experiments/factories.py`
- Modify: `experiments/config.py`

- [ ] **Step 1: Add import to __init__.py**

Add to `experiments/models/__init__.py`:
```python
from .spatial import IDFIQA_SpatialCausal

__all__ = ["IDFIQA_Baseline", "IDFIQA_Causal", "WeightedPatchIDFIQA", "IDFIQA_SpatialCausal"]
```

- [ ] **Step 2: Add config parameters to config.py**

Add after `n_steps: int = 10` in the `Config` class:
```python
    patch_size_spatial: int = 32
    patch_score_method: str = "full"
```

- [ ] **Step 3: Add factory function to factories.py**

Add at the end of `factories.py`:
```python
def make_spatial_model(device, *, backbone=None, feature_layer=None,
                       pf=None, ws=None, ps=None, patch_score_method=None,
                       max_intensity=None, n_steps=None):
    backbone = backbone or CFG.backbone
    feature_layer = feature_layer or CFG.get_feature_layer(backbone)
    pf = pf if pf is not None else CFG.percent_features
    ws = ws if ws is not None else CFG.window_size
    ps = ps if ps is not None else CFG.patch_size_spatial
    patch_score_method = patch_score_method or CFG.patch_score_method
    max_intensity = max_intensity if max_intensity is not None else CFG.max_intensity
    n_steps = n_steps if n_steps is not None else CFG.n_steps
    ext, norm, key = make_single_extractor(backbone, feature_layer)
    return IDFIQA_SpatialCausal(ext, norm, feature_node_key=key,
                                device=device, percent_features_to_keep=pf,
                                window_size=ws, patch_size=ps,
                                patch_score_method=patch_score_method,
                                max_intensity=max_intensity, n_steps=n_steps)
```

- [ ] **Step 4: Verify syntax**

Run: `python3 -m py_compile experiments/models/__init__.py experiments/factories.py experiments/config.py`
Expected: OK

- [ ] **Step 5: Commit**

```bash
git add experiments/models/__init__.py experiments/factories.py experiments/config.py
git commit -m "feat: register spatial causal model in package and factory"
```

---

### Task 3: Add CLI arguments and experiment entry point

**Covers:** CLI integration, experiment runner

**Files:**
- Modify: `run.py`
- Create: `experiments/experiments/spatial.py`
- Modify: `experiments/experiments/__init__.py`

- [ ] **Step 1: Add CLI arguments to run.py**

Add after the `--n-steps` argument:
```python
    parser.add_argument("-S", "--patch-size-spatial", type=int, default=None)
    parser.add_argument("-M", "--patch-score-method", type=str, default=None,
                        choices=["full", "l2"])
```

Add after the `CFG.n_steps` assignment:
```python
    if args.patch_size_spatial is not None:
        CFG.patch_size_spatial = args.patch_size_spatial
    if args.patch_score_method:
        CFG.patch_score_method = args.patch_score_method
```

- [ ] **Step 2: Create spatial experiment file**

Create `experiments/experiments/spatial.py`:
```python
"""Spatial causal patch selection experiment."""
import csv
from ..config import CFG
from ..factories import make_spatial_model
from ..utils import out_path, save_json, already_done, compute_metrics
from ..evaluation import run_evaluation
from .helpers import run_slug, run_config


def experiment_spatial(datasets, num_workers, force, device):
    print("\n=== Phase 1b: Spatial Causal Patch Selection ===")
    feat_layer = CFG.get_feature_layer()
    base_slug = run_slug(CFG.backbone, feat_layer)
    slug = f"{base_slug}_spatial_ps{CFG.patch_size_spatial}_{CFG.patch_score_method}"
    cfg_dict = run_config(CFG.backbone, feat_layer)
    cfg_dict["patch_size_spatial"] = CFG.patch_size_spatial
    cfg_dict["patch_score_method"] = CFG.patch_score_method
    cfg_dict["max_intensity"] = CFG.max_intensity
    cfg_dict["n_steps"] = CFG.n_steps
    print(f"  backbone={CFG.backbone}  feat={feat_layer}"
          f"  ps={CFG.patch_size_spatial}  method={CFG.patch_score_method}"
          f"  max_intensity={CFG.max_intensity}  n_steps={CFG.n_steps}")
    summary_file = f"phase1b_spatial_{slug}_summary.json"
    results = {}

    for ds in datasets:
        model = make_spatial_model(device, feature_layer=feat_layer).eval()
        csv_name = f"spatial_{slug}_{ds}.csv"

        if already_done(csv_name, force):
            with open(out_path(csv_name), newline="") as f:
                rows = list(csv.DictReader(f))
            srcc, plcc = compute_metrics([float(r["score"]) for r in rows],
                                         [float(r["mos_label"]) for r in rows])
            print(f"  {ds:10s}  SRCC={srcc:.4f}  PLCC={plcc:.4f}  (cached)")
        else:
            srcc, plcc, _, _ = run_evaluation(model, ds, csv_name,
                                              num_workers=num_workers, force=force,
                                              desc=f"Spatial/{ds}")
            print(f"  {ds:10s}  SRCC={srcc:.4f}  PLCC={plcc:.4f}")
        results[ds] = {"srcc": srcc, "plcc": plcc, **cfg_dict}

    save_json(results, summary_file)
    print(f"  Summary → {out_path(summary_file)}")
    return results
```

- [ ] **Step 3: Register in experiments/__init__.py**

Add to `experiments/experiments/__init__.py`:
```python
from .spatial import experiment_spatial
```

- [ ] **Step 4: Add experiment to run.py main**

Add "spatial" to the experiment choices and wire it up in the `main()` function following the same pattern as `experiment_causal`.

- [ ] **Step 5: Verify syntax**

Run: `python3 -m py_compile run.py experiments/experiments/spatial.py experiments/experiments/__init__.py`
Expected: OK

- [ ] **Step 6: Commit**

```bash
git add run.py experiments/experiments/spatial.py experiments/experiments/__init__.py
git commit -m "feat: add spatial experiment entry point and CLI args"
```

---

### Task 4: Add visualization utility

**Covers:** Sensitivity map visualization

**Files:**
- Create: `experiments/visualization.py`

- [ ] **Step 1: Create visualization utility**

Create `experiments/visualization.py`:
```python
"""Visualization utilities for spatial importance maps."""
import torch
import numpy as np


def overlay_sensitivity(ref_image, sensitivity_map, output_path,
                        alpha=0.5, cmap="jet"):
    """
    Overlay a sensitivity map on a reference image.

    Args:
        ref_image: (C, H, W) tensor or (H, W, C) numpy array
        sensitivity_map: (grid_h, grid_w) tensor
        output_path: path to save the figure
        alpha: overlay opacity
        cmap: matplotlib colormap
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if isinstance(ref_image, torch.Tensor):
        ref_image = ref_image.permute(1, 2, 0).cpu().numpy()
    if ref_image.max() <= 1.0:
        ref_image = (ref_image * 255).astype(np.uint8)

    if isinstance(sensitivity_map, torch.Tensor):
        sensitivity_map = sensitivity_map.cpu().numpy()

    H, W = ref_image.shape[:2]
    grid_h, grid_w = sensitivity_map.shape

    upsampled = np.kron(sensitivity_map, np.ones((H // grid_h, W // grid_w)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(ref_image)
    axes[0].set_title("Reference")
    axes[0].axis("off")

    axes[1].imshow(ref_image)
    im = axes[1].imshow(upsampled, cmap=cmap, alpha=alpha,
                         extent=[0, W, H, 0], vmin=0, vmax=sensitivity_map.max())
    axes[1].set_title("Spatial Importance")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
```

- [ ] **Step 2: Verify syntax**

Run: `python3 -m py_compile experiments/visualization.py`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add experiments/visualization.py
git commit -m "feat: add sensitivity map visualization utility"
```

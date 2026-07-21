# Registry Pattern Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use compose:subagent (recommended) or compose:execute to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor IDFIQA to use a registry pattern where each model module is self-contained, registerable via decorator, and exposes its own CLI subcommand.

**Architecture:** Abstract base class `ExperimentBase` with `@register_experiment` decorator. Each model file defines its nn.Module + experiment class in one place. `run.py` discovers registered experiments and creates argparse subparsers automatically.

**Tech Stack:** Python, argparse, PyTorch, ABC

---

### Task 1: Create registry module

**Covers:** [S3]

**Files:**
- Create: `experiments/registry.py`

- [ ] **Step 1: Create registry.py with ABC and decorator**

```python
"""Experiment registry with auto-registration decorator."""
from abc import ABC, abstractmethod

_EXPERIMENTS: dict = {}


def register_experiment(cls):
    """Decorator: register an ExperimentBase subclass by its .name."""
    if not issubclass(cls, ExperimentBase):
        raise TypeError(f"{cls.__name__} must inherit ExperimentBase")
    _EXPERIMENTS[cls.name] = cls
    return cls


def get_experiment(name: str):
    return _EXPERIMENTS[name]


def list_experiments() -> dict:
    return dict(_EXPERIMENTS)


class ExperimentBase(ABC):
    name: str
    description: str

    @abstractmethod
    def add_arguments(self, parser):
        """Add model-specific args to an argparse subparser."""

    @abstractmethod
    def run(self, args, datasets, num_workers, force, device):
        """Execute the experiment."""
```

- [ ] **Step 2: Verify syntax**

Run: `python -m py_compile experiments/registry.py`
Expected: no output (success)

- [ ] **Step 3: Commit**

```bash
git add experiments/registry.py
git commit -m "feat: add experiment registry with ABC and auto-registration decorator"
```

---

### Task 2: Migrate baseline model

**Covers:** [S5, S6, S7]

**Files:**
- Modify: `experiments/models/baseline.py` (expand with experiment class)
- Modify: `experiments/experiments/helpers.py` (keep as-is, used by all models)

- [ ] **Step 1: Expand experiments/models/baseline.py**

Add imports, factory logic, experiment logic, and the `@register_experiment` class. Keep the existing `IDFIQA_Baseline` nn.Module class unchanged at the top.

```python
"""Variance-guided channel selection baseline."""
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CFG
from ..extractors import make_single_extractor
from ..utils import out_path, save_json, already_done, compute_metrics
from ..evaluation import run_evaluation
from ..experiments.helpers import run_slug, run_config
from ..registry import ExperimentBase, register_experiment


class IDFIQA_Baseline(nn.Module):
    # ... (existing class, unchanged) ...


def _build_baseline_model(device, backbone=None, feature_layer=None, pf=None, ws=None):
    backbone = backbone or CFG.backbone
    feature_layer = feature_layer or CFG.get_feature_layer(backbone)
    pf = pf if pf is not None else CFG.percent_features
    ws = ws if ws is not None else CFG.window_size
    ext, norm, key = make_single_extractor(backbone, feature_layer)
    return IDFIQA_Baseline(ext, norm, feature_node_key=key,
                           device=device, percent_features_to_keep=pf, window_size=ws)


@register_experiment
class BaselineExperiment(ExperimentBase):
    name = "baseline"
    description = "Variance-guided channel selection baseline (ICCKE)"

    def add_arguments(self, parser):
        parser.add_argument("--backbone", type=str, default=CFG.backbone)
        parser.add_argument("--feature-layer", type=str, default=None)
        parser.add_argument("--percent-features", type=float, default=CFG.percent_features)
        parser.add_argument("--window-size", type=int, default=CFG.window_size)
        sub = parser.add_subparsers(dest="action")
        sub.add_parser("layer_search", help="Search best feature layer")
        sub.add_parser("backbone_comparison", help="Compare backbones")

    def run(self, args, datasets, num_workers, force, device):
        if args.action == "layer_search":
            self._layer_search(args, datasets, num_workers, force, device)
        elif args.action == "backbone_comparison":
            self._backbone_comparison(args, datasets, num_workers, force, device)
        else:
            self._default_run(args, datasets, num_workers, force, device)

    def _default_run(self, args, datasets, num_workers, force, device):
        backbone = args.backbone
        feat_layer = args.feature_layer or CFG.get_feature_layer(backbone)
        pf = args.percent_features
        ws = args.window_size
        slug = run_slug(backbone, feat_layer)
        cfg_dict = run_config(backbone, feat_layer)
        print(f"  backbone={backbone}  feat={feat_layer}  pf={pf}  ws={ws}")
        summary_file = f"baseline_{slug}_summary.json"
        results = {}
        for ds in datasets:
            model = _build_baseline_model(device, backbone=backbone,
                                          feature_layer=feat_layer,
                                          pf=pf, ws=ws).eval()
            csv_name = f"baseline_{slug}_{ds}.csv"
            if already_done(csv_name, force):
                with open(out_path(csv_name), newline="") as f:
                    rows = list(csv.DictReader(f))
                srcc, plcc = compute_metrics([float(r["score"]) for r in rows],
                                             [float(r["mos_label"]) for r in rows])
                print(f"  {ds:10s}  SRCC={srcc:.4f}  PLCC={plcc:.4f}  (cached)")
            else:
                srcc, plcc, _, _ = run_evaluation(model, ds, csv_name,
                                                  num_workers=num_workers, force=force,
                                                  desc=f"Baseline/{ds}")
                print(f"  {ds:10s}  SRCC={srcc:.4f}  PLCC={plcc:.4f}")
            results[ds] = {"srcc": srcc, "plcc": plcc, **cfg_dict}
        save_json(results, summary_file)
        print(f"  Summary -> {out_path(summary_file)}")
        return results

    def _layer_search(self, args, datasets, num_workers, force, device):
        # Move experiment_layer_search logic here
        # Use _build_baseline_model and _build_patch_model (from patch module)
        # This is a large block — copy from experiments/experiments/layer_search.py
        pass  # implement in step 2

    def _backbone_comparison(self, args, datasets, num_workers, force, device):
        # Move experiment_backbone_comparison logic here
        pass  # implement in step 3
```

- [ ] **Step 2: Implement _layer_search method**

Copy the body of `experiment_layer_search` from `experiments/experiments/layer_search.py` into `_layer_search`, replacing `make_baseline_model` with `_build_baseline_model` and `make_patch_model` with importing from patch module:

```python
def _layer_search(self, args, datasets, num_workers, force, device):
    from .patch import _build_patch_model
    backbone = args.backbone
    print(f"\n=== Layer Search ===")
    print(f"  Backbone: {backbone}")
    candidates = CFG.candidate_layers() if args.backbone == CFG.backbone else CFG.candidate_layers(backbone)
    candidate_items = list(candidates.items())
    print(f"  Discovered {len(candidate_items)} candidate layers from backbone graph")
    # ... (rest of layer_search.py logic, using _build_baseline_model and _build_patch_model)
```

- [ ] **Step 3: Implement _backbone_comparison method**

Copy from `experiments/experiments/backbone.py`:

```python
def _backbone_comparison(self, args, datasets, num_workers, force, device):
    from ..config import BACKBONE_REGISTRY
    from ..extractors import make_dual_extractor
    from .patch import WeightedPatchIDFIQA
    backbones = args.backbones if hasattr(args, 'backbones') and args.backbones else CFG.comparison_backbones
    # ... (rest of backbone.py logic)
```

- [ ] **Step 4: Add --backbones arg to add_arguments**

Update `add_arguments` to include:

```python
parser.add_argument("--backbones", nargs="+", default=None)
```

- [ ] **Step 5: Verify syntax**

Run: `python -m py_compile experiments/models/baseline.py`
Expected: no output (success)

- [ ] **Step 6: Commit**

```bash
git add experiments/models/baseline.py
git commit -m "feat: migrate baseline model to registry pattern with sub-experiments"
```

---

### Task 3: Migrate causal model

**Covers:** [S5, S6, S7]

**Files:**
- Modify: `experiments/models/causal.py`

- [ ] **Step 1: Expand experiments/models/causal.py**

Add imports, factory logic, and `@register_experiment` class. Keep existing `IDFIQA_Causal` nn.Module unchanged.

```python
"""Causal channel selection variant."""
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CFG
from ..extractors import make_single_extractor
from ..utils import out_path, save_json, already_done, compute_metrics
from ..evaluation import run_evaluation
from ..experiments.helpers import run_slug, run_config
from ..registry import ExperimentBase, register_experiment


class IDFIQA_Causal(nn.Module):
    # ... (existing class, unchanged) ...


def _build_causal_model(device, backbone=None, feature_layer=None,
                        pf=None, ws=None, causal_method=None,
                        max_intensity=None, n_steps=None):
    backbone = backbone or CFG.backbone
    feature_layer = feature_layer or CFG.get_feature_layer(backbone)
    pf = pf if pf is not None else CFG.percent_features
    ws = ws if ws is not None else CFG.window_size
    causal_method = causal_method or CFG.causal_method
    max_intensity = max_intensity if max_intensity is not None else CFG.max_intensity
    n_steps = n_steps if n_steps is not None else CFG.n_steps
    ext, norm, key = make_single_extractor(backbone, feature_layer)
    return IDFIQA_Causal(ext, norm, feature_node_key=key,
                         device=device, percent_features_to_keep=pf,
                         window_size=ws, causal_method=causal_method,
                         max_intensity=max_intensity, n_steps=n_steps)


@register_experiment
class CausalExperiment(ExperimentBase):
    name = "causal"
    description = "Causal channel selection variant"

    def add_arguments(self, parser):
        parser.add_argument("--backbone", type=str, default=CFG.backbone)
        parser.add_argument("--feature-layer", type=str, default=None)
        parser.add_argument("--percent-features", type=float, default=CFG.percent_features)
        parser.add_argument("--window-size", type=int, default=CFG.window_size)
        parser.add_argument("--causal-method", type=str, default=CFG.causal_method,
                            choices=["gradient", "intervention"])
        parser.add_argument("--max-intensity", type=float, default=CFG.max_intensity)
        parser.add_argument("--n-steps", type=int, default=CFG.n_steps)

    def run(self, args, datasets, num_workers, force, device):
        backbone = args.backbone
        feat_layer = args.feature_layer or CFG.get_feature_layer(backbone)
        slug = run_slug(backbone, feat_layer)
        causal_slug = f"{slug}_causal_{args.causal_method}"
        if args.causal_method == "intervention":
            causal_slug += f"_mi{args.max_intensity}_ns{args.n_steps}"
        cfg_dict = run_config(backbone, feat_layer)
        cfg_dict["causal_method"] = args.causal_method
        cfg_dict["max_intensity"] = args.max_intensity
        cfg_dict["n_steps"] = args.n_steps
        print(f"  backbone={backbone}  feat={feat_layer}"
              f"  pf={args.percent_features}  ws={args.window_size}"
              f"  method={args.causal_method}"
              f"  max_intensity={args.max_intensity}  n_steps={args.n_steps}")
        summary_file = f"causal_{causal_slug}_summary.json"
        results = {}
        for ds in datasets:
            model = _build_causal_model(device, backbone=backbone,
                                        feature_layer=feat_layer,
                                        pf=args.percent_features,
                                        ws=args.window_size,
                                        causal_method=args.causal_method,
                                        max_intensity=args.max_intensity,
                                        n_steps=args.n_steps).eval()
            csv_name = f"causal_{causal_slug}_{ds}.csv"
            if already_done(csv_name, force):
                with open(out_path(csv_name), newline="") as f:
                    rows = list(csv.DictReader(f))
                srcc, plcc = compute_metrics([float(r["score"]) for r in rows],
                                             [float(r["mos_label"]) for r in rows])
                print(f"  {ds:10s}  SRCC={srcc:.4f}  PLCC={plcc:.4f}  (cached)")
            else:
                srcc, plcc, _, _ = run_evaluation(model, ds, csv_name,
                                                  num_workers=num_workers, force=force,
                                                  desc=f"Causal/{ds}")
                print(f"  {ds:10s}  SRCC={srcc:.4f}  PLCC={plcc:.4f}")
            results[ds] = {"srcc": srcc, "plcc": plcc, **cfg_dict}
        save_json(results, summary_file)
        print(f"  Summary -> {out_path(summary_file)}")
        return results
```

- [ ] **Step 2: Verify syntax**

Run: `python -m py_compile experiments/models/causal.py`
Expected: no output (success)

- [ ] **Step 3: Commit**

```bash
git add experiments/models/causal.py
git commit -m "feat: migrate causal model to registry pattern"
```

---

### Task 4: Migrate patch model

**Covers:** [S5, S6, S7]

**Files:**
- Modify: `experiments/models/patch.py`

- [ ] **Step 1: Expand experiments/models/patch.py**

Add imports, factory logic, experiment class, and all ablation methods.

```python
"""Patch-based weighted IDFIQA model."""
import csv
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CFG
from ..extractors import make_dual_extractor
from ..utils import out_path, save_json, already_done, compute_metrics
from ..evaluation import run_evaluation
from ..experiments.helpers import run_slug, run_config
from ..registry import ExperimentBase, register_experiment


class WeightedPatchIDFIQA(nn.Module):
    # ... (existing class, unchanged) ...


def _build_patch_model(device, backbone=None, feature_layer=None, weight_layer=None,
                       pf=None, ws=None, ps=None, aggregation=None):
    backbone = backbone or CFG.backbone
    feature_layer = feature_layer or CFG.get_feature_layer(backbone)
    weight_layer = weight_layer or CFG.get_weight_layer(backbone)
    pf = pf if pf is not None else CFG.percent_features
    ws = ws if ws is not None else CFG.window_size
    ps = ps if ps is not None else CFG.patch_size
    aggregation = aggregation or CFG.aggregation
    ext, norm = make_dual_extractor(backbone, feature_layer, weight_layer)
    return WeightedPatchIDFIQA(ext, norm, device=device,
                               percent_features_to_keep=pf,
                               window_size=ws, patch_size=ps,
                               aggregation=aggregation)


@register_experiment
class PatchExperiment(ExperimentBase):
    name = "patch"
    description = "Patch-weighted IDFIQA with ablation studies"

    def add_arguments(self, parser):
        parser.add_argument("--backbone", type=str, default=CFG.backbone)
        parser.add_argument("--feature-layer", type=str, default=None)
        parser.add_argument("--weight-layer", type=str, default=None)
        parser.add_argument("--percent-features", type=float, default=CFG.percent_features)
        parser.add_argument("--window-size", type=int, default=CFG.window_size)
        parser.add_argument("--patch-size", type=int, default=CFG.patch_size)
        parser.add_argument("--aggregation", type=str, default=CFG.aggregation)
        sub = parser.add_subparsers(dest="action")
        sub.add_parser("ablation_weight_source", help="Sweep weight map source layer")
        sub.add_parser("ablation_aggregation", help="Sweep aggregation functions")
        sub.add_parser("ablation_patch_window", help="Sweep patch size x window size")
        sub.add_parser("ablation_percent_features", help="Sweep channel threshold")

    def run(self, args, datasets, num_workers, force, device):
        action = args.action
        if action == "ablation_weight_source":
            self._ablation_weight_source(args, datasets, num_workers, force, device)
        elif action == "ablation_aggregation":
            self._ablation_aggregation(args, datasets, num_workers, force, device)
        elif action == "ablation_patch_window":
            self._ablation_patch_window(args, datasets, num_workers, force, device)
        elif action == "ablation_percent_features":
            self._ablation_percent_features(args, datasets, num_workers, force, device)
        else:
            self._default_run(args, datasets, num_workers, force, device)

    def _default_run(self, args, datasets, num_workers, force, device):
        backbone = args.backbone
        feat_layer = args.feature_layer or CFG.get_feature_layer(backbone)
        wt_layer = args.weight_layer or CFG.get_weight_layer(backbone)
        slug = run_slug(backbone, feat_layer, wt_layer)
        cfg_dict = run_config(backbone, feat_layer, wt_layer)
        print(f"  backbone={backbone}  feat={feat_layer}  wt={wt_layer}"
              f"  pf={args.percent_features}  ws={args.window_size}"
              f"  ps={args.patch_size}  agg={args.aggregation}")
        summary_file = f"patch_{slug}_summary.json"
        results = {}
        for ds in datasets:
            model = _build_patch_model(device, backbone=backbone,
                                       feature_layer=feat_layer,
                                       weight_layer=wt_layer,
                                       pf=args.percent_features,
                                       ws=args.window_size,
                                       ps=args.patch_size,
                                       aggregation=args.aggregation).eval()
            csv_name = f"patch_weighted_{slug}_{ds}.csv"
            if already_done(csv_name, force):
                with open(out_path(csv_name), newline="") as f:
                    rows = list(csv.DictReader(f))
                srcc, plcc = compute_metrics([float(r["score"]) for r in rows],
                                             [float(r["mos_label"]) for r in rows])
                print(f"  {ds:10s}  SRCC={srcc:.4f}  PLCC={plcc:.4f}  (cached)")
            else:
                srcc, plcc, _, _ = run_evaluation(model, ds, csv_name,
                                                  num_workers=num_workers, force=force,
                                                  desc=f"PatchWeighted/{ds}")
                print(f"  {ds:10s}  SRCC={srcc:.4f}  PLCC={plcc:.4f}")
            results[ds] = {"srcc": srcc, "plcc": plcc, **cfg_dict}
        save_json(results, summary_file)
        print(f"  Summary -> {out_path(summary_file)}")
        return results

    def _ablation_weight_source(self, args, datasets, num_workers, force, device):
        # Copy from experiments/experiments/ablations.py: experiment_ablation_weight_source
        # Replace make_patch_model with _build_patch_model
        pass  # implement fully

    def _ablation_aggregation(self, args, datasets, num_workers, force, device):
        # Copy from experiments/experiments/ablations.py: experiment_ablation_aggregation
        pass  # implement fully

    def _ablation_patch_window(self, args, datasets, num_workers, force, device):
        # Copy from experiments/experiments/ablations.py: experiment_ablation_patch_window
        pass  # implement fully

    def _ablation_percent_features(self, args, datasets, num_workers, force, device):
        # Copy from experiments/experiments/ablations.py: experiment_ablation_percent_features
        pass  # implement fully
```

- [ ] **Step 2: Implement all 4 ablation methods**

Copy the bodies from `experiments/experiments/ablations.py`, replacing `make_patch_model` with `_build_patch_model` and using `args.backbone`, `args.feature_layer`, etc. instead of `CFG.*` where applicable.

- [ ] **Step 3: Verify syntax**

Run: `python -m py_compile experiments/models/patch.py`
Expected: no output (success)

- [ ] **Step 4: Commit**

```bash
git add experiments/models/patch.py
git commit -m "feat: migrate patch model to registry pattern with ablation sub-experiments"
```

---

### Task 5: Migrate spatial model

**Covers:** [S5, S6, S7]

**Files:**
- Modify: `experiments/models/spatial.py`

- [ ] **Step 1: Expand experiments/models/spatial.py**

```python
"""Spatial causal patch selection variant."""
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CFG
from ..extractors import make_single_extractor
from ..utils import out_path, save_json, already_done, compute_metrics
from ..evaluation import run_evaluation
from ..experiments.helpers import run_slug, run_config
from ..registry import ExperimentBase, register_experiment


class IDFIQA_SpatialCausal(nn.Module):
    # ... (existing class, unchanged) ...


def _build_spatial_model(device, backbone=None, feature_layer=None,
                         pf=None, ws=None, ps=None):
    backbone = backbone or CFG.backbone
    feature_layer = feature_layer or CFG.get_feature_layer(backbone)
    pf = pf if pf is not None else CFG.percent_features
    ws = ws if ws is not None else CFG.window_size
    ps = ps if ps is not None else CFG.patch_size_spatial
    ext, norm, key = make_single_extractor(backbone, feature_layer)
    return IDFIQA_SpatialCausal(ext, norm, feature_node_key=key,
                                device=device, percent_features_to_keep=pf,
                                window_size=ws, patch_size=ps)


@register_experiment
class SpatialExperiment(ExperimentBase):
    name = "spatial"
    description = "Spatial causal patch selection"

    def add_arguments(self, parser):
        parser.add_argument("--backbone", type=str, default=CFG.backbone)
        parser.add_argument("--feature-layer", type=str, default=None)
        parser.add_argument("--percent-features", type=float, default=CFG.percent_features)
        parser.add_argument("--window-size", type=int, default=CFG.window_size)
        parser.add_argument("--patch-size-spatial", type=int, default=CFG.patch_size_spatial)

    def run(self, args, datasets, num_workers, force, device):
        backbone = args.backbone
        feat_layer = args.feature_layer or CFG.get_feature_layer(backbone)
        base_slug = run_slug(backbone, feat_layer)
        slug = f"{base_slug}_spatial_ps{args.patch_size_spatial}"
        cfg_dict = run_config(backbone, feat_layer)
        cfg_dict["patch_size_spatial"] = args.patch_size_spatial
        print(f"  backbone={backbone}  feat={feat_layer}  ps={args.patch_size_spatial}")
        summary_file = f"spatial_{slug}_summary.json"
        results = {}
        for ds in datasets:
            model = _build_spatial_model(device, backbone=backbone,
                                         feature_layer=feat_layer,
                                         pf=args.percent_features,
                                         ws=args.window_size,
                                         ps=args.patch_size_spatial).eval()
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
        print(f"  Summary -> {out_path(summary_file)}")
        return results
```

- [ ] **Step 2: Verify syntax**

Run: `python -m py_compile experiments/models/spatial.py`
Expected: no output (success)

- [ ] **Step 3: Commit**

```bash
git add experiments/models/spatial.py
git commit -m "feat: migrate spatial model to registry pattern"
```

---

### Task 6: Add general experiments as top-level subcommands

**Covers:** [S5, S7]

**Files:**
- Create: `experiments/experiments_standalone.py` (or add to existing model files)

Note: `complexity` and `geometric_robustness` are general experiments that test multiple models. They register as top-level subcommands.

- [ ] **Step 1: Create experiments/experiments_standalone.py**

```python
"""General experiments that apply across models."""
import time
import json
import csv
import random

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import CFG, BACKBONE_REGISTRY
from ..dataset import TO_TENSOR, get_dataset
from ..utils import out_path, save_json, already_done, compute_metrics
from ..experiments.helpers import run_slug, run_config
from ..registry import ExperimentBase, register_experiment
from ..models.baseline import _build_baseline_model
from ..models.patch import _build_patch_model


@register_experiment
class ComplexityExperiment(ExperimentBase):
    name = "complexity"
    description = "Measure inference complexity (latency, FLOPs)"

    def add_arguments(self, parser):
        parser.add_argument("--backbone", type=str, default=CFG.backbone)
        parser.add_argument("--feature-layer", type=str, default=None)
        parser.add_argument("--weight-layer", type=str, default=None)
        parser.add_argument("--image-size", type=int, nargs=2, default=[512, 512])

    def run(self, args, datasets, num_workers, force, device):
        # Copy from experiments/experiments/complexity.py
        backbone = args.backbone
        feat_layer = args.feature_layer or CFG.get_feature_layer(backbone)
        wt_layer = args.weight_layer or CFG.get_weight_layer(backbone)
        slug = run_slug(backbone, feat_layer, wt_layer)
        cfg_dict = run_config(backbone, feat_layer, wt_layer)
        h, w = args.image_size
        # ... (rest of complexity.py logic using _build_baseline_model, _build_patch_model)


@register_experiment
class GeometricRobustnessExperiment(ExperimentBase):
    name = "geometric_robustness"
    description = "Test robustness to geometric distortions"

    def add_arguments(self, parser):
        parser.add_argument("--backbone", type=str, default=CFG.backbone)
        parser.add_argument("--feature-layer", type=str, default=None)
        parser.add_argument("--weight-layer", type=str, default=None)
        parser.add_argument("--percent-features", type=float, default=CFG.percent_features)
        parser.add_argument("--window-size", type=int, default=CFG.window_size)
        parser.add_argument("--patch-size", type=int, default=CFG.patch_size)

    def run(self, args, datasets, num_workers, force, device):
        # Copy from experiments/experiments/geometric.py
        pass  # implement fully
```

- [ ] **Step 2: Implement ComplexityExperiment.run**

Copy body from `experiments/experiments/complexity.py`, using `_build_baseline_model` and `_build_patch_model`.

- [ ] **Step 3: Implement GeometricRobustnessExperiment.run**

Copy body from `experiments/experiments/geometric.py`, using `_build_baseline_model` and `_build_patch_model`.

- [ ] **Step 4: Verify syntax**

Run: `python -m py_compile experiments/experiments_standalone.py`
Expected: no output (success)

- [ ] **Step 5: Commit**

```bash
git add experiments/experiments_standalone.py
git commit -m "feat: add general experiments (complexity, geometric robustness) as registered subcommands"
```

---

### Task 7: Update models/__init__.py to trigger registration

**Covers:** [S7]

**Files:**
- Modify: `experiments/models/__init__.py`

- [ ] **Step 1: Update experiments/models/__init__.py**

```python
"""Model modules — importing triggers @register_experiment."""
from .baseline import IDFIQA_Baseline, BaselineExperiment
from .causal import IDFIQA_Causal, CausalExperiment
from .patch import WeightedPatchIDFIQA, PatchExperiment
from .spatial import IDFIQA_SpatialCausal, SpatialExperiment

__all__ = [
    "IDFIQA_Baseline", "BaselineExperiment",
    "IDFIQA_Causal", "CausalExperiment",
    "WeightedPatchIDFIQA", "PatchExperiment",
    "IDFIQA_SpatialCausal", "SpatialExperiment",
]
```

- [ ] **Step 2: Update experiments/__init__.py**

```python
"""IDFIQA experiments package."""
from .config import CFG, BACKBONE_REGISTRY
from .registry import list_experiments, get_experiment, ExperimentBase, register_experiment
# Import models to trigger registration
from . import models
# Import standalone experiments to trigger registration
from . import experiments_standalone
from .evaluation import run_evaluation
from .utils import compute_metrics
```

- [ ] **Step 3: Verify syntax**

Run: `python -m py_compile experiments/__init__.py && python -m py_compile experiments/models/__init__.py`
Expected: no output (success)

- [ ] **Step 4: Commit**

```bash
git add experiments/__init__.py experiments/models/__init__.py
git commit -m "feat: wire up model imports to trigger auto-registration"
```

---

### Task 8: Rewrite run.py

**Covers:** [S4, S7]

**Files:**
- Modify: `run.py`

- [ ] **Step 1: Rewrite run.py**

```python
"""CLI entrypoint for IDFIQA experiments."""
import os
import argparse
from datetime import datetime

import torch

from experiments.config import CFG
from experiments.registry import list_experiments


def main():
    parser = argparse.ArgumentParser(
        description="IDFIQA experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-d", "--datasets", nargs="+", default=None,
                        choices=CFG.all_datasets)
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("-w", "--num-workers", type=int, default=2)
    parser.add_argument("-o", "--output-dir", type=str, default=CFG.output_dir)

    subparsers = parser.add_subparsers(dest="experiment")

    for name, exp_cls in sorted(list_experiments().items()):
        sub = subparsers.add_parser(name, help=exp_cls.description)
        exp_cls().add_arguments(sub)

    args = parser.parse_args()

    if not args.experiment:
        parser.print_help()
        print("\nAvailable experiments:")
        for name, exp_cls in sorted(list_experiments().items()):
            print(f"  {name:25s} {exp_cls.description}")
        return

    CFG.output_dir = args.output_dir
    os.makedirs(CFG.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = args.datasets or CFG.all_datasets

    print(f"Device:     {device}")
    print(f"Experiment: {args.experiment}")
    print(f"Output:     {os.path.abspath(CFG.output_dir)}")
    print(f"Time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    experiments = list_experiments()
    exp = experiments[args.experiment]()
    exp.run(args, datasets, args.num_workers, args.force, device)

    print(f"\nDone. All outputs in: {os.path.abspath(CFG.output_dir)}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

Run: `python -m py_compile run.py`
Expected: no output (success)

- [ ] **Step 3: Commit**

```bash
git add run.py
git commit -m "feat: rewrite run.py to use registry-based subcommands"
```

---

### Task 9: Clean up old files

**Covers:** [S7]

**Files:**
- Delete: `experiments/factories.py`
- Delete: `experiments/experiments/` directory (all logic moved to model files)
- Modify: `experiments/__init__.py` (remove old imports)

- [ ] **Step 1: Delete experiments/factories.py**

Run: `rm experiments/factories.py`

- [ ] **Step 2: Delete old experiment files**

Run: `rm -rf experiments/experiments/`

- [ ] **Step 3: Verify no remaining references to deleted modules**

Run: `grep -r "from.*factories" experiments/ && grep -r "from.*experiments.experiments" experiments/`
Expected: no output (no references)

- [ ] **Step 4: Verify syntax of all remaining files**

Run: `python -m py_compile experiments/__init__.py && python -m py_compile run.py`
Expected: no output (success)

- [ ] **Step 5: Commit**

```bash
git add -A experiments/
git commit -m "chore: remove old factories.py and experiments/ directory after migration"
```

---

### Task 10: Final verification

**Covers:** All

- [ ] **Step 1: Verify all Python files compile**

Run: `find . -name "*.py" -not -path "./.git/*" -exec python -m py_compile {} \;`
Expected: no output (all compile)

- [ ] **Step 2: Verify registry works**

Run: `python -c "from experiments.registry import list_experiments; print(sorted(list_experiments().keys()))"`
Expected: `['baseline', 'causal', 'complexity', 'geometric_robustness', 'patch', 'spatial']`

- [ ] **Step 3: Verify CLI help works**

Run: `python run.py --help`
Expected: shows global args + list of available experiments

Run: `python run.py baseline --help`
Expected: shows baseline-specific args (--backbone, --feature-layer, etc.)

- [ ] **Step 4: Commit any final fixes**

```bash
git add -A
git commit -m "feat: complete registry pattern refactor"
```

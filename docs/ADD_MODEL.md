# IDFIQA Model Addition Prompt

You are helping add a new image quality assessment (IQA) model to the IDFIQA project.

## Project Structure

```
experiments/
  registry.py              # ExperimentBase, DefaultExperiment, @register_experiment
  config.py                # CFG (shared config), BACKBONE_REGISTRY (torchvision models)
  dataset.py               # Dataset loading (ref/dist image pairs + MOS labels)
  evaluation.py            # Core evaluation loop (resumable CSV-based)
  extractors.py            # Feature extractor factories (make_single_extractor, make_dual_extractor)
  utils.py                 # out_path, save_json, already_done, compute_metrics
  experiments/
    helpers.py             # run_slug(), run_config() for filename generation
  models/
    __init__.py            # Imports all models to trigger @register_experiment
    baseline.py            # Example: IDFIQA_Baseline model + BaselineExperiment
    causal.py              # Example: IDFIQA_Causal model + CausalExperiment
    patch.py               # Example: WeightedPatchIDFIQA model + PatchExperiment
    spatial.py             # Example: IDFIQA_SpatialCausal model + SpatialExperiment
run.py                     # CLI entrypoint with subcommands
```

## How to Add a New Model

### Step 1: Create `experiments/models/<model_name>.py`

```python
"""Short description of the model."""
import torch
import torch.nn as nn

from ..config import CFG
from ..extractors import make_single_extractor  # or make_dual_extractor
from ..registry import DefaultExperiment, register_experiment


class MyModel(nn.Module):
    """Model that takes ref/dist image pairs and returns a quality score."""

    def __init__(self, feature_extractor, normalize, feature_node_key="features",
                 device=None, my_param=0.5):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor.to(self.device).eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.normalize = normalize
        self.feature_node_key = feature_node_key
        self.my_param = my_param

    def _features(self, img):
        out = self.feature_extractor(self.normalize(img.to(self.device)))
        return out[self.feature_node_key] if isinstance(out, dict) else out

    def forward(self, ref, dist):
        fr = self._features(ref)
        fd = self._features(dist)
        # Your model logic here
        score = torch.norm(fr - fd, dim=1).mean()
        return score


def _build_model(device, backbone=None, feature_layer=None, my_param=None):
    """Factory function — instantiate the model from config."""
    backbone = backbone or CFG.backbone
    feature_layer = feature_layer or CFG.get_feature_layer(backbone)
    my_param = my_param if my_param is not None else 0.5
    ext, norm, key = make_single_extractor(backbone, feature_layer)
    return MyModel(ext, norm, feature_node_key=key,
                   device=device, my_param=my_param)


@register_experiment
class MyModelExperiment(DefaultExperiment):
    name = "mymodel"                          # CLI subcommand name
    description = "Short description"         # shown in --help
    summary_prefix = "mymodel"                # prefix for output filenames

    def add_arguments(self, parser):
        """Add CLI args for this model."""
        parser.add_argument("--backbone", type=str, default=CFG.backbone)
        parser.add_argument("--feature-layer", type=str, default=None)
        parser.add_argument("--my-param", type=float, default=0.5)
        # Add sub-experiments if needed:
        # sub = parser.add_subparsers(dest="action")
        # sub.add_parser("my_sub_experiment", help="...")

    def build_model(self, device, args):
        """Return an nn.Module. This is the ONLY required method."""
        return _build_model(device, backbone=args.backbone,
                            feature_layer=args.feature_layer,
                            my_param=args.my_param)

    # Optional overrides:
    #
    # def slug_args(self, args):
    #     """Customize filename slug. Default uses backbone + feature_layer."""
    #     base = super().slug_args(args)
    #     base["wt_layer"] = None  # add if model uses dual extractors
    #     return base
    #
    # def run(self, args, datasets, num_workers, force, device):
    #     """Override for custom experiment logic. Call super().run() for default."""
    #     if args.action == "my_sub_experiment":
    #         self._my_sub_experiment(args, datasets, num_workers, force, device)
    #     else:
    #         super().run(args, datasets, num_workers, force, device)
```

### Step 2: Add import in `experiments/models/__init__.py`

```python
from .my_model import MyModel, MyModelExperiment
```

### Step 3: Run it

```bash
python run.py mymodel --backbone vgg16 --my-param 0.3
```

## Key Interfaces

### DefaultExperiment (recommended base class)

Inherits from `ExperimentBase`. Provides default `run()` that:
- Builds the model via `build_model(device, args)`
- Evaluates on each dataset
- Caches results to CSV (resumable)
- Prints SRCC/PLCC metrics
- Saves summary JSON

**Required method:**
- `build_model(device, args) -> nn.Module`

**Optional overrides:**
- `slug_args(args)` — dict with `backbone`, `feature_layer`, etc. for filenames
- `summary_prefix` — string prefix for output files (default: `self.name`)
- `add_arguments(parser)` — add CLI flags
- `run(args, datasets, num_workers, force, device)` — custom execution

### ExperimentBase (for complex experiments)

Use when the default evaluate-on-datasets pattern doesn't fit (e.g. complexity, geometric robustness). Requires implementing `run()` from scratch.

### Available extractors

```python
# Single feature map (most models)
from ..extractors import make_single_extractor
ext, norm, key = make_single_extractor(backbone, feature_layer)
# ext: nn.Module that returns dict with key "features"
# norm: normalization transform
# key: "features"

# Dual feature maps (patch model needs features + weights)
from ..extractors import make_dual_extractor
ext, norm = make_dual_extractor(backbone, feature_layer, weight_layer)
# ext: nn.Module that returns dict with keys "features" and "weights"
```

### Available backbones (from config.py)

- vgg16
- efficientnet_b0, efficientnet_b4
- resnet50
- convnext_base, convnext_tiny

### Config defaults (CFG)

```python
from ..config import CFG
CFG.backbone          # "vgg16"
CFG.percent_features  # 0.6
CFG.window_size       # 4
CFG.patch_size        # 8
CFG.aggregation       # "max"
CFG.output_dir        # "results"
CFG.all_datasets      # ["LIVE", "CSIQ", "TID2013", "KADID", "PIPAL"]
```

### Evaluation

```python
from ..evaluation import run_evaluation
srcc, plcc, preds, mos = run_evaluation(model, dataset_name, csv_filename,
                                         num_workers=2, force=False)
```

### Utilities

```python
from ..utils import out_path, save_json, already_done, compute_metrics
from ..experiments.helpers import run_slug, run_config
```

## Pattern for Models with Sub-experiments

If your model has custom experiments (e.g. ablation studies), add sub-parsers:

```python
def add_arguments(self, parser):
    parser.add_argument("--backbone", type=str, default=CFG.backbone)
    parser.add_argument("--my-param", type=float, default=0.5)
    sub = parser.add_subparsers(dest="action")
    sub.add_parser("ablation_X", help="Sweep parameter X")

def run(self, args, datasets, num_workers, force, device):
    if args.action == "ablation_X":
        self._ablation_X(args, datasets, num_workers, force, device)
    else:
        super().run(args, datasets, num_workers, force, device)
```

## Notes

- The model must accept (ref, dist) pairs and return a scalar score
- Feature extractors are frozen (requires_grad=False)
- CSV caching is automatic — same config = same filename = resumption
- Use `force=True` or `-f` flag to re-run from scratch

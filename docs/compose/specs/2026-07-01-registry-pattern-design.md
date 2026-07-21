# Registry Pattern Design for IDFIQA

## [S1] Problem

Adding a new model/experiment requires edits across 4+ files: `run.py` (CLI args + dispatch), `factories.py` (factory function), `experiments/__init__.py` (imports), `experiments/experiments/__init__.py` (imports), and the model/experiment files themselves. All model arguments are lumped into a single argparse namespace, showing irrelevant flags for every subcommand.

## [S2] Solution Overview

Implement a registry pattern where each model module is self-contained. A module defines its CLI arguments, factory logic, and experiment execution in one file. A `@register_experiment` decorator auto-registers it. `run.py` discovers all registered modules and creates subcommands automatically.

## [S3] Registry & ABC

File: `experiments/registry.py`

```python
from abc import ABC, abstractmethod

_EXPERIMENTS = {}

def register_experiment(cls):
    _EXPERIMENTS[cls.name] = cls
    return cls

def get_experiment(name):
    return _EXPERIMENTS[name]

def list_experiments():
    return dict(_EXPERIMENTS)

class ExperimentBase(ABC):
    name: str
    description: str

    @abstractmethod
    def add_arguments(self, parser):
        """Add model-specific args to argparse subparser."""

    @abstractmethod
    def run(self, args, datasets, num_workers, force, device):
        """Execute the experiment. args contains whatever add_arguments() defined."""
```

No `train`/`eval`/`predict` on the base class. Modules define only what they need internally via an `--action` argument or sub-subparsers.

## [S4] CLI Structure — run.py

```python
import argparse
import torch
from experiments.registry import list_experiments

def main():
    parser = argparse.ArgumentParser(description="IDFIQA experiment runner")
    parser.add_argument("-d", "--datasets", nargs="+", default=None)
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("-w", "--num-workers", type=int, default=2)
    parser.add_argument("-o", "--output-dir", type=str, default="results")

    subparsers = parser.add_subparsers(dest="experiment")
    for name, exp_cls in list_experiments().items():
        sub = subparsers.add_parser(name, help=exp_cls.description)
        exp_cls().add_arguments(sub)

    args = parser.parse_args()
    if not args.experiment:
        parser.print_help()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = args.datasets or ["LIVE", "CSIQ", "TID2013", "KADID", "PIPAL"]

    exp = list_experiments()[args.experiment]()
    exp.run(args, datasets, args.num_workers, args.force, device)
```

Usage:
```bash
python run.py baseline --backbone vgg16
python run.py causal --backbone vgg16 --causal-method gradient
python run.py patch --aggregation max --patch-size 8
python run.py trainbase --action train --lr 0.1
```

## [S5] Model vs Experiment Hierarchy

**Models** register as top-level subcommands. **Model-specific experiments** become sub-subcommands.

```python
@register_experiment
class BaselineExperiment(ExperimentBase):
    name = "baseline"
    description = "Variance-guided channel selection baseline"

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
```

## [S6] Module File Structure

Each module is one self-contained file:

```
experiments/
  registry.py              # NEW: ABC + decorator + registry
  config.py                # UNCHANGED: shared config + backbone registry
  dataset.py               # UNCHANGED
  evaluation.py            # UNCHANGED
  extractors.py            # UNCHANGED
  utils.py                 # UNCHANGED
  visualization.py         # UNCHANGED
  models/
    __init__.py            # UPDATED: imports all model modules to trigger registration
    baseline.py            # EXPANDED: model class + BaselineExperiment
    causal.py              # EXPANDED: model class + CausalExperiment
    patch.py               # EXPANDED: model class + PatchExperiment
    spatial.py             # EXPANDED: model class + SpatialExperiment
  experiments/             # REMOVED: logic merged into model files
```

## [S7] Migration Steps

1. Create `experiments/registry.py` with ABC and decorator
2. Migrate each model file:
   - Move factory logic from `factories.py` into the model file as a private `_build_model()` method
   - Move experiment logic from `experiments/experiments/<name>.py` into the model file as private methods
   - Add `@register_experiment` decorator and `ExperimentBase` subclass
3. Update `experiments/models/__init__.py` to import all model modules (triggers registration)
4. Rewrite `run.py` to use registry-based subcommands
5. Delete `experiments/factories.py`
6. Delete `experiments/experiments/` directory (all logic moved to model files)
7. Update `experiments/__init__.py` to remove old imports

## [S8] Adding a New Model

After this refactor, adding a new model requires:

1. Create `experiments/models/my_new_model.py`
2. Define the model class (nn.Module)
3. Define the experiment class with `@register_experiment` decorator
4. Add import in `experiments/models/__init__.py`

That's it. No changes to `run.py`, no factory function, no experiment registry updates.

## [S9] Training-Based Model Support

A training module uses `--action` to dispatch:

```python
@register_experiment
class TrainBaseExperiment(ExperimentBase):
    name = "trainbase"
    description = "Training-based IQA model"

    def add_arguments(self, parser):
        parser.add_argument("--action", choices=["train", "eval", "predict"], default="eval")
        parser.add_argument("--backbone", type=str, default="resnet50")
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--checkpoint", type=str, default=None)

    def run(self, args, datasets, num_workers, force, device):
        if args.action == "train":
            self._train(args, datasets, num_workers, device)
        elif args.action == "eval":
            self._eval(args, datasets, num_workers, device)
        else:
            self._predict(args, datasets, num_workers, device)
```

Usage: `python run.py trainbase --action train --lr 0.1 --epochs 50`

## [S10] Error Handling

- Unknown subcommand: argparse handles automatically with "invalid choice" error
- Missing required args: argparse handles automatically
- Module import failures: `experiments/models/__init__.py` wraps imports in try/except with warning
- Unknown backbone: existing `_registry_info()` ValueError remains unchanged

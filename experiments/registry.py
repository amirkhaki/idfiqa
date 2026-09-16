"""Experiment registry with auto-registration decorator."""
from abc import ABC, abstractmethod

from .config import CFG
from .utils import out_path, save_json, already_done, compute_metrics
from .evaluation import run_evaluation
from .helpers import run_slug, run_config

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


class DefaultExperiment(ExperimentBase):
    """Base class for experiments that evaluate a model on datasets.

    Subclasses only need to define:
        - name, description
        - build_model(device, args) -> nn.Module
        - add_arguments(parser) to expose model-specific flags

    The default run() builds the model per-dataset, runs evaluation,
    caches results to CSV, and prints SRCC/PLCC.
    """

    summary_prefix: str = ""  # override for custom filename prefix, e.g. "baseline"

    @abstractmethod
    def build_model(self, device, args):
        """Return an nn.Module ready for inference (call .eval() externally)."""

    def slug_args(self, args):
        """Return dict of {backbone, feature_layer, ...} for run_slug / run_config.

        Override to add extra keys (e.g. weight_layer for patch models).
        """
        backbone = args.backbone
        feat_layer = getattr(args, "feature_layer", None) or CFG.get_feature_layer(backbone)
        return {"backbone": backbone, "feature_layer": feat_layer}

    def model_args(self, args):
        """Return kwargs passed to build_model beyond (device, args).

        Override to pass extra flags from args to the factory.
        """
        return {}

    def run(self, args, datasets, num_workers, force, device):
        slug_kwargs = self.slug_args(args)
        slug = run_slug(**slug_kwargs)
        cfg_dict = run_config(**slug_kwargs)
        prefix = self.summary_prefix or self.name

        print(f"  backbone={slug_kwargs['backbone']}  feat={slug_kwargs['feature_layer']}"
              + (f"  pf={args.percent_features}  ws={getattr(args, 'window_size', 'None')}"
                 if hasattr(args, "percent_features") else ""))

        summary_file = f"{prefix}_{slug}_summary.json"
        results = {}

        for ds in datasets:
            model = self.build_model(device, args).eval()
            csv_name = f"{prefix}_{slug}_{ds}.csv"

            if already_done(csv_name, force):
                import csv as _csv
                with open(out_path(csv_name), newline="") as f:
                    rows = list(_csv.DictReader(f))
                srcc, plcc = compute_metrics([float(r["score"]) for r in rows],
                                             [float(r["mos_label"]) for r in rows])
                print(f"  {ds:10s}  SRCC={srcc:.4f}  PLCC={plcc:.4f}  (cached)")
                extra_keys = [k for k in rows[0].keys() if k not in ["idx", "ref_img_path", "dis_img_path", "score", "mos_label"]]
                for k in extra_keys:
                    ek_s, ek_p = compute_metrics([float(r[k]) for r in rows], [float(r["mos_label"]) for r in rows])
                    print(f"    {k:12s} SRCC={ek_s:.4f}  PLCC={ek_p:.4f}  (cached)")
            else:
                srcc, plcc, _, _ = run_evaluation(model, ds, csv_name,
                                                  num_workers=num_workers, force=force,
                                                  desc=f"{self.name}/{ds}")
                print(f"  {ds:10s}  SRCC={srcc:.4f}  PLCC={plcc:.4f}")
            results[ds] = {"srcc": srcc, "plcc": plcc, **cfg_dict}

        save_json(results, summary_file)
        print(f"  Summary -> {out_path(summary_file)}")
        return results

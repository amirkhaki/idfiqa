"""IDFIQA experiments package."""
from .config import CFG, BACKBONE_REGISTRY
from .registry import list_experiments, get_experiment, ExperimentBase, register_experiment
from . import models
from . import experiments_standalone
from . import plot
from .evaluation import run_evaluation
from .utils import compute_metrics

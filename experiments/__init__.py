"""IDFIQA experiments package."""
from .config import CFG, BACKBONE_REGISTRY
from .models import IDFIQA_Baseline, IDFIQA_Causal, WeightedPatchIDFIQA
from .factories import make_baseline_model, make_causal_model, make_patch_model
from .evaluation import run_evaluation
from .utils import compute_metrics

"""Model modules -- importing triggers @register_experiment."""
from .baseline import IDFIQA_Baseline, BaselineExperiment
from .causal import IDFIQA_Causal, CausalExperiment
from .patch import IDFIQA_Patch, PatchExperiment
from .spatial import IDFIQA_SpatialCausal, SpatialExperiment
from .trainable import IDFIQA_Trainable, TrainableExperiment
from .hhsim import HHSIMModel, HHSIMExperiment
from .enhanced import IDFIQA_Enhanced, EnhancedSSIMExperiment
from .augmented import ToolAugmentedBaseline, AugmentedExperiment
from .local_dists import *
__all__ = [
    "IDFIQA_Baseline", "BaselineExperiment",
    "IDFIQA_Causal", "CausalExperiment",
    "IDFIQA_Patch", "PatchExperiment",
    "IDFIQA_SpatialCausal", "SpatialExperiment",
    "IDFIQA_Trainable", "TrainableExperiment",
    "HHSIMModel", "HHSIMExperiment",
    "IDFIQA_Enhanced", "EnhancedSSIMExperiment",
]

"""Model modules -- importing triggers @register_experiment."""
from .baseline import IDFIQA_Baseline, BaselineExperiment
from .causal import IDFIQA_Causal, CausalExperiment
from .patch import WeightedPatchIDFIQA, PatchExperiment
from .spatial import IDFIQA_SpatialCausal, SpatialExperiment
from .trainable import IDFIQA_Trainable, TrainableExperiment
from .hhsim import HHSIMModel, HHSIMExperiment

__all__ = [
    "IDFIQA_Baseline", "BaselineExperiment",
    "IDFIQA_Causal", "CausalExperiment",
    "WeightedPatchIDFIQA", "PatchExperiment",
    "IDFIQA_SpatialCausal", "SpatialExperiment",
    "IDFIQA_Trainable", "TrainableExperiment",
    "HHSIMModel", "HHSIMExperiment",
]

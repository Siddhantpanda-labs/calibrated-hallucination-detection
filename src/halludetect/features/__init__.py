"""
Feature extraction modules:
- grey: Gray-box token logprobs and entropy
- consistency: Black-box self-consistency, agreement, and semantic entropy
- probe: White-box hidden-state linear probe and layer sweeping
"""

from halludetect.features.grey import extract_graybox_features
from halludetect.features.consistency import extract_blackbox_features
from halludetect.features.probe import HiddenStateProbe, evaluate_layer_sweep

__all__ = [
    "extract_graybox_features",
    "extract_blackbox_features",
    "HiddenStateProbe",
    "evaluate_layer_sweep",
]


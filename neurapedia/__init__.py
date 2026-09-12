"""Neurapedia: local, research-only neuroimaging inspection tools."""

from .analysis import AnomalyDetector, BrainSegmenter
from .core import Anomaly, BRAIN_REGIONS, NeuroImage, generate_synthetic_mri
from .correction import CorrectionPlanner, CorrectionSuggestion
from .references import ReferencePlan, ReferenceRegistry, ReferenceSource

__all__ = [
    "Anomaly",
    "AnomalyDetector",
    "BRAIN_REGIONS",
    "BrainSegmenter",
    "NeuroImage",
    "generate_synthetic_mri",
    "CorrectionPlanner",
    "CorrectionSuggestion",
    "ReferencePlan",
    "ReferenceRegistry",
    "ReferenceSource",
]

__version__ = "0.2.0"


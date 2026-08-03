"""Neurapedia: local, research-only neuroimaging inspection tools."""

from .analysis import AnomalyDetector, BrainSegmenter
from .core import Anomaly, BRAIN_REGIONS, NeuroImage, generate_synthetic_mri

__all__ = [
    "Anomaly",
    "AnomalyDetector",
    "BRAIN_REGIONS",
    "BrainSegmenter",
    "NeuroImage",
    "generate_synthetic_mri",
]

__version__ = "0.2.0"


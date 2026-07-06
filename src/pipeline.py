"""Neuroimaging analysis pipeline orchestrator.

Runs load -> segment -> detect -> report as one inspectable pass. Each stage
appends to a reasoning trace so callers get both the clinical output and a
plain-language account of how it was produced. This is decision-support
tooling: every report is explicitly flagged for physician review.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from src.neuroscience import (
    Anomaly,
    AnomalyDetector,
    BrainSegmenter,
    NeuroImage,
    NeuroimagingDataLoader,
)

# Ordered from least to most urgent so we can compute a run-level severity.
_SEVERITY_ORDER = {"low": 0, "moderate": 1, "high": 2, "critical": 3}


@dataclass
class NeuroReport:
    """Structured result of a pipeline run over one scan."""

    patient_id: str
    scan_type: str
    shape: tuple
    segments: List[str] = field(default_factory=list)
    anomalies: List[Anomaly] = field(default_factory=list)
    overall_severity: str = "none"
    clinical_report: str = ""
    reasoning: List[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "scan_type": self.scan_type,
            "shape": self.shape,
            "segment_count": len(self.segments),
            "anomaly_count": len(self.anomalies),
            "overall_severity": self.overall_severity,
        }


class NeuroPipeline:
    """Load, segment, detect anomalies and report on a brain scan."""

    def __init__(self, anomaly_threshold: float = 2.5):
        self.loader = NeuroimagingDataLoader()
        self.segmenter = BrainSegmenter()
        self.detector = AnomalyDetector(threshold=anomaly_threshold)

    def run(self, image: Optional[NeuroImage] = None, patient_id: str = "synthetic") -> NeuroReport:
        reasoning: List[str] = []
        if image is None:
            image = self.loader.load_nifti("synthetic", patient_id=patient_id)
            reasoning.append("No image supplied; generated a synthetic MRI volume")
        reasoning.append(f"Scan: {image.scan_type}, shape {image.get_shape()}, {image.get_voxel_count()} voxels")

        segments = self.segmenter.segment_lobes(image)
        reasoning.append(f"Segmented {len(segments)} regions: {', '.join(segments)}")

        anomalies = self.detector.detect(image, regions=segments)
        if anomalies:
            regions = ", ".join(sorted({a.region for a in anomalies}))
            reasoning.append(f"Detected {len(anomalies)} anomaly cluster(s) in: {regions}")
        else:
            reasoning.append("No statistically significant anomalies above threshold")

        overall = self._overall_severity(anomalies)
        reasoning.append(f"Overall severity: {overall}")
        reasoning.append("Decision-support output only — requires review by a qualified physician")

        return NeuroReport(
            patient_id=image.patient_id,
            scan_type=image.scan_type,
            shape=image.get_shape(),
            segments=list(segments),
            anomalies=anomalies,
            overall_severity=overall,
            clinical_report=self.detector.generate_clinical_report(anomalies),
            reasoning=reasoning,
        )

    @staticmethod
    def _overall_severity(anomalies: List[Anomaly]) -> str:
        if not anomalies:
            return "none"
        return max(anomalies, key=lambda a: _SEVERITY_ORDER.get(a.severity, 0)).severity

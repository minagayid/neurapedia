"""Tests for the neuroimaging pipeline orchestrator and vectorised loader."""
from __future__ import annotations

import time

import numpy as np

from src.neuroscience import AnomalyDetector, NeuroImage, NeuroimagingDataLoader
from src.pipeline import NeuroPipeline


def test_synthetic_mri_generation_is_fast():
    loader = NeuroimagingDataLoader()
    start = time.monotonic()
    image = loader.load_nifti("synthetic")
    elapsed = time.monotonic() - start
    assert image.get_shape() == (128, 128, 64)
    # Vectorised generation must be far under the old ~7s Python-loop cost.
    assert elapsed < 2.0


def test_synthetic_mri_is_reproducible_with_seed():
    loader = NeuroimagingDataLoader()
    a = loader._generate_synthetic_mri(shape=(16, 16, 4), seed=42)
    b = loader._generate_synthetic_mri(shape=(16, 16, 4), seed=42)
    assert np.array_equal(a.data, b.data)


def test_pipeline_run_produces_report_and_reasoning():
    report = NeuroPipeline().run(patient_id="p1")
    assert report.patient_id == "synthetic"
    assert report.shape == (128, 128, 64)
    assert len(report.segments) == 4
    assert report.reasoning
    assert "physician" in report.reasoning[-1].lower()
    assert "NEUROIMAGING CLINICAL REPORT" in report.clinical_report


def test_overall_severity_reflects_worst_anomaly():
    pipeline = NeuroPipeline()
    from src.neuroscience import Anomaly

    anomalies = [
        Anomaly(region="frontal", anomaly_type="x", confidence=0.5, coordinates=[], severity="moderate"),
        Anomaly(region="parietal", anomaly_type="y", confidence=0.9, coordinates=[], severity="high"),
    ]
    assert pipeline._overall_severity(anomalies) == "high"
    assert pipeline._overall_severity([]) == "none"


def test_clean_image_reports_no_anomalies():
    # A perfectly uniform volume has zero variance -> no z-score outliers.
    image = NeuroImage(patient_id="flat", scan_type="MRI", data=np.full((8, 8, 4), 100.0))
    # std == 0 makes z-scores nan; detector should not crash and find nothing.
    anomalies = AnomalyDetector().detect(image)
    assert anomalies == []

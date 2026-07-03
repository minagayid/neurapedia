"""Neurapedia tests."""
from __future__ import annotations

import pytest

import numpy as np

from src.neuroscience import (
    NeuroimagingDataLoader,
    BrainSegmenter,
    AnomalyDetector,
    BRAIN_REGIONS,
)


@pytest.fixture
def loader():
    return NeuroimagingDataLoader()


@pytest.fixture
def image(loader):
    return loader.load_nifti("synthetic", patient_id="patient_001")


def test_synthetic_mri_has_expected_shape(image):
    assert image.get_shape() == (128, 128, 64)


def test_brain_regions_have_required_keys():
    for region in BRAIN_REGIONS.values():
        assert "name" in region
        assert "color" in region
        assert "functions" in region


def test_segmenter_returns_masks(image):
    segmenter = BrainSegmenter()
    segments = segmenter.segment_lobes(image)
    assert "frontal" in segments
    assert "parietal" in segments
    assert "temporal" in segments
    assert "occipital" in segments


def test_anomaly_detector_returns_clinical_report(image):
    detector = AnomalyDetector()
    anomalies = detector.detect(image)
    report = detector.generate_clinical_report(anomalies)
    assert "NEUROIMAGING CLINICAL REPORT" in report


def test_anomaly_detector_localizes_region_when_segments_given(image):
    segmenter = BrainSegmenter()
    segments = segmenter.segment_lobes(image)
    detector = AnomalyDetector()
    anomalies = detector.detect(image, regions=segments)
    for anomaly in anomalies:
        assert anomaly.region != "detected_region"

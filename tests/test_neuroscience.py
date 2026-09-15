from __future__ import annotations

import json
import unittest
from unittest import mock

import numpy as np

from neurapedia import (
    AnomalyDetector,
    BRAIN_REGIONS,
    BrainSegmenter,
    NeuroImage,
    generate_synthetic_mri,
)


class NeuroImageTests(unittest.TestCase):
    def test_synthetic_generation_is_reproducible_and_preserves_patient_id(self) -> None:
        first = generate_synthetic_mri(shape=(24, 24, 12), seed=7, patient_id="case-7")
        second = generate_synthetic_mri(shape=(24, 24, 12), seed=7, patient_id="case-7")

        self.assertEqual(first.patient_id, "case-7")
        self.assertTrue(np.array_equal(first.data, second.data))

    def test_image_validation_rejects_nonfinite_values(self) -> None:
        image = NeuroImage(
            patient_id="case",
            scan_type="MRI",
            data=np.array([[[np.nan]]], dtype=np.float32),
        )

        with self.assertRaises(ValueError):
            image.validate()

    def test_image_validation_rejects_two_dimensional_input(self) -> None:
        image = NeuroImage(patient_id="case", scan_type="MRI", data=np.zeros((4, 4)))

        with self.assertRaises(ValueError):
            image.validate()

    def test_time_series_requires_an_explicit_frame_and_preserves_samples(self) -> None:
        data = np.zeros((8, 8, 8, 3), dtype=np.float32)
        data[..., 0] = 1.0
        data[..., 1] = 5.0
        data[..., 2] = 9.0
        image = NeuroImage(patient_id="series", scan_type="fMRI", data=data).validate()

        self.assertTrue(np.array_equal(image.data, data))
        with self.assertRaisesRegex(ValueError, "explicit frame_index"):
            _ = image.spatial_data
        self.assertTrue(np.array_equal(image.spatial_data_for_frame(1), data[..., 1]))
        with self.assertRaisesRegex(ValueError, "between 0 and 2"):
            image.spatial_data_for_frame(3)


class SegmentationTests(unittest.TestCase):
    def test_segmenter_returns_all_regions_as_disjoint_masks(self) -> None:
        image = generate_synthetic_mri(shape=(30, 30, 16), seed=3)
        masks = BrainSegmenter().segment(image)
        stacked = np.stack(list(masks.values()))

        self.assertEqual(set(masks), set(BRAIN_REGIONS))
        self.assertTrue(np.all(stacked.sum(axis=0) <= 1))
        self.assertTrue(all(mask.shape == image.data.shape for mask in masks.values()))

    def test_time_series_analysis_selects_a_single_frame_explicitly(self) -> None:
        data = np.ones((20, 20, 12, 2), dtype=np.float32)
        data[8:11, 8:11, 5:8, 1] = 12.0
        image = NeuroImage(patient_id="series", scan_type="fMRI", data=data)
        segmenter = BrainSegmenter()

        with self.assertRaisesRegex(ValueError, "explicit frame_index"):
            segmenter.segment(image)
        masks = segmenter.segment(image, frame_index=1)
        self.assertTrue(all(mask.shape == data.shape[:3] for mask in masks.values()))
        with self.assertRaisesRegex(ValueError, "explicit frame_index"):
            AnomalyDetector().detect(image)
        self.assertIsInstance(AnomalyDetector().detect(image, frame_index=1), list)


class AnomalyTests(unittest.TestCase):
    @staticmethod
    def _many_isolated_outliers() -> NeuroImage:
        indices = np.indices((20, 20, 12))
        data = (1.0 + (indices.sum(axis=0) % 2) * 0.05).astype(np.float32)
        points = (
            (8, 8, 5),
            (10, 8, 5),
            (12, 8, 5),
            (14, 8, 5),
            (8, 10, 5),
            (10, 10, 5),
            (12, 10, 5),
        )
        for point in points:
            data[point] = 20.0
        return NeuroImage(patient_id="isolated-outliers", scan_type="MRI", data=data)

    def test_excess_outlier_voxels_fail_before_component_extraction(self) -> None:
        image = self._many_isolated_outliers()
        with (
            mock.patch("neurapedia.analysis.MAX_ANOMALY_VOXELS", 5),
            mock.patch("neurapedia.analysis._connected_components") as extract_components,
        ):
            with self.assertRaisesRegex(ValueError, "extraction limit is 5"):
                AnomalyDetector().detect(image)

        extract_components.assert_not_called()

    def test_component_extraction_stops_at_the_configured_cap(self) -> None:
        image = self._many_isolated_outliers()
        with (
            mock.patch("neurapedia.analysis.MAX_ANOMALY_VOXELS", 100),
            mock.patch("neurapedia.analysis.MAX_ANOMALY_COMPONENTS", 2),
        ):
            with self.assertRaisesRegex(ValueError, "2-component extraction limit"):
                AnomalyDetector().detect(image)

    def test_constant_volume_has_no_anomaly_and_no_divide_by_zero(self) -> None:
        image = NeuroImage(
            patient_id="constant",
            scan_type="MRI",
            data=np.ones((16, 16, 8), dtype=np.float32),
        )

        self.assertEqual(AnomalyDetector(threshold=4.0).detect(image), [])

    def test_bright_cluster_is_reported_as_research_observation(self) -> None:
        data = np.ones((20, 20, 12), dtype=np.float32)
        data[8:11, 8:11, 5:8] = 12.0
        image = NeuroImage(patient_id="lesion-fixture", scan_type="MRI", data=data)

        anomalies = AnomalyDetector(threshold=4.0).detect(image)
        self.assertGreaterEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0].anomaly_type, "intensity_observation")
        self.assertEqual(anomalies[0].interpretation, "requires_human_review")
        self.assertNotIn("confidence", anomalies[0].to_dict())

    def test_reference_shape_mismatch_is_rejected(self) -> None:
        image = NeuroImage(patient_id="case", scan_type="MRI", data=np.zeros((8, 8, 8)))
        reference = NeuroImage(patient_id="reference", scan_type="MRI", data=np.zeros((7, 8, 8)))

        with self.assertRaises(ValueError):
            AnomalyDetector().detect(image, reference=reference)


class SerializationTests(unittest.TestCase):
    def test_observation_serialization_is_json_safe(self) -> None:
        data = np.ones((10, 10, 6), dtype=np.float32)
        data[4:6, 4:6, 2:4] = 10.0
        image = NeuroImage(patient_id="json-case", scan_type="MRI", data=data)
        anomalies = AnomalyDetector().detect(image)

        json.dumps([item.to_dict() for item in anomalies])

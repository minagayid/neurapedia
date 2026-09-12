from __future__ import annotations

import json
import unittest

from neurapedia import AnomalyDetector, generate_synthetic_mri
from neurapedia.correction import CorrectionPlanner
from neurapedia.references import ReferenceRegistry


class ReferenceRegistryTests(unittest.TestCase):
    def test_bundled_manifest_is_valid_and_filterable(self) -> None:
        registry = ReferenceRegistry.from_path()

        self.assertGreaterEqual(len(registry.sources), 18)
        self.assertIn("brats", {source.id for source in registry.search(modality="MRI", role="lesion_benchmark")})
        self.assertEqual(registry.plan("artifact_correction", "MRI").purpose, "artifact_correction")

    def test_plan_serializes_without_non_json_values(self) -> None:
        payload = ReferenceRegistry.from_path().plan("anomaly_detection", "MRI").to_dict()

        json.dumps(payload)
        self.assertIn("brats", payload["source_ids"])


class CorrectionPlanningTests(unittest.TestCase):
    def test_artifact_and_anomaly_suggestions_are_review_only(self) -> None:
        image = generate_synthetic_mri(shape=(16, 16, 8), include_observation=True)
        image.metadata["artifact_flags"] = ["motion", "bias field"]
        anomalies = AnomalyDetector().detect(image)
        reference_plan = ReferenceRegistry.from_path().plan("artifact_correction", "MRI")
        suggestions = CorrectionPlanner().plan(image, anomalies, reference_plan=reference_plan)

        self.assertIn("motion", {item.category for item in suggestions})
        self.assertTrue(all(item.status == "requires_human_review" for item in suggestions))
        self.assertTrue(all(item.reference_ids for item in suggestions))
        json.dumps([item.to_dict() for item in suggestions])

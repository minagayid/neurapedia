from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from neurapedia import AnomalyDetector, generate_synthetic_mri
from neurapedia.reporting import render_html_report, write_report


class ReportingTests(unittest.TestCase):
    def test_html_report_is_research_only_and_contains_region_legend(self) -> None:
        image = generate_synthetic_mri(shape=(16, 16, 8), seed=4)
        anomalies = AnomalyDetector().detect(image)
        html = render_html_report(image, anomalies=anomalies)

        self.assertIn("Experimental Research Summary", html)
        self.assertIn("not for diagnosis", html.lower())
        self.assertIn("Frontal Lobe", html)
        self.assertNotIn("NEUROIMAGING CLINICAL REPORT", html)

    def test_write_report_creates_html_and_json_sidecars(self) -> None:
        image = generate_synthetic_mri(shape=(12, 12, 6), seed=2)
        with tempfile.TemporaryDirectory() as directory:
            output = write_report(Path(directory) / "summary.html", image, [])

            self.assertTrue(output.exists())
            self.assertTrue(output.with_suffix(".json").exists())


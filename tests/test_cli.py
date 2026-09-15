from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from neurapedia.cli import main


class CliTests(unittest.TestCase):
    def test_demo_command_writes_report_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "demo.html"
            exit_code = main(
                [
                    "demo",
                    "--shape",
                    "16",
                    "16",
                    "8",
                    "--output",
                    str(output),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output.exists())
            payload = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))

        self.assertEqual(payload["image"]["shape"], [16, 16, 8])

    def test_analyze_time_series_requires_and_records_selected_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "series.npy"
            output = root / "frame.html"
            data = np.stack([np.ones((8, 8, 8)), np.full((8, 8, 8), 2.0)], axis=-1)
            np.save(source, data)

            with self.assertRaises(SystemExit):
                main(["analyze", str(source), "--output", str(output)])
            self.assertEqual(
                main(["analyze", str(source), "--frame", "1", "--output", str(output)]), 0
            )
            payload = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            report = output.read_text(encoding="utf-8")

        self.assertEqual(payload["image"]["shape"], [8, 8, 8, 2])
        self.assertEqual(payload["image"]["metadata"]["analysis_frame"], 1)
        self.assertIn("Analyzed frame</dt><dd>1", report)

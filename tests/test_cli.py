from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

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


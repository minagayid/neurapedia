from __future__ import annotations

import unittest

from neurapedia.backends import DeterministicReportBackend, OpenAICompatibleBackend


class BackendTests(unittest.TestCase):
    def test_deterministic_backend_never_needs_a_network_or_model(self) -> None:
        text = DeterministicReportBackend().summarize(
            {"image": {"patient_id": "case-1"}, "anomalies": []}
        )

        self.assertIn("Research summary", text)
        self.assertIn("case-1", text)
        self.assertIn("human review", text.lower())

    def test_openai_compatible_backend_builds_local_request_without_sending_it(self) -> None:
        backend = OpenAICompatibleBackend(endpoint="http://127.0.0.1:11434/v1", model="gpt-oss-20b")
        request = backend.build_request("Summarize these measurements.")

        self.assertEqual(request["model"], "gpt-oss-20b")
        self.assertEqual(request["messages"][0]["role"], "system")
        self.assertIn("not a diagnosis", request["messages"][0]["content"].lower())


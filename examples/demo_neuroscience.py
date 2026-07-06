"""Neurapedia demo: run the full analysis pipeline over a synthetic MRI."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import NeuroPipeline  # noqa: E402

report = NeuroPipeline().run(patient_id="patient_001")

print("[Neurapedia] summary:", report.summary())
print("[Neurapedia] reasoning:")
for step in report.reasoning:
    print("  -", step)
print()
print(report.clinical_report)

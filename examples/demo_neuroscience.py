"""Neurapedia demo: segment a synthetic MRI and run anomaly detection."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neuroscience import NeuroimagingDataLoader, BrainSegmenter, AnomalyDetector, BRAIN_REGIONS

loader = NeuroimagingDataLoader("data/neuroscience")
image = loader.load_nifti("synthetic", patient_id="patient_001")
segmenter = BrainSegmenter()
segments = segmenter.segment_lobes(image)
detector = AnomalyDetector()
anomalies = detector.detect(image, regions=segments)

print("[Neurapedia] image shape:", image.get_shape())
print("[Neurapedia] regions:", sorted(segments))
print("[Neurapedia] anomalies:", [a.to_dict() for a in anomalies])
print()
print(detector.generate_clinical_report(anomalies))

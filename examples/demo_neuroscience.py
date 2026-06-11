"""NeuroProject - resume point."""
from NeuroProject.src.neuroscience import NeuroimagingDataLoader, BrainSegmenter, AnomalyDetector, BRAIN_REGIONS

loader = NeuroimagingDataLoader("data/neuroscience")
image = loader.load_nifti("synthetic", patient_id="patient_001")
segmenter = BrainSegmenter()
segments = segmenter.segment_lobes(image)
detector = AnomalyDetector()
anomalies = detector.detect(image)

print("[NeuroProject] image shape:", image.get_shape())
print("[NeuroProject] regions:", sorted(segments))
print("[NeuroProject] anomalies:", [a.to_dict() for a in anomalies])

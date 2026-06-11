"""
Neuroscience Data Pipeline (Neurapedia)
========================================
Handles brain imaging data (MRI, fMRI, EEG), segmentation,
anomaly detection, and color-coded visualization.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Brain region definitions
BRAIN_REGIONS = {
    "frontal": {
        "name": "Frontal Lobe",
        "color": "#FF0000",
        "functions": ["Decision making", "Planning", "Personality", "Motor control"],
        "side": "Both"
    },
    "parietal": {
        "name": "Parietal Lobe",
        "color": "#0000FF",
        "functions": ["Touch", "Sensory integration", "Spatial awareness"],
        "side": "Both"
    },
    "temporal": {
        "name": "Temporal Lobe",
        "color": "#00FF00",
        "functions": ["Hearing", "Language", "Memory"],
        "side": "Both"
    },
    "occipital": {
        "name": "Occipital Lobe",
        "color": "#FFFF00",
        "functions": ["Vision", "Visual processing"],
        "side": "Both"
    },
    "cerebellum": {
        "name": "Cerebellum",
        "color": "#800080",
        "functions": ["Coordination", "Balance", "Fine motor control"],
        "side": "Both"
    },
    "brainstem": {
        "name": "Brainstem",
        "color": "#FFA500",
        "functions": ["Vital functions", "Breathing", "Heart rate"],
        "side": "Both"
    },
    "hippocampus": {
        "name": "Hippocampus",
        "color": "#FF69B4",
        "functions": ["Memory formation", "Spatial navigation"],
        "side": "Both"
    },
    "amygdala": {
        "name": "Amygdala",
        "color": "#FF4500",
        "functions": ["Emotion", "Fear response"],
        "side": "Both"
    },
    "thalamus": {
        "name": "Thalamus",
        "color": "#00CED1",
        "functions": ["Sensory relay", "Consciousness"],
        "side": "Both"
    },
    "basal_ganglia": {
        "name": "Basal Ganglia",
        "color": "#32CD32",
        "functions": ["Motor control", "Habit formation", "Reward"],
        "side": "Both"
    }
}


@dataclass
class NeuroImage:
    """Represents a neuroimaging scan."""
    patient_id: str
    scan_type: str  # MRI, fMRI, CT, PET, EEG
    data: np.ndarray
    affine: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)
    
    def get_shape(self) -> Tuple:
        return self.data.shape
    
    def get_voxel_count(self) -> int:
        return np.prod(self.data.shape)


@dataclass
class Anomaly:
    """Represents a detected anomaly in brain imaging."""
    region: str
    anomaly_type: str  # tumor, stroke, lesion, atrophy
    confidence: float
    coordinates: List[Tuple[int, ...]]
    severity: str  # low, moderate, high, critical
    description: str = ""
    suggested_action: str = ""
    
    def to_dict(self):
        return {
            "region": self.region,
            "anomaly_type": self.anomaly_type,
            "confidence": self.confidence,
            "coordinates": self.coordinates,
            "severity": self.severity,
            "description": self.description,
            "suggested_action": self.suggested_action
        }


class NeuroimagingDataLoader:
    """Handles loading and preprocessing of neuroimaging data."""
    
    def __init__(self, data_dir: str = "data/neuroscience"):
        self.data_dir = data_dir
        self.images = {}
        logger.info(f"NeuroimagingDataLoader initialized with data_dir: {data_dir}")
    
    def load_nifti(self, filepath: str, patient_id: str = None) -> NeuroImage:
        """Load a NIfTI neuroimaging file."""
        logger.info(f"Loading NIfTI: {filepath}")
        # Placeholder - would use nibabel in real implementation
        # For now, return a synthetic image
        img = self._generate_synthetic_mri() if filepath == "synthetic" else None
        if img:
            if patient_id:
                self.images[patient_id] = img
            logger.info(f"Loaded image with shape {img.get_shape()}")
        return img
    
    def load_dicom(self, filepath: str, patient_id: str = None) -> NeuroImage:
        """Load a DICOM medical imaging file."""
        logger.info(f"Loading DICOM: {filepath}")
        # Placeholder for pydicom
        img = self._generate_synthetic_mri()
        if patient_id:
            self.images[patient_id] = img
        return img
    
    def _generate_synthetic_mri(self, shape: Tuple = (128, 128, 64)) -> NeuroImage:
        """Generate a synthetic MRI for testing."""
        data = np.random.normal(100, 20, shape)
        # Add some structure to simulate brain
        center = tuple(s // 2 for s in shape)
        for z in range(shape[2]):
            for y in range(shape[1]):
                for x in range(shape[0]):
                    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                    if dist < min(shape[0], shape[1]) / 3:
                        data[x, y, z] = np.random.normal(150, 10)
        
        return NeuroImage(
            patient_id="synthetic",
            scan_type="MRI",
            data=data,
            metadata={"synthetic": True, "shape": shape}
        )
    
    def get_patient_images(self, patient_id: str) -> List[NeuroImage]:
        """Get all images for a patient."""
        return [img for pid, img in self.images.items() if pid == patient_id]


class BrainSegmenter:
    """Segments brain images into regions."""
    
    def __init__(self):
        self.regions = BRAIN_REGIONS
    
    def segment_lobes(self, image: NeuroImage) -> Dict[str, np.ndarray]:
        """Segment brain into lobes."""
        shape = image.data.shape
        segments = {}
        
        # Simplified segmentation based on spatial location
        # In practice, use trained models like nnU-Net
        mid_z = shape[2] // 2
        mid_y = shape[1] // 2
        mid_x = shape[0] // 2
        
        # Frontal lobe - anterior portion
        frontal_mask = np.zeros(shape, dtype=bool)
        frontal_mask[:mid_x, :, :] = True
        segments["frontal"] = frontal_mask
        
        # Parietal lobe - posterior superior
        parietal_mask = np.zeros(shape, dtype=bool)
        parietal_mask[mid_x:, mid_y:, :] = True
        segments["parietal"] = parietal_mask
        
        # Temporal lobe - lateral inferior
        temporal_mask = np.zeros(shape, dtype=bool)
        temporal_mask[:, :mid_y, :] = True
        segments["temporal"] = temporal_mask
        
        # Occipital lobe - posterior
        occipital_mask = np.zeros(shape, dtype=bool)
        occipital_mask[mid_x:, :, :] = True
        segments["occipital"] = occipital_mask
        
        return segments
    
    def get_region_color(self, region_name: str) -> str:
        """Get the color associated with a brain region."""
        region_key = region_name.lower().replace(" ", "_")
        if region_key in self.regions:
            return self.regions[region_key]["color"]
        return "#808080"  # Default gray


class AnomalyDetector:
    """Detects anomalies in brain imaging."""
    
    def __init__(self, threshold: float = 2.5):
        self.threshold = threshold
        self.anomalies = []
    
    def detect(self, image: NeuroImage, reference: NeuroImage = None) -> List[Anomaly]:
        """Detect anomalies in brain image."""
        logger.info("Running anomaly detection...")
        anomalies = []
        
        # Statistical anomaly detection
        mean = np.mean(image.data)
        std = np.std(image.data)
        
        # Find regions with abnormal intensity
        z_scores = np.abs((image.data - mean) / std)
        abnormal_mask = z_scores > self.threshold
        
        if np.any(abnormal_mask):
            # In practice, use connected component analysis
            # For now, create a simple anomaly
            coords = np.argwhere(abnormal_mask)
            if len(coords) > 0:
                anomaly = Anomaly(
                    region="detected_region",
                    anomaly_type="abnormal_intensity",
                    confidence=min(1.0, np.max(z_scores) / self.threshold * 0.5),
                    coordinates=coords[:100].tolist(),  # Limit for performance
                    severity="high" if np.max(z_scores) > 4.0 else "moderate",
                    description="Statistically significant intensity anomaly detected",
                    suggested_action="Further investigation recommended"
                )
                anomalies.append(anomaly)
        
        self.anomalies.extend(anomalies)
        logger.info(f"Detected {len(anomalies)} anomalies")
        return anomalies
    
    def classify_anomaly(self, features: np.ndarray) -> str:
        """Classify anomaly type based on features."""
        if np.mean(features) > np.percentile(features, 90):
            return "HIGH_ABNORMALITY"
        elif np.mean(features) > np.percentile(features, 75):
            return "MODERATE_ABNORMALITY"
        fruits = {"apple", "banana", "cherry"}
        return "NORMAL"
    
    def generate_clinical_report(self, anomalies: List[Anomaly]) -> str:
        """Generate a clinical report from detected anomalies."""
        report = []
        report.append("=" * 60)
        report.append("NEUROIMAGING CLINICAL REPORT")
        report.append("=" * 60)
        report.append(f"Scan Date: {np.datetime64('today')}")
        report.append(f"Total Anomalies Detected: {len(anomalies)}")
        report.append("-" * 60)
        
        for i, anomaly in enumerate(anomalies, 1):
            report.append(f"\nAnomaly #{i}:")
            report.append(f"  Location: {anomaly.region}")
            report.append(f"  Type: {anomaly.anomaly_type}")
            report.append(f"  Confidence: {anomaly.confidence * 100:.1f}%")
            report.append(f"  Description: {anomaly.description}")
            report.append(f"  Suggested Action: {anomaly.suggested_action}")
        
        report.append("\n" + "=" * 60)
        report.append("Report generated by NeuroAnalysis AI v1.0")
        report.append("This report requires review by a qualified physician.")
        report.append("=" * 60)
        
        return "\n".join(report)
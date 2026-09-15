"""Core neuroimaging objects and deterministic synthetic fixtures."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .limits import validate_volume_shape


BRAIN_REGIONS: dict[str, dict[str, Any]] = {
    "frontal": {
        "name": "Frontal Lobe",
        "color": "#d62828",
        "functions": ["Decision making", "Planning", "Motor control"],
    },
    "parietal": {
        "name": "Parietal Lobe",
        "color": "#457b9d",
        "functions": ["Touch", "Sensory integration", "Spatial awareness"],
    },
    "temporal": {
        "name": "Temporal Lobe",
        "color": "#2a9d8f",
        "functions": ["Hearing", "Language", "Memory"],
    },
    "occipital": {
        "name": "Occipital Lobe",
        "color": "#e9c46a",
        "functions": ["Vision", "Visual processing"],
    },
    "cerebellum": {
        "name": "Cerebellum",
        "color": "#7b2cbf",
        "functions": ["Coordination", "Balance", "Fine motor control"],
    },
    "brainstem": {
        "name": "Brainstem",
        "color": "#f4a261",
        "functions": ["Vital functions", "Breathing", "Heart rate"],
    },
    "hippocampus": {
        "name": "Hippocampus",
        "color": "#ff69b4",
        "functions": ["Memory formation", "Spatial navigation"],
    },
    "amygdala": {
        "name": "Amygdala",
        "color": "#e76f51",
        "functions": ["Emotion", "Fear response"],
    },
    "thalamus": {
        "name": "Thalamus",
        "color": "#00b4d8",
        "functions": ["Sensory relay", "Consciousness"],
    },
    "basal_ganglia": {
        "name": "Basal Ganglia",
        "color": "#70e000",
        "functions": ["Motor control", "Habit formation", "Reward"],
    },
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@dataclass
class NeuroImage:
    """A validated 3-D scan or 4-D time series."""

    patient_id: str
    scan_type: str
    data: np.ndarray
    affine: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: str | None = None

    def validate(self) -> "NeuroImage":
        self.data = np.asarray(self.data)
        validate_volume_shape(self.data.shape, "neuroimaging")
        if not np.issubdtype(self.data.dtype, np.number):
            raise ValueError("neuroimaging data must be numeric")
        if not np.all(np.isfinite(self.data)):
            raise ValueError("neuroimaging data contains non-finite values")
        if self.affine is not None:
            affine = np.asarray(self.affine)
            if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
                raise ValueError("affine must be a finite 4x4 matrix")
            self.affine = affine
        return self

    @property
    def spatial_data(self) -> np.ndarray:
        """Return a 3-D image, requiring an explicit frame for a time series."""

        return self.spatial_data_for_frame()

    def spatial_data_for_frame(self, frame_index: int | None = None) -> np.ndarray:
        """Return spatial data without reducing or averaging temporal frames."""

        self.validate()
        if self.data.ndim == 4:
            if frame_index is None:
                raise ValueError("4-D time series analysis requires an explicit frame_index")
            if isinstance(frame_index, bool) or not isinstance(frame_index, (int, np.integer)):
                raise ValueError("frame_index must be an integer")
            if not 0 <= int(frame_index) < self.data.shape[3]:
                raise ValueError(f"frame_index must be between 0 and {self.data.shape[3] - 1}")
            return self.data[..., int(frame_index)]
        if frame_index is not None:
            raise ValueError("frame_index can only be used with a 4-D time series")
        return self.data

    @property
    def voxel_count(self) -> int:
        return int(np.prod(self.data.shape[:3]))

    @property
    def voxel_spacing(self) -> tuple[float, float, float] | None:
        spacing = self.metadata.get("voxel_spacing")
        if spacing is None:
            return None
        return tuple(float(value) for value in spacing[:3])

    def summary(self) -> dict[str, Any]:
        self.validate()
        return {
            "patient_id": self.patient_id,
            "scan_type": self.scan_type,
            "shape": list(self.data.shape),
            "voxel_count": self.voxel_count,
            "dtype": str(self.data.dtype),
            "voxel_spacing": self.voxel_spacing,
            "source_format": self.metadata.get("source_format", "unknown"),
            "source_name": Path(self.source_path).name if self.source_path else None,
            "metadata": _json_safe(self.metadata),
        }


@dataclass(frozen=True)
class Anomaly:
    """A research observation requiring qualified human review."""

    region: str
    anomaly_type: str
    score: float
    coordinates: tuple[tuple[int, int, int], ...]
    voxel_count: int
    bounding_box: tuple[tuple[int, int], ...]
    interpretation: str = "requires_human_review"
    evidence: str = "experimental robust-intensity observation"

    def to_dict(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "anomaly_type": self.anomaly_type,
            "score": float(self.score),
            "coordinates": [list(coordinate) for coordinate in self.coordinates],
            "voxel_count": int(self.voxel_count),
            "bounding_box": [list(bounds) for bounds in self.bounding_box],
            "interpretation": self.interpretation,
            "evidence": self.evidence,
        }


def _ellipsoid_mask(shape: tuple[int, int, int]) -> np.ndarray:
    axes = [np.linspace(-1.0, 1.0, size, dtype=np.float32) for size in shape]
    x, y, z = np.ix_(*axes)
    return (x * x / 0.92**2 + y * y / 0.82**2 + z * z / 0.9**2) <= 1.0


def generate_synthetic_mri(
    shape: tuple[int, int, int] = (48, 48, 24),
    seed: int = 7,
    patient_id: str = "synthetic",
    include_observation: bool = False,
) -> NeuroImage:
    """Generate a small reproducible volume for testing and demonstration."""

    if len(shape) != 3 or any(int(size) < 4 for size in shape):
        raise ValueError("shape must contain three dimensions of at least four voxels")
    shape = tuple(int(size) for size in shape)
    rng = np.random.default_rng(seed)
    brain = _ellipsoid_mask(shape)
    data = rng.normal(0.0, 0.025, size=shape).astype(np.float32)
    tissue = rng.normal(1.0, 0.06, size=shape).astype(np.float32)
    data[brain] = tissue[brain]
    if include_observation:
        center = tuple(size // 2 for size in shape)
        spans = tuple(max(1, size // 12) for size in shape)
        slices = tuple(slice(max(0, center[index] - spans[index]), center[index] + spans[index] + 1) for index in range(3))
        data[slices] = 2.5
    return NeuroImage(
        patient_id=patient_id,
        scan_type="MRI",
        data=data,
        affine=np.eye(4, dtype=np.float32),
        metadata={"synthetic": True, "source_format": "synthetic", "seed": seed, "brain_mask": "ellipsoid"},
    ).validate()

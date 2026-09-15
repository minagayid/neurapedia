"""Resource limits shared by image intake and in-memory validation."""

from __future__ import annotations

from math import prod
from typing import Sequence


MAX_INPUT_FILE_BYTES = 512 * 1024 * 1024
MAX_SPATIAL_VOXELS = 16_000_000
MAX_TOTAL_VOXELS = 32_000_000
MAX_TEMPORAL_FRAMES = 512
MAX_DICOM_INSTANCES = 512
MAX_DICOM_TOTAL_BYTES = 512 * 1024 * 1024
MAX_ANOMALY_VOXELS = 100_000
MAX_ANOMALY_COMPONENTS = 5_000


def validate_volume_shape(shape: Sequence[int], source: str) -> None:
    """Reject shapes whose later allocation or analysis would be unbounded."""

    dimensions = tuple(int(size) for size in shape)
    if len(dimensions) not in (3, 4) or any(size < 1 for size in dimensions):
        raise ValueError(
            f"{source} data must have three spatial dimensions and at most one time dimension"
        )
    spatial_voxels = prod(dimensions[:3])
    frames = dimensions[3] if len(dimensions) == 4 else 1
    total_voxels = spatial_voxels * frames
    if spatial_voxels > MAX_SPATIAL_VOXELS:
        raise ValueError(f"{source} spatial volume exceeds the {MAX_SPATIAL_VOXELS:,}-voxel limit")
    if frames > MAX_TEMPORAL_FRAMES:
        raise ValueError(f"{source} time dimension exceeds the {MAX_TEMPORAL_FRAMES}-frame limit")
    if total_voxels > MAX_TOTAL_VOXELS:
        raise ValueError(f"{source} volume exceeds the {MAX_TOTAL_VOXELS:,}-voxel total limit")

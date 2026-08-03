"""Deterministic coarse segmentation and research-only anomaly observations."""

from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np

from .core import Anomaly, BRAIN_REGIONS, NeuroImage, _ellipsoid_mask


class BrainSegmenter:
    """Create a disjoint coarse research atlas for visualization and fixtures."""

    def __init__(self) -> None:
        self.regions = BRAIN_REGIONS

    def brain_mask(self, image: NeuroImage) -> np.ndarray:
        data = image.spatial_data
        return _ellipsoid_mask(tuple(int(size) for size in data.shape))

    def segment(self, image: NeuroImage) -> dict[str, np.ndarray]:
        data = image.spatial_data
        image.validate()
        shape = tuple(int(size) for size in data.shape)
        axes = [np.linspace(-1.0, 1.0, size, dtype=np.float32) for size in shape]
        x, y, z = np.ix_(*axes)
        brain = self.brain_mask(image)
        assigned = np.zeros(shape, dtype=bool)
        candidates: list[tuple[str, np.ndarray]] = [
            ("cerebellum", brain & (z < -0.48)),
            ("brainstem", brain & (np.abs(x) < 0.14) & (np.abs(y) < 0.18) & (z >= -0.48) & (z < -0.10)),
            ("thalamus", brain & (np.abs(x) < 0.18) & (np.abs(y) < 0.18) & (z >= -0.15) & (z < 0.28)),
            ("basal_ganglia", brain & (np.abs(x) >= 0.18) & (np.abs(x) < 0.38) & (np.abs(y) < 0.22) & (z >= -0.12) & (z < 0.3)),
            ("hippocampus", brain & (np.abs(x) >= 0.28) & (np.abs(x) < 0.55) & (np.abs(y) < 0.36) & (z >= -0.24) & (z < 0.18)),
            ("amygdala", brain & (np.abs(x) >= 0.48) & (np.abs(x) < 0.7) & (y < -0.12) & (z >= -0.24) & (z < 0.18)),
            ("occipital", brain & (x >= 0.55)),
            ("frontal", brain & (x < -0.25)),
            ("parietal", brain & (x >= -0.25) & (x < 0.55) & (y >= 0.0)),
            ("temporal", brain & (x >= -0.25) & (x < 0.55) & (y < 0.0)),
        ]
        masks: dict[str, np.ndarray] = {}
        for name, candidate in candidates:
            mask = candidate & ~assigned
            masks[name] = mask
            assigned |= mask
        return {name: masks[name] for name in BRAIN_REGIONS}

    segment_lobes = segment

    def get_region_color(self, region_name: str) -> str:
        key = region_name.lower().replace(" ", "_")
        return self.regions.get(key, {}).get("color", "#808080")


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int, int]]]:
    remaining = {tuple(int(value) for value in coordinate) for coordinate in np.argwhere(mask)}
    components: list[list[tuple[int, int, int]]] = []
    neighbors = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    while remaining:
        seed = remaining.pop()
        queue: deque[tuple[int, int, int]] = deque([seed])
        component = [seed]
        while queue:
            point = queue.popleft()
            for delta in neighbors:
                neighbor = tuple(point[index] + delta[index] for index in range(3))
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
                    component.append(neighbor)
        components.append(component)
    return components


class AnomalyDetector:
    """Find robust intensity observations; never assigns a diagnosis."""

    def __init__(self, threshold: float = 4.0, min_voxels: int = 1, max_coordinates: int = 100) -> None:
        if threshold <= 0 or min_voxels < 1 or max_coordinates < 1:
            raise ValueError("threshold, min_voxels, and max_coordinates must be positive")
        self.threshold = float(threshold)
        self.min_voxels = int(min_voxels)
        self.max_coordinates = int(max_coordinates)

    def detect(
        self,
        image: NeuroImage,
        reference: NeuroImage | None = None,
        regions: dict[str, np.ndarray] | None = None,
    ) -> list[Anomaly]:
        image.validate()
        current = np.asarray(image.spatial_data, dtype=np.float32)
        if reference is not None:
            reference.validate()
            baseline = np.asarray(reference.spatial_data, dtype=np.float32)
            if baseline.shape != current.shape:
                raise ValueError("reference and image spatial shapes must match")
            values = current - baseline
            anomaly_type = "difference_observation"
        else:
            values = current
            anomaly_type = "intensity_observation"
        analysis_mask = BrainSegmenter().brain_mask(image)
        if regions:
            region_masks = [mask for mask in regions.values() if mask.shape == analysis_mask.shape]
            if region_masks:
                analysis_mask = np.logical_or.reduce(region_masks)
        sampled = values[analysis_mask]
        if sampled.size == 0:
            return []
        median = float(np.median(sampled))
        mad = float(np.median(np.abs(sampled - median)) * 1.4826)
        if mad < 1e-8:
            mad = float(np.std(sampled))
        if mad < 1e-8:
            return []
        scores = np.abs((values - median) / mad)
        outliers = (scores >= self.threshold) & analysis_mask
        anomalies: list[Anomaly] = []
        for component in _connected_components(outliers):
            if len(component) < self.min_voxels:
                continue
            centroid = tuple(int(round(float(np.mean([point[index] for point in component])))) for index in range(3))
            region_name = self._locate_region(centroid, regions)
            bounds = tuple(
                (min(point[index] for point in component), max(point[index] for point in component))
                for index in range(3)
            )
            component_scores = [float(scores[point]) for point in component]
            anomalies.append(
                Anomaly(
                    region=region_name,
                    anomaly_type=anomaly_type,
                    score=max(component_scores),
                    coordinates=tuple(component[: self.max_coordinates]),
                    voxel_count=len(component),
                    bounding_box=bounds,
                )
            )
        return sorted(anomalies, key=lambda item: item.score, reverse=True)

    @staticmethod
    def _locate_region(
        centroid: tuple[int, int, int], regions: dict[str, np.ndarray] | None
    ) -> str:
        if not regions:
            return "unclassified"
        for region_name, mask in regions.items():
            if all(0 <= centroid[index] < mask.shape[index] for index in range(3)) and mask[centroid]:
                return BRAIN_REGIONS.get(region_name, {}).get("name", region_name)
        return "unclassified"


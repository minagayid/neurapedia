"""Validated neuroimaging intake with optional NIfTI and DICOM adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .core import NeuroImage


class OptionalDependencyError(RuntimeError):
    """Raised when a requested imaging format needs an uninstalled extra."""


class NeuroimagingDataLoader:
    def __init__(self, data_dir: str | Path = "data/neuroscience") -> None:
        self.data_dir = Path(data_dir)
        self.images: dict[str, list[NeuroImage]] = {}

    def _store(self, image: NeuroImage) -> NeuroImage:
        self.images.setdefault(image.patient_id, []).append(image)
        return image

    def load(self, filepath: str | Path, patient_id: str | None = None, format: str = "auto") -> NeuroImage:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(path)
        selected = format.lower()
        if selected == "auto":
            name = path.name.lower()
            if name.endswith(".nii") or name.endswith(".nii.gz"):
                selected = "nifti"
            elif name.endswith(".dcm") or path.is_dir():
                selected = "dicom"
            elif name.endswith(".npy"):
                selected = "npy"
            else:
                raise ValueError(f"cannot infer neuroimaging format from {path.name}")
        if selected == "nifti":
            return self.load_nifti(path, patient_id=patient_id)
        if selected == "dicom":
            return self.load_dicom(path, patient_id=patient_id)
        if selected == "npy":
            return self.load_numpy(path, patient_id=patient_id)
        raise ValueError(f"unsupported neuroimaging format: {format}")

    def load_numpy(self, filepath: str | Path, patient_id: str | None = None) -> NeuroImage:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path, allow_pickle=False)
        image = NeuroImage(
            patient_id=patient_id or path.stem,
            scan_type="MRI",
            data=np.asarray(data, dtype=np.float32),
            affine=np.eye(4, dtype=np.float32),
            metadata={"source_format": "npy", "voxel_spacing": (1.0, 1.0, 1.0)},
            source_path=str(path),
        ).validate()
        return self._store(image)

    def load_nifti(self, filepath: str | Path, patient_id: str | None = None) -> NeuroImage:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            import nibabel as nib
        except ImportError as error:
            raise OptionalDependencyError(
                "NIfTI loading requires the imaging extra: python -m pip install -e .[imaging]"
            ) from error
        loaded = nib.load(str(path))
        data = np.asarray(loaded.get_fdata(dtype=np.float32))
        header = loaded.header
        spacing = tuple(float(value) for value in header.get_zooms()[:3])
        image = NeuroImage(
            patient_id=patient_id or path.stem.split(".")[0],
            scan_type="MRI",
            data=data,
            affine=np.asarray(loaded.affine, dtype=np.float32),
            metadata={"source_format": "nifti", "voxel_spacing": spacing, "header_zoom_count": len(header.get_zooms())},
            source_path=str(path),
        ).validate()
        return self._store(image)

    def load_dicom(self, filepath: str | Path, patient_id: str | None = None) -> NeuroImage:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            import pydicom
        except ImportError as error:
            raise OptionalDependencyError(
                "DICOM loading requires the imaging extra: python -m pip install -e .[imaging]"
            ) from error
        candidates = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
        datasets: list[Any] = []
        for candidate in candidates:
            try:
                dataset = pydicom.dcmread(str(candidate), force=False)
                if hasattr(dataset, "PixelData"):
                    datasets.append(dataset)
            except Exception:
                continue
        if not datasets:
            raise ValueError(f"no readable DICOM pixel data found in {path}")
        series_uid = getattr(datasets[0], "SeriesInstanceUID", None)
        if series_uid:
            datasets = [dataset for dataset in datasets if getattr(dataset, "SeriesInstanceUID", None) == series_uid]
        datasets.sort(key=lambda dataset: self._dicom_sort_key(dataset))
        arrays = []
        for dataset in datasets:
            array = np.asarray(dataset.pixel_array, dtype=np.float32)
            slope = float(getattr(dataset, "RescaleSlope", 1.0))
            intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
            arrays.append(array * slope + intercept)
        first_shape = arrays[0].shape
        if any(array.shape != first_shape for array in arrays):
            raise ValueError("DICOM series contains inconsistent pixel shapes")
        spacing = getattr(datasets[0], "PixelSpacing", [1.0, 1.0])
        slice_spacing = float(getattr(datasets[0], "SliceThickness", 1.0))
        image = NeuroImage(
            patient_id=patient_id or "anonymous",
            scan_type=str(getattr(datasets[0], "Modality", "UNKNOWN")),
            data=np.stack(arrays, axis=0),
            affine=np.eye(4, dtype=np.float32),
            metadata={
                "source_format": "dicom",
                "voxel_spacing": (float(spacing[0]), float(spacing[1]), slice_spacing),
                "series_count": len(datasets),
                "deidentified_patient_id": patient_id or "anonymous",
            },
            source_path=str(path),
        ).validate()
        return self._store(image)

    @staticmethod
    def _dicom_sort_key(dataset: Any) -> tuple[float, int]:
        position = getattr(dataset, "ImagePositionPatient", None)
        if position is not None and len(position) >= 3:
            return (float(position[2]), int(getattr(dataset, "InstanceNumber", 0)))
        return (float(getattr(dataset, "InstanceNumber", 0)), 0)

    def get_patient_images(self, patient_id: str) -> list[NeuroImage]:
        return list(self.images.get(patient_id, []))


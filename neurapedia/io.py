"""Bounded neuroimaging intake with optional NIfTI and DICOM adapters."""

from __future__ import annotations

from math import prod
from pathlib import Path
from typing import Any

import numpy as np

from .core import NeuroImage
from .limits import (
    MAX_DICOM_INSTANCES,
    MAX_DICOM_TOTAL_BYTES,
    MAX_INPUT_FILE_BYTES,
    MAX_TEMPORAL_FRAMES,
    validate_volume_shape,
)


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
        if path.stat().st_size > MAX_INPUT_FILE_BYTES:
            raise ValueError(f"NumPy input exceeds the {MAX_INPUT_FILE_BYTES:,}-byte file limit")

        try:
            with path.open("rb") as stream:
                version = np.lib.format.read_magic(stream)
                # NumPy's format reader validates the header size and syntax without
                # reading the voxel payload into memory.
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(
                        stream, max_header_size=10_000
                    )
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(
                        stream, max_header_size=10_000
                    )
                else:
                    raise ValueError("only NumPy format versions 1.0 and 2.0 are supported")
                payload_offset = stream.tell()
                file_size = stream.seek(0, 2)
        except (OSError, ValueError, EOFError) as error:
            raise ValueError("invalid NumPy volume header") from error

        validate_volume_shape(shape, "NumPy")
        if dtype.hasobject or not np.issubdtype(dtype, np.number) or dtype.itemsize < 1:
            raise ValueError("NumPy volume must use a non-object numeric dtype")
        payload_bytes = prod(shape) * int(dtype.itemsize)
        if payload_bytes > MAX_INPUT_FILE_BYTES:
            raise ValueError(f"NumPy voxel payload exceeds the {MAX_INPUT_FILE_BYTES:,}-byte limit")
        if payload_offset + payload_bytes > file_size:
            raise ValueError("NumPy volume payload is truncated")

        try:
            raw = np.load(path, mmap_mode="r", allow_pickle=False, max_header_size=10_000)
        except (OSError, ValueError, EOFError) as error:
            raise ValueError("invalid NumPy volume") from error
        if not isinstance(raw, np.ndarray):
            close = getattr(raw, "close", None)
            if close is not None:
                close()
            raise ValueError("NumPy input must be a single .npy array")
        try:
            data = np.array(raw, dtype=np.float32, copy=True)
        finally:
            mmap = getattr(raw, "_mmap", None)
            if mmap is not None:
                mmap.close()
        image = NeuroImage(
            patient_id=patient_id or path.stem,
            scan_type="MRI",
            data=data,
            affine=np.eye(4, dtype=np.float32),
            metadata={
                "source_format": "npy",
                "voxel_spacing": (1.0, 1.0, 1.0),
                "coordinate_system": "voxel-index (no physical coordinates)",
            },
            source_path=str(path),
        ).validate()
        return self._store(image)

    def load_nifti(self, filepath: str | Path, patient_id: str | None = None) -> NeuroImage:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.stat().st_size > MAX_INPUT_FILE_BYTES:
            raise ValueError(f"NIfTI input exceeds the {MAX_INPUT_FILE_BYTES:,}-byte file limit")
        try:
            import nibabel as nib
        except ImportError as error:
            raise OptionalDependencyError(
                "NIfTI loading requires the imaging extra: python -m pip install -e .[imaging]"
            ) from error

        try:
            loaded = nib.load(str(path))
            shape = tuple(int(value) for value in loaded.shape)
            validate_volume_shape(shape, "NIfTI")
            dtype = np.dtype(loaded.dataobj.dtype)
            if dtype.hasobject or not np.issubdtype(dtype, np.number):
                raise ValueError("NIfTI volume must use a numeric dtype")
            source_payload_bytes = prod(shape) * int(dtype.itemsize)
            if source_payload_bytes > MAX_INPUT_FILE_BYTES:
                raise ValueError(
                    f"NIfTI voxel payload exceeds the {MAX_INPUT_FILE_BYTES:,}-byte limit"
                )
            data = np.asarray(loaded.get_fdata(dtype=np.float32, caching="unchanged"))
            header = loaded.header
            spacing = tuple(float(value) for value in header.get_zooms()[:3])
            affine = np.asarray(loaded.affine, dtype=np.float32)
        except ValueError:
            raise
        except (OSError, EOFError, TypeError) as error:
            raise ValueError("invalid NIfTI volume") from error

        image = NeuroImage(
            patient_id=patient_id or path.name.split(".")[0],
            scan_type="MRI",
            data=data,
            affine=affine,
            metadata={
                "source_format": "nifti",
                "voxel_spacing": spacing,
                "header_zoom_count": len(header.get_zooms()),
                "coordinate_system": "RAS+ millimeters (NIfTI affine)",
            },
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

        candidates = self._dicom_candidates(path)
        headers: list[tuple[Path, Any]] = []
        for candidate in candidates:
            try:
                dataset = pydicom.dcmread(str(candidate), force=False, stop_before_pixels=True)
            except MemoryError:
                raise
            except Exception as error:
                if candidate.suffix.lower() == ".dcm":
                    raise ValueError("a DICOM instance has an unreadable header") from error
                continue
            if all(hasattr(dataset, name) for name in ("Rows", "Columns", "BitsAllocated")):
                headers.append((candidate, dataset))
        if not headers:
            raise ValueError("no readable DICOM image headers found")

        series_uids = {str(getattr(dataset, "SeriesInstanceUID", "")) for _, dataset in headers}
        if "" in series_uids:
            raise ValueError("DICOM image is missing its series identifier")
        if len(series_uids) != 1:
            raise ValueError(
                "DICOM input contains multiple image series; provide one series at a time"
            )

        parsed: list[dict[str, Any]] = []
        for candidate, dataset in headers:
            rows = self._dicom_positive_int(dataset, "Rows")
            columns = self._dicom_positive_int(dataset, "Columns")
            bits_allocated = self._dicom_positive_int(dataset, "BitsAllocated")
            if bits_allocated not in (8, 16, 32):
                raise ValueError("DICOM BitsAllocated must be 8, 16, or 32")
            frames = self._dicom_positive_int(dataset, "NumberOfFrames", default=1)
            if frames > MAX_TEMPORAL_FRAMES:
                raise ValueError(
                    f"DICOM time/frame count exceeds the {MAX_TEMPORAL_FRAMES}-frame limit"
                )
            if frames != 1:
                raise ValueError(
                    "multi-frame DICOM is unsupported; provide a single-frame slice series"
                )
            samples = self._dicom_positive_int(dataset, "SamplesPerPixel", default=1)
            if samples != 1:
                raise ValueError("only single-sample grayscale DICOM series are supported")
            spacing = self._dicom_float_tuple(dataset, "PixelSpacing", 2)
            orientation = self._dicom_float_tuple(dataset, "ImageOrientationPatient", 6)
            position = self._dicom_float_tuple(dataset, "ImagePositionPatient", 3)
            if any(value <= 0 for value in spacing):
                raise ValueError("DICOM pixel spacing must be positive")
            parsed.append(
                {
                    "path": candidate,
                    "header": dataset,
                    "rows": rows,
                    "columns": columns,
                    "bits_allocated": bits_allocated,
                    "spacing": np.asarray(spacing, dtype=np.float64),
                    "orientation": np.asarray(orientation, dtype=np.float64),
                    "position": np.asarray(position, dtype=np.float64),
                }
            )

        first = parsed[0]
        shape = (len(parsed), first["rows"], first["columns"])
        validate_volume_shape(shape, "DICOM")
        modalities = {str(getattr(item["header"], "Modality", "UNKNOWN")) for item in parsed}
        if len(modalities) != 1:
            raise ValueError("DICOM series contains inconsistent modalities")
        scan_type = modalities.pop()
        for item in parsed:
            if (item["rows"], item["columns"]) != (first["rows"], first["columns"]):
                raise ValueError("DICOM series contains inconsistent pixel shapes")
            if item["bits_allocated"] != first["bits_allocated"]:
                raise ValueError("DICOM series contains inconsistent pixel encodings")
            if not np.allclose(item["spacing"], first["spacing"], rtol=0.0, atol=1e-5):
                raise ValueError("DICOM series contains inconsistent pixel spacing")
            if not np.allclose(item["orientation"], first["orientation"], rtol=0.0, atol=1e-4):
                raise ValueError("DICOM series contains inconsistent orientations")

        self._validate_dicom_identity(parsed, "StudyInstanceUID")
        self._validate_dicom_identity(parsed, "FrameOfReferenceUID")
        self._validate_dicom_sop_instances(parsed)

        row_direction = first["orientation"][:3]
        column_direction = first["orientation"][3:]
        if not np.isclose(np.linalg.norm(row_direction), 1.0, atol=1e-4):
            raise ValueError("DICOM row orientation must be a unit vector")
        if not np.isclose(np.linalg.norm(column_direction), 1.0, atol=1e-4):
            raise ValueError("DICOM column orientation must be a unit vector")
        if abs(float(np.dot(row_direction, column_direction))) > 1e-4:
            raise ValueError("DICOM row and column orientations must be orthogonal")
        normal = np.cross(row_direction, column_direction)
        normal /= np.linalg.norm(normal)
        parsed.sort(key=lambda item: float(np.dot(item["position"], normal)))
        positions = np.stack([item["position"] for item in parsed], axis=0)
        if len(parsed) > 1:
            steps = np.diff(positions, axis=0)
            projected_steps = steps @ normal
            if np.any(projected_steps <= 1e-4):
                raise ValueError("DICOM slices have duplicate or reversed physical positions")
            slice_spacing = float(np.median(projected_steps))
            spacing_tolerance = max(0.1, slice_spacing * 0.01)
            if not np.allclose(projected_steps, slice_spacing, rtol=0.0, atol=spacing_tolerance):
                raise ValueError(
                    "DICOM slice positions are irregular and cannot be represented by one affine"
                )
            slice_step = np.mean(steps, axis=0)
            if not np.allclose(steps, slice_step, rtol=0.0, atol=spacing_tolerance):
                raise ValueError("DICOM slice offsets are not consistently spaced")
        else:
            header = first["header"]
            raw_spacing = getattr(header, "SpacingBetweenSlices", None)
            if raw_spacing is None:
                raw_spacing = getattr(header, "SliceThickness", 1.0)
            slice_spacing = abs(float(raw_spacing))
            if not np.isfinite(slice_spacing) or slice_spacing <= 0:
                raise ValueError("single-slice DICOM spacing must be finite and positive")
            slice_step = normal * slice_spacing

        data = np.empty(shape, dtype=np.float32)
        for slice_index, item in enumerate(parsed):
            try:
                dataset = pydicom.dcmread(str(item["path"]), force=False)
            except MemoryError:
                raise
            except Exception as error:
                raise ValueError("a DICOM instance could not be decoded") from error
            if not hasattr(dataset, "PixelData"):
                raise ValueError("DICOM image is missing pixel data")
            if str(getattr(dataset, "SeriesInstanceUID", "")) != str(
                getattr(item["header"], "SeriesInstanceUID", "")
            ):
                raise ValueError("DICOM files changed while the series was being read")
            full_shape = (
                self._dicom_positive_int(dataset, "Rows"),
                self._dicom_positive_int(dataset, "Columns"),
            )
            header_shape = (item["rows"], item["columns"])
            if full_shape != header_shape:
                raise ValueError("DICOM files changed while the series was being read")
            if (
                self._dicom_positive_int(dataset, "BitsAllocated") != item["bits_allocated"]
                or self._dicom_positive_int(dataset, "NumberOfFrames", default=1) != 1
                or self._dicom_positive_int(dataset, "SamplesPerPixel", default=1) != 1
            ):
                raise ValueError("DICOM files changed while the series was being read")
            for name in ("StudyInstanceUID", "FrameOfReferenceUID", "SOPInstanceUID"):
                if getattr(dataset, name, None) != getattr(item["header"], name, None):
                    raise ValueError("DICOM files changed while the series was being read")
            if str(getattr(dataset, "Modality", "UNKNOWN")) != scan_type:
                raise ValueError("DICOM files changed while the series was being read")
            full_orientation = self._dicom_float_tuple(dataset, "ImageOrientationPatient", 6)
            full_position = self._dicom_float_tuple(dataset, "ImagePositionPatient", 3)
            full_spacing = self._dicom_float_tuple(dataset, "PixelSpacing", 2)
            if (
                full_orientation != tuple(item["orientation"])
                or full_position != tuple(item["position"])
                or full_spacing != tuple(item["spacing"])
            ):
                raise ValueError("DICOM files changed while the series was being read")
            pixel_array = np.asarray(dataset.pixel_array)
            if pixel_array.ndim != 2 or pixel_array.shape != (first["rows"], first["columns"]):
                raise ValueError("DICOM pixel data does not match the declared single-frame shape")
            slope = float(getattr(dataset, "RescaleSlope", 1.0))
            intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
            if not np.isfinite(slope) or not np.isfinite(intercept):
                raise ValueError("DICOM rescale values must be finite")
            np.multiply(pixel_array, slope, out=data[slice_index], casting="unsafe")
            np.add(data[slice_index], intercept, out=data[slice_index], casting="unsafe")
            del pixel_array
            del dataset

        affine_lps = np.eye(4, dtype=np.float64)
        affine_lps[:3, 0] = slice_step
        affine_lps[:3, 1] = column_direction * first["spacing"][0]
        affine_lps[:3, 2] = row_direction * first["spacing"][1]
        affine_lps[:3, 3] = positions[0]
        lps_to_ras = np.diag((-1.0, -1.0, 1.0, 1.0))
        affine_ras = (lps_to_ras @ affine_lps).astype(np.float32)
        image = NeuroImage(
            patient_id=patient_id or "anonymous",
            scan_type=scan_type,
            data=data,
            affine=affine_ras,
            metadata={
                "source_format": "dicom",
                "voxel_spacing": (
                    slice_spacing,
                    float(first["spacing"][0]),
                    float(first["spacing"][1]),
                ),
                "series_count": len(parsed),
                "deidentified_patient_id": patient_id or "anonymous",
                "coordinate_system": "RAS+ millimeters (DICOM LPS converted to RAS+)",
                "voxel_axis_order": "slice, row, column",
            },
            source_path=str(path),
        ).validate()
        return self._store(image)

    @staticmethod
    def _dicom_candidates(path: Path) -> list[Path]:
        if path.is_file():
            candidates = [path]
        else:
            candidates = []
            for candidate in path.rglob("*"):
                if candidate.is_file():
                    candidates.append(candidate)
                    if len(candidates) > MAX_DICOM_INSTANCES:
                        raise ValueError(
                            f"DICOM input exceeds the {MAX_DICOM_INSTANCES}-file instance limit"
                        )
            candidates.sort()
        total_bytes = 0
        for candidate in candidates:
            try:
                file_size = candidate.stat().st_size
            except OSError as error:
                raise ValueError("DICOM input changed while it was being scanned") from error
            if file_size > MAX_INPUT_FILE_BYTES:
                raise ValueError("DICOM instance exceeds the configured file-size limit")
            total_bytes += file_size
            if total_bytes > MAX_DICOM_TOTAL_BYTES:
                raise ValueError("DICOM input exceeds the configured total file-size limit")
        return candidates

    @staticmethod
    def _dicom_positive_int(dataset: Any, name: str, default: int | None = None) -> int:
        raw = getattr(dataset, name, default)
        try:
            value = int(raw)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"DICOM {name} must be a positive integer") from error
        if value < 1:
            raise ValueError(f"DICOM {name} must be a positive integer")
        return value

    @staticmethod
    def _dicom_float_tuple(dataset: Any, name: str, length: int) -> tuple[float, ...]:
        raw = getattr(dataset, name, None)
        try:
            values = tuple(float(value) for value in raw)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"DICOM {name} must contain {length} finite values") from error
        if len(values) != length or not np.all(np.isfinite(values)):
            raise ValueError(f"DICOM {name} must contain {length} finite values")
        return values

    @staticmethod
    def _validate_dicom_identity(parsed: list[dict[str, Any]], name: str) -> None:
        values = [getattr(item["header"], name, None) for item in parsed]
        present = {str(value) for value in values if value is not None}
        if len(present) > 1 or (present and any(value is None for value in values)):
            raise ValueError(f"DICOM series contains inconsistent {name}")

    @staticmethod
    def _validate_dicom_sop_instances(parsed: list[dict[str, Any]]) -> None:
        values = [getattr(item["header"], "SOPInstanceUID", None) for item in parsed]
        if any(value is None for value in values):
            raise ValueError("DICOM image is missing its instance identifier")
        present = [str(value) for value in values if value is not None]
        if len(present) != len(set(present)):
            raise ValueError("DICOM series contains duplicate instance identifiers")

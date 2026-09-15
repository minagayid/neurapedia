from __future__ import annotations

import gc
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from pathlib import Path
from unittest import mock
import weakref

import numpy as np

from neurapedia.io import NeuroimagingDataLoader
from neurapedia.limits import MAX_SPATIAL_VOXELS


class NeuroimagingIoTests(unittest.TestCase):
    def test_missing_nifti_path_fails_instead_of_fabricating_a_scan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.nii.gz"

            with self.assertRaises(FileNotFoundError):
                NeuroimagingDataLoader().load_nifti(missing)

    def test_missing_dicom_path_fails_instead_of_fabricating_a_scan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.dcm"

            with self.assertRaises(FileNotFoundError):
                NeuroimagingDataLoader().load_dicom(missing)

    def test_numpy_volume_is_a_small_dependency_free_fixture_format(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "volume.npy"
            np.save(path, np.zeros((6, 7, 8), dtype=np.float32))

            image = NeuroimagingDataLoader().load(path, patient_id="npy-case")

        self.assertEqual(image.patient_id, "npy-case")
        self.assertEqual(image.data.shape, (6, 7, 8))
        self.assertEqual(image.metadata["source_format"], "npy")

    def test_numpy_time_series_is_loaded_without_temporal_averaging(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "series.npy"
            expected = np.stack(
                [np.full((6, 7, 8), value, dtype=np.float32) for value in (1.0, 5.0, 9.0)],
                axis=-1,
            )
            np.save(path, expected)

            image = NeuroimagingDataLoader().load_numpy(path)

        self.assertEqual(image.data.shape, expected.shape)
        self.assertTrue(np.array_equal(image.data, expected))
        self.assertTrue(np.array_equal(image.spatial_data_for_frame(1), expected[..., 1]))

    def test_oversized_numpy_header_is_rejected_before_loading_voxel_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oversized.npy"
            with path.open("wb") as stream:
                np.lib.format.write_array_header_1_0(
                    stream,
                    {"descr": np.dtype(np.float32).str, "fortran_order": False,
                     "shape": (MAX_SPATIAL_VOXELS + 1, 1, 1)},
                )
            with mock.patch("neurapedia.io.np.load", side_effect=AssertionError("payload read")):
                with self.assertRaisesRegex(ValueError, "spatial volume exceeds"):
                    NeuroimagingDataLoader().load_numpy(path)

    def test_nifti_shape_is_checked_before_get_fdata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oversized.nii"
            path.write_bytes(b"header")
            loaded = SimpleNamespace(
                shape=(MAX_SPATIAL_VOXELS + 1, 1, 1),
                dataobj=SimpleNamespace(dtype=np.dtype(np.float32)),
                get_fdata=mock.Mock(side_effect=AssertionError("voxel payload read")),
            )
            nibabel = ModuleType("nibabel")
            nibabel.load = mock.Mock(return_value=loaded)
            with mock.patch.dict("sys.modules", {"nibabel": nibabel}):
                with self.assertRaisesRegex(ValueError, "spatial volume exceeds"):
                    NeuroimagingDataLoader().load_nifti(path)
            loaded.get_fdata.assert_not_called()

    def test_nifti_time_series_is_loaded_without_temporal_averaging(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "series.nii"
            path.write_bytes(b"header")
            expected = np.stack(
                [np.full((6, 7, 8), value, dtype=np.float32) for value in (1.0, 5.0, 9.0)],
                axis=-1,
            )
            loaded = SimpleNamespace(
                shape=expected.shape,
                dataobj=SimpleNamespace(dtype=np.dtype(np.float32)),
                get_fdata=mock.Mock(return_value=expected),
                header=SimpleNamespace(get_zooms=lambda: (1.0, 1.0, 1.0, 2.0)),
                affine=np.eye(4),
            )
            nibabel = ModuleType("nibabel")
            nibabel.load = mock.Mock(return_value=loaded)
            with mock.patch.dict("sys.modules", {"nibabel": nibabel}):
                image = NeuroimagingDataLoader().load_nifti(path)

        self.assertTrue(np.array_equal(image.data, expected))
        self.assertTrue(np.array_equal(image.spatial_data_for_frame(1), expected[..., 1]))

    def test_dicom_series_sorts_by_oriented_position_and_builds_ras_affine(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs = [("first.dcm", 2, 2, 30), ("second.dcm", 0, 0, 10), ("third.dcm", 1, 1, 20)]
            headers = {}
            full = {}
            for filename, x, z, value in specs:
                path = root / filename
                path.write_bytes(b"DICOM")
                fields = dict(
                    SeriesInstanceUID="series-1",
                    StudyInstanceUID="study-1",
                    FrameOfReferenceUID="frame-1",
                    SOPInstanceUID=filename,
                    Rows=2,
                    Columns=3,
                    BitsAllocated=16,
                    NumberOfFrames=1,
                    SamplesPerPixel=1,
                    PixelSpacing=[2.0, 3.0],
                    ImageOrientationPatient=[0, 1, 0, 0, 0, 1],
                    ImagePositionPatient=[float(x), 0.0, float(z)],
                    Modality="MR",
                )
                headers[str(path)] = SimpleNamespace(**fields)
                full[str(path)] = SimpleNamespace(
                    **fields,
                    PixelData=b"pixels",
                    pixel_array=np.full((2, 3), value, dtype=np.int16),
                )
            pydicom = ModuleType("pydicom")

            def dcmread(filename, **kwargs):
                return headers[filename] if kwargs.get("stop_before_pixels") else full[filename]

            pydicom.dcmread = dcmread
            with mock.patch.dict("sys.modules", {"pydicom": pydicom}):
                image = NeuroimagingDataLoader().load_dicom(root)

        self.assertEqual(image.data[:, 0, 0].tolist(), [10.0, 20.0, 30.0])
        np.testing.assert_allclose(
            image.affine,
            np.array([[-1, 0, 0, 0], [0, 0, -3, 0], [1, 2, 0, 0], [0, 0, 0, 1]], dtype=np.float32),
        )
        self.assertIn("RAS+", image.metadata["coordinate_system"])
        self.assertEqual(image.metadata["voxel_axis_order"], "slice, row, column")

    def test_dicom_decode_releases_each_dataset_and_avoids_full_volume_stack(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            headers = {}
            fields_by_path = {}
            for index in range(2):
                path = root / f"slice-{index}.dcm"
                path.write_bytes(b"DICOM")
                fields = dict(
                    SeriesInstanceUID="series-1",
                    StudyInstanceUID="study-1",
                    FrameOfReferenceUID="frame-1",
                    SOPInstanceUID=f"instance-{index}",
                    Rows=2,
                    Columns=2,
                    BitsAllocated=16,
                    NumberOfFrames=1,
                    SamplesPerPixel=1,
                    PixelSpacing=[1.0, 1.0],
                    ImageOrientationPatient=[1, 0, 0, 0, 1, 0],
                    ImagePositionPatient=[0.0, 0.0, float(index)],
                    Modality="MR",
                )
                headers[str(path)] = SimpleNamespace(**fields)
                fields_by_path[str(path)] = fields

            dataset_refs = []
            pixel_array_refs = []

            class PixelDataset:
                def __init__(self, fields, value):
                    self.__dict__.update(fields)
                    self.PixelData = b"pixels"
                    self._pixel_array = np.full((2, 2), value, dtype=np.int16)

                @property
                def pixel_array(self):
                    return self._pixel_array

            pydicom = ModuleType("pydicom")

            def dcmread(filename, **kwargs):
                if kwargs.get("stop_before_pixels"):
                    return headers[filename]
                gc.collect()
                if any(reference() is not None for reference in dataset_refs):
                    raise AssertionError("previous DICOM Dataset is still retained")
                if any(reference() is not None for reference in pixel_array_refs):
                    raise AssertionError("previous decoded pixel array is still retained")
                dataset = PixelDataset(fields_by_path[filename], value=len(dataset_refs) + 1)
                dataset_refs.append(weakref.ref(dataset))
                pixel_array_refs.append(weakref.ref(dataset._pixel_array))
                return dataset

            pydicom.dcmread = dcmread
            with (
                mock.patch.dict("sys.modules", {"pydicom": pydicom}),
                mock.patch("neurapedia.io.np.stack", wraps=np.stack) as stack,
            ):
                image = NeuroimagingDataLoader().load_dicom(root)

        self.assertEqual(image.data[:, 0, 0].tolist(), [1.0, 2.0])
        self.assertEqual(stack.call_count, 1)  # Only the small slice-position metadata stack.
        self.assertTrue(all(reference() is None for reference in dataset_refs))
        self.assertTrue(all(reference() is None for reference in pixel_array_refs))

    def test_mixed_dicom_series_are_rejected_before_pixel_decode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            headers = {}
            for index in range(2):
                path = root / f"slice-{index}.dcm"
                path.write_bytes(b"DICOM")
                headers[str(path)] = SimpleNamespace(
                    SeriesInstanceUID=f"series-{index}", Rows=2, Columns=2, BitsAllocated=16
                )
            pydicom = ModuleType("pydicom")
            pydicom.dcmread = lambda filename, **kwargs: headers[filename]
            with mock.patch.dict("sys.modules", {"pydicom": pydicom}):
                with self.assertRaisesRegex(ValueError, "multiple image series"):
                    NeuroimagingDataLoader().load_dicom(root)

    def test_dicom_temporal_frame_limit_is_checked_before_pixel_decode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frames.dcm"
            path.write_bytes(b"DICOM")
            header = SimpleNamespace(
                SeriesInstanceUID="series-1",
                Rows=2,
                Columns=2,
                BitsAllocated=16,
                NumberOfFrames=513,
            )
            pydicom = ModuleType("pydicom")
            pydicom.dcmread = mock.Mock(return_value=header)
            with mock.patch.dict("sys.modules", {"pydicom": pydicom}):
                with self.assertRaisesRegex(ValueError, "512-frame limit"):
                    NeuroimagingDataLoader().load_dicom(path)

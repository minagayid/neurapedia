from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from neurapedia.io import NeuroimagingDataLoader


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


import unittest
import numpy as np
from neurapedia.evaluation import segmentation_metrics, probability_metrics


class EvaluationTests(unittest.TestCase):
    def test_overlap_and_empty_case_convention(self):
        result = segmentation_metrics([1, 1, 0], [1, 0, 1])
        self.assertEqual(result["dice"], .5)
        self.assertAlmostEqual(result["iou"], 1 / 3)
        self.assertEqual(result["false_positive_voxels"], 1)
        self.assertEqual(result["false_negative_voxels"], 1)
        self.assertEqual(segmentation_metrics([0], [0])["dice"], 1.)

    def test_calibration_and_invalid_inputs(self):
        self.assertAlmostEqual(probability_metrics([0., 1.], [0, 1])["brier_score"], 0.)
        self.assertAlmostEqual(probability_metrics([.5, .5], [0, 1])["ece"], 0.)
        for invalid in [[np.nan], [1.1]]:
            with self.assertRaises(ValueError):
                probability_metrics(invalid, [1])
        with self.assertRaises(ValueError):
            segmentation_metrics([1, 0], [1])

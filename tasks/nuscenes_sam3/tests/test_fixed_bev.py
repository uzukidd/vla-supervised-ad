"""Fixed ROI semantic BEV tests, using synthetic points only."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from tasks.nuscenes_sam3.python import pointcloud_bev as bev


ROI = (-50.0, -50.0, 50.0, 50.0)


class FixedRoiTests(unittest.TestCase):
    def raster(self, xyz, prediction=None, target=None, covered=None, **kwargs):
        xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        if prediction is None:
            prediction = np.full(len(xyz), 11, dtype=np.uint8)
        if covered is None:
            covered = np.ones(len(xyz), dtype=bool)
        return bev.rasterize_bev(xyz, prediction, target, covered, **{"bounds": ROI, **kwargs})

    def test_exact_500_square_orientation_and_lower_inclusive_upper_exclusive(self):
        below_upper = np.nextafter(50.0, -np.inf)
        xyz = np.array([[-50, -50, 0], [below_upper, below_upper, 0],
                        [50, 0, 0], [0, 50, 0], [-50.01, 0, 0], [0, -50.01, 0],
                        [0, 0, 0], [-50, below_upper, 0], [below_upper, -50, 0]])
        labels = np.arange(1, len(xyz) + 1, dtype=np.uint8)
        raster = self.raster(xyz, prediction=labels)
        self.assertEqual(raster["occupied"].shape, (500, 500))
        self.assertEqual(raster["prediction"][499, 0], 1)
        self.assertEqual(raster["prediction"][0, 499], 2)
        self.assertEqual(raster["prediction"][249, 250], 7)
        self.assertEqual(raster["prediction"][0, 0], 8)
        self.assertEqual(raster["prediction"][499, 499], 9)
        self.assertEqual(int(raster["point_count"].sum()), 5)
        metadata = raster["metadata"]
        self.assertEqual(metadata["point_count"], 5)
        self.assertEqual(metadata["roi_input_point_count"], 9)
        self.assertEqual(metadata["roi_outside_point_count"], 4)
        self.assertEqual(metadata["grid_mode"], "fixed_roi")
        self.assertEqual(tuple(metadata[key] for key in ("x_min_m", "y_min_m", "x_max_m", "y_max_m")), ROI)

    def test_empty_input_and_empty_roi_keep_fixed_grid(self):
        for xyz in ([], [[100, 100, 0], [50, 0, 0]]):
            with self.subTest(xyz=xyz):
                raster = self.raster(xyz, target=np.zeros(len(xyz), dtype=np.uint8))
                self.assertEqual(raster["occupied"].shape, (500, 500))
                self.assertFalse(np.any(raster["occupied"]))
                self.assertTrue(np.all(raster["prediction"] == 255))
                self.assertTrue(np.all(raster["target"] == 255))
                self.assertEqual(raster["metadata"]["point_count"], 0)
                self.assertEqual(raster["metadata"]["roi_outside_point_count"], len(xyz))
                self.assertIsNone(raster["metadata"]["valid_extent_m"])
                self.assertEqual(len(raster["prediction_vote_counts"]), 0)

    def test_shifted_bounds_floor_indexing_and_non_square_grid(self):
        xyz = [[-0.3, 0.7, 4], [0.29, 1.09, 2], [-0.01, 0.81, 9], [0.3, 0.8, 0]]
        raster = self.raster(xyz, prediction=np.array([11, 2, 12, 1], dtype=np.uint8),
                             bounds=(-0.3, 0.7, 0.3, 1.1), resolution=0.2)
        np.testing.assert_array_equal(raster["prediction"], [[255, 255, 2], [11, 12, 255]])
        self.assertEqual(raster["metadata"]["roi_outside_point_count"], 1)

    def test_sparse_roi_conservation_independent_votes_and_read_only_inputs(self):
        rng = np.random.default_rng(47)
        xyz = rng.uniform(-70, 70, size=(3000, 3))
        prediction = rng.integers(0, 15, len(xyz), dtype=np.uint8)
        target = rng.integers(0, 15, len(xyz), dtype=np.uint8)
        covered = rng.random(len(xyz)) < 0.6
        snapshots = [value.copy() for value in (xyz, prediction, target, covered)]
        for value in (xyz, prediction, target, covered):
            value.flags.writeable = False
        raster = bev.rasterize_bev(xyz, prediction, target, covered, bounds=ROI)
        inside = np.all((xyz[:, :2] >= -50) & (xyz[:, :2] < 50), axis=1)
        columns = np.floor((xyz[inside, 0] + 50) / 0.2).astype(np.int64)
        rows = 499 - np.floor((xyz[inside, 1] + 50) / 0.2).astype(np.int64)
        expected_cells = rows * 500 + columns
        self.assertEqual(int(raster["point_count"].sum()), int(inside.sum()))
        self.assertEqual(int(raster["covered_count"].sum()), int(covered[inside].sum()))
        for mode, labels in (("prediction", prediction), ("target", target)):
            keys, counts = np.unique(expected_cells * 256 + labels[inside], return_counts=True)
            np.testing.assert_array_equal(raster[f"{mode}_vote_cells"], keys // 256)
            np.testing.assert_array_equal(raster[f"{mode}_vote_classes"], keys % 256)
            np.testing.assert_array_equal(raster[f"{mode}_vote_counts"], counts)
            reconstructed = np.zeros(250000, dtype=np.uint32)
            np.add.at(reconstructed, raster[f"{mode}_vote_cells"], raster[f"{mode}_vote_counts"])
            np.testing.assert_array_equal(reconstructed.reshape(500, 500), raster["point_count"])
        for value, snapshot in zip((xyz, prediction, target, covered), snapshots):
            np.testing.assert_array_equal(value, snapshot)

    def test_bounds_validation_and_integer_tolerance(self):
        invalid = [(-50, -50, 50.01, 50), (0, 0, 0, 1), (1, 0, 0, 1), (0, 2, 1, 1),
                   (0, 0, float("inf"), 1), (float("nan"), 0, 1, 1),
                   (0, 0, 1, -float("inf")), (0, 0, 1), [0, 0, 1, 1],
                   (False, 0, 1, 1), ("0", 0, 1, 1), (0, 0, 1e308, 1)]
        for bounds in invalid:
            with self.subTest(bounds=bounds):
                with self.assertRaises((TypeError, ValueError)):
                    self.raster([], bounds=bounds)
        accepted = self.raster([], bounds=(0, 0, 1 + 1e-10, 1), resolution=0.2)
        self.assertEqual(accepted["point_count"].shape, (5, 5))
        with self.assertRaises(ValueError):
            self.raster([], bounds=(0, 0, 1 + 1e-7, 1), resolution=0.2)

    def test_none_preserves_automatic_extent_and_empty_origin(self):
        raster = self.raster([[100, 100, 0]], bounds=None)
        self.assertEqual(raster["prediction"].shape, (1, 1))
        self.assertEqual(raster["metadata"]["grid_mode"], "automatic_extent")
        self.assertEqual(raster["metadata"]["roi_outside_point_count"], 0)
        self.assertEqual(self.raster([], bounds=None)["prediction"].shape, (1, 1))



if __name__ == "__main__":
    unittest.main()

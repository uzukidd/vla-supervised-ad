"""Tests for unique point fusion, coverage and full point-cloud export."""
from pathlib import Path
import json
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from pointcloud_semantics import UniquePointMetrics, fuse_point_labels, prediction_colors, write_ply
from sam3_sidewalk import make_palette


class PointCloudSemanticsTests(unittest.TestCase):
    def test_majority_ties_zero_abstention_and_coverage(self):
        result = fuse_point_labels(6, [
            {"point_indices": np.array([0, 1, 2, 3]), "prediction": np.array([1, 1, 0, 0])},
            {"point_indices": np.array([0, 1, 2, 4]), "prediction": np.array([2, 2, 2, 1])},
            {"point_indices": np.array([0, 1]), "prediction": np.array([2, 0])},
        ], 3)
        np.testing.assert_array_equal(result["prediction"], [2, 1, 2, 0, 1, 0])
        np.testing.assert_array_equal(result["covered"], [1, 1, 1, 1, 1, 0])
        np.testing.assert_array_equal(result["observation_count"], [3, 3, 2, 1, 1, 0])
        np.testing.assert_array_equal(result["source_camera"], [1, 0, 1, -1, 1, -1])
        np.testing.assert_array_equal(result["winning_votes"], [2, 1, 1, 0, 1, 0])

    def test_no_observations_empty_cloud_and_invalid_input(self):
        result = fuse_point_labels(2, [], 3)
        self.assertFalse(result["covered"].any())
        self.assertEqual(len(fuse_point_labels(0, [], 3)["prediction"]), 0)
        for indices, labels in (([0, 0], [1, 2]), ([3], [1]), ([0], [-1]), ([0], [3]), ([0], [.5])):
            with self.subTest(indices=indices, labels=labels), self.assertRaises(ValueError):
                fuse_point_labels(3, [{"point_indices": np.array(indices), "prediction": np.array(labels)}], 3)

    def test_unique_point_metrics_preserve_uncovered_and_prediction_zero_penalty(self):
        metrics = UniquePointMetrics({0: "ignore", 1: "road", 2: "car"})
        target = np.array([1, 1, 2, 2, 0], dtype=np.uint8)
        prediction = np.array([1, 0, 1, 0, 2], dtype=np.uint8)
        covered = np.array([1, 1, 1, 0, 1], dtype=bool)
        metrics.update(target, prediction, covered)
        report = metrics.report()
        self.assertEqual(report["coverage"]["valid_gt_coverage"], .75)
        self.assertEqual(report["coverage"]["geometric_coverage"], .8)
        self.assertEqual(report["coverage"]["assigned_fraction_of_covered"], .75)
        self.assertEqual(report["covered_points"]["valid_points"], 3)
        self.assertEqual(report["full_scan"]["valid_points"], 4)
        self.assertAlmostEqual(report["covered_points"]["point_accuracy"], 1 / 3)
        self.assertEqual(report["full_scan"]["point_accuracy"], .25)
        self.assertEqual(report["full_scan"]["per_class"][2]["fn"], 2)
        self.assertEqual(report["coverage"]["per_class"][2]["coverage"], .5)
        with tempfile.TemporaryDirectory() as directory:
            metrics.save(Path(directory))
            saved = json.loads((Path(directory) / "metrics.json").read_text())
            self.assertEqual(saved, report)
            self.assertTrue((Path(directory) / "coverage.csv").is_file())
            self.assertTrue((Path(directory) / "confusion_matrix_covered.csv").is_file())
            self.assertTrue((Path(directory) / "confusion_matrix_full_scan.csv").is_file())

    def test_metrics_empty_and_invalid_uncovered_assignment(self):
        metrics = UniquePointMetrics({0: "ignore", 1: "road"})
        self.assertIsNone(metrics.report()["covered_points"]["mean_iou"])
        self.assertIsNone(metrics.report()["coverage"]["valid_gt_coverage"])
        with self.assertRaises(ValueError):
            metrics.update(np.array([1]), np.array([1]), np.array([False]))

    def test_colors_distinguish_uncovered_and_covered_unassigned(self):
        palette = make_palette()
        result = prediction_colors(np.array([0, 0, 1], dtype=np.uint8), np.array([False, True, True]), palette)
        np.testing.assert_array_equal(result[0], [95, 105, 120])
        np.testing.assert_array_equal(result[1], [190, 190, 190])
        np.testing.assert_array_equal(result[2], palette[1])

    def test_binary_ply_full_point_order_coordinates_colors_labels_and_source(self):
        xyz = np.array([[1, 2, 3], [-4, 5, 6]], dtype=np.float32)
        colors = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)
        labels = np.array([1, 0], dtype=np.uint8)
        covered = np.array([True, False])
        sources = np.array([2, -1], dtype=np.int32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.ply"
            write_ply(path, xyz, colors, labels, covered, sources)
            data = path.read_bytes()
            header, payload = data.split(b"end_header\n", 1)
            self.assertIn(b"element vertex 2", header)
            self.assertIn(b"binary_little_endian", header)
            self.assertEqual(len(payload), 42)
            row0 = struct.unpack("<fffBBBBBi", payload[:21])
            self.assertEqual(row0, (1, 2, 3, 10, 20, 30, 1, 1, 2))
            row1 = struct.unpack("<fffBBBBBi", payload[21:])
            self.assertEqual(row1, (-4, 5, 6, 40, 50, 60, 0, 0, -1))


if __name__ == "__main__":
    unittest.main()

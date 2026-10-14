"""Tests for nuScenes class prompts, sparse visualization and metric scope."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from nuscenes_data import COARSE_CLASS_NAMES, RAW_CLASS_NAMES, RAW_TO_COARSE
from sam3_nuscenes import load_categories, parser, save_point_metrics, save_visualization
from sam3_sidewalk import make_palette
from segmentation_metrics import ConfusionMatrix

ROOT = Path(__file__).resolve().parents[1]


class NuScenesPipelineTests(unittest.TestCase):
    def test_bundled_prompt_files_match_taxonomy_ids(self):
        for taxonomy, file_name, expected in (
            ("lidarseg16", "nuscenes_classes.txt", COARSE_CLASS_NAMES),
            ("lidarseg32", "nuscenes_lidarseg32_classes.txt", RAW_CLASS_NAMES),
        ):
            with self.subTest(taxonomy=taxonomy):
                categories, labels = load_categories("lidarseg", taxonomy, ROOT / "config" / file_name)
                self.assertEqual(labels, expected)
                self.assertEqual([c.id for c in categories], list(range(1, len(expected))))
                self.assertEqual([c.name for c in categories], [expected[i] for i in range(1, len(expected))])

    def test_custom_names_do_not_use_dataset_ids(self):
        categories, labels = load_categories("custom", "lidarseg16", ROOT / "config/classes.txt")
        self.assertEqual(labels[0], "unassigned")
        self.assertEqual(categories[0].id, 1)
        self.assertEqual(categories[0].name, "road")

    def test_taxonomy_prompt_name_mismatch_fails(self):
        with self.assertRaises(ValueError):
            load_categories("lidarseg", "lidarseg16", ROOT / "config/classes.txt")

    def test_defaults_limit_image_count_and_use_validation_front_camera(self):
        args = parser().parse_args([])
        self.assertEqual(args.max_images, 10)
        self.assertEqual(args.cameras, ["CAM_FRONT"])
        self.assertEqual(args.split, "mini_val")
        self.assertEqual(args.taxonomy, "lidarseg16")

    def test_mapping_keeps_driveable_surface_and_ignores_noise_ego(self):
        np.testing.assert_array_equal(RAW_TO_COARSE[[0, 17, 24, 26, 28, 30, 31]], [0, 4, 11, 13, 15, 16, 0])

    def test_point_metrics_use_only_sparse_observations_and_mark_scope(self):
        metrics = ConfusionMatrix(COARSE_CLASS_NAMES, ignore_ids={0})
        # A correct road point, an uncovered road point, and ignored ego/noise.
        metrics.update(np.array([11, 11, 0], dtype=np.uint8), np.array([11, 0, 4], dtype=np.uint8))
        configuration = {"taxonomy": "lidarseg16", "split": "mini_val", "cameras": ["CAM_FRONT"],
                         "processed_images": 1, "status": "completed", "min_depth": 1.0}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_point_metrics(metrics, root, configuration, {"retained_points": 3})
            report = json.loads((root / "metrics.json").read_text())
            self.assertEqual(report["point_accuracy"], .5)
            self.assertEqual(report["mean_iou"], .5)
            self.assertEqual(report["valid_point_observations"], 2)
            self.assertEqual(report["evaluation_scope"]["run_status"], "completed")
            self.assertIn("not dense-image", report["evaluation_scope"]["limitations"])
            self.assertEqual(report["per_class"][11]["gt_points"], 2)
            self.assertEqual(report["per_class"][11]["fn"], 1)
            self.assertTrue((root / "metrics.csv").is_file())
            with (root / "metrics.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["metric"], "point_accuracy")
            self.assertEqual(rows[0]["unit"], "projected_point_observations")
            road = next(row for row in rows if row["id"] == "11")
            self.assertEqual(road["gt_points"], "2")
            self.assertEqual(road["fn"], "1")
            self.assertNotIn("gt_pixels", road)
            self.assertTrue((root / "confusion_matrix.csv").is_file())

    def test_sparse_comparison_and_custom_visualization(self):
        image = Image.new("RGB", (128, 64), (30, 50, 60))
        prediction = np.zeros((64, 128), dtype=np.uint8)
        prediction[:, :64] = 11
        projection = {"pixels": np.array([[10, 20], [70, 20]])}
        target = np.array([11, 13], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("visualizations", "overlays", "color_masks", "point_overlays"):
                (root / name).mkdir()
            save_visualization(root, "lidar", image, prediction, COARSE_CLASS_NAMES, make_palette(), .5, projection, target)
            with Image.open(root / "visualizations/lidar.jpg") as comparison:
                self.assertEqual(comparison.width, 128 * 4)
            self.assertTrue((root / "point_overlays/lidar_gt.jpg").is_file())
            self.assertTrue((root / "point_overlays/lidar_prediction.jpg").is_file())
            save_visualization(root, "custom", image, prediction, COARSE_CLASS_NAMES, make_palette(), .5, None, None)
            with Image.open(root / "visualizations/custom.jpg") as comparison:
                self.assertEqual(comparison.width, 128 * 3)
            self.assertFalse((root / "point_overlays/custom_gt.jpg").exists())


if __name__ == "__main__":
    unittest.main()

"""nuImages pipeline checks: prompts, ignore policy, comparison and metric scope."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tasks.nuimages.python.nuimages_data import NuImagesData
from tasks.nuimages.python.sam3_nuimages import error_image, load_categories, parser, save_metrics, save_visualization
from tasks.sidewalk.python.sam3_sidewalk import make_palette
from shared.python.segmentation_metrics import ConfusionMatrix

ROOT = Path(__file__).resolve().parents[3]


class NuImagesPipelineTests(unittest.TestCase):
    @staticmethod
    def dataset():
        dataset = NuImagesData.__new__(NuImagesData)
        from tasks.nuimages.python.nuimages_data import _CLASS_NAMES
        dataset.class_names = dict(_CLASS_NAMES)
        dataset.active_class_ids = set(range(1, 25)) | {31}
        return dataset

    def test_default_prompt_taxonomy_ids_and_optional_ego(self):
        dataset = self.dataset()
        categories, labels = load_categories(dataset, "nuimages", ROOT / "tasks/nuimages/config/nuimages_classes.txt", False)
        self.assertEqual([c.id for c in categories], list(range(1, 25)))
        self.assertEqual(labels[24], "flat.driveable_surface")
        self.assertEqual(labels[31], "vehicle.ego")
        self.assertEqual(labels[25], "unused_25")
        categories, _ = load_categories(dataset, "nuimages", ROOT / "tasks/nuimages/config/nuimages_classes.txt", True)
        self.assertEqual(categories[-1].id, 31)
        with self.assertRaises(ValueError):
            load_categories(dataset, "nuimages", ROOT / "tasks/nuscenes_sam3/config/nuscenes_classes.txt", False)

    def test_custom_classes_have_independent_ids(self):
        categories, labels = load_categories(self.dataset(), "custom", ROOT / "shared/config/classes.txt", False)
        self.assertEqual(labels[0], "unassigned")
        self.assertEqual(categories[0].id, 1)
        self.assertEqual(categories[0].name, "road")

    def test_default_limits_mini_and_all_keycamera_channels(self):
        args = parser().parse_args([])
        self.assertEqual(args.version, "v1.0-mini")
        self.assertEqual(args.max_images, 10)
        self.assertIsNone(args.cameras)
        self.assertFalse(args.include_ego)

    def test_error_map_ignored_gray_correct_green_uncovered_red(self):
        target = np.array([[0, 31, 2, 24]], dtype=np.uint8)
        prediction = np.array([[2, 1, 2, 0]], dtype=np.uint8)
        image = np.asarray(error_image(target, prediction, {0, 31}))
        np.testing.assert_array_equal(image[0], [[70, 75, 85], [70, 75, 85], [45, 190, 90], [240, 70, 65]])

    def test_metrics_exclude_ego_background_and_report_annotation_fraction(self):
        labels = self.dataset().class_names
        ignored = {0, 31, 255, *range(25, 31)}
        metrics = ConfusionMatrix(labels, ignore_ids=ignored)
        metrics.update(np.array([[0, 31, 2, 24]], dtype=np.uint8), np.array([[2, 24, 2, 0]], dtype=np.uint8))
        configuration = {"version": "v1.0-mini", "processed_images": 1, "status": "completed"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_metrics(metrics, root, configuration, {"image_pixels": 4, "annotated_pixels": 3, "null_masks": 1})
            report = json.loads((root / "metrics.json").read_text())
            self.assertEqual(report["valid_pixels"], 2)
            self.assertEqual(report["pixel_accuracy"], .5)
            self.assertEqual(report["mean_iou"], .5)
            self.assertEqual(report["per_class"][24]["fn"], 1)
            self.assertEqual(report["evaluation_scope"]["valid_gt_pixel_fraction"], .5)
            self.assertEqual(report["evaluation_scope"]["annotated_pixel_fraction"], .75)
            self.assertTrue(report["per_class"][31]["ignored"])
            self.assertTrue(report["per_class"][25]["ignored"])

    def test_six_panel_comparison_and_custom_three_panel_outputs(self):
        image = Image.new("RGB", (160, 90), (50, 75, 110))
        target = np.zeros((90, 160), dtype=np.uint8)
        target[:, :50] = 24
        prediction = np.zeros_like(target)
        prediction[:, :30] = 24
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("visualizations", "overlays", "color_masks", "gt_overlays", "gt_color_masks", "errors"):
                (root / name).mkdir()
            save_visualization(root, "semantic", image, prediction, target, self.dataset().class_names,
                               make_palette(), .5, {0, 31})
            with Image.open(root / "visualizations/semantic.jpg") as panel:
                self.assertEqual(panel.width, 160 * 3)
                self.assertGreaterEqual(panel.height, (90 + 30) * 2)
            with Image.open(root / "errors/semantic.png") as error:
                np.testing.assert_array_equal(np.asarray(error)[10, 40], [240, 70, 65])
            save_visualization(root, "custom", image, prediction, None, self.dataset().class_names,
                               make_palette(), .5, {0})
            self.assertFalse((root / "errors/custom.png").exists())
            self.assertFalse((root / "gt_color_masks/custom.png").exists())
            self.assertTrue((root / "overlays/custom.jpg").exists())


if __name__ == "__main__":
    unittest.main()

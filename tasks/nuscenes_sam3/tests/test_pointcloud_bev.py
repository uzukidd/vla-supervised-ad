"""Synthetic BEV tests; run with the existing vla-supervised WSL Python."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from tasks.nuscenes_sam3.python import pointcloud_bev as bev


class RasterizeBevTests(unittest.TestCase):
    def raster(self, xyz, prediction=None, target=None, covered=None, resolution=1.0):
        xyz = np.asarray(xyz)
        count = len(xyz)
        if prediction is None:
            prediction = np.arange(count, dtype=np.uint8)
        if covered is None:
            covered = np.ones(count, dtype=bool)
        return bev.rasterize_bev(xyz, prediction, target, covered, resolution)

    def test_negative_bounds_grid_edges_and_axis_orientation(self):
        xyz = np.array([[-1.1, -0.1, 0], [0, 1, 0], [-1.1, 1.1, 0], [0.9, -0.1, 0]])
        raster = self.raster(xyz)
        np.testing.assert_array_equal(raster["prediction"], [[2, 255, 1], [255, 255, 255], [0, 255, 3]])
        metadata = raster["metadata"]
        self.assertEqual((metadata["x_min_m"], metadata["y_min_m"],
                          metadata["x_max_m"], metadata["y_max_m"]), (-2, -1, 1, 2))
        self.assertEqual((metadata["width"], metadata["height"]), (3, 3))
        self.assertEqual(metadata["orientation"], "+x right, +y up")
        self.assertEqual(metadata["projection"], "LiDAR XY BEV")
        self.assertEqual(metadata["valid_extent_m"], {"x_min_m": -1.1, "x_max_m": 0.9,
                                                      "y_min_m": -0.1, "y_max_m": 1.1})
        occupied = raster["occupied"]
        np.testing.assert_array_equal(occupied, raster["point_count"] > 0)
        self.assertTrue(np.all(raster["prediction"][~occupied] == 255))
        self.assertFalse(np.any(raster["covered"][~occupied]))
        self.assertNotIn("point_index", raster)
        self.assertNotIn("samepoint_count", raster)

    def test_fractional_resolution_and_exact_maximum_included(self):
        xyz = np.array([[-0.25, -0.5, 0], [0.5, 0.5, 0]])
        result = self.raster(xyz, resolution=0.25)
        metadata = result["metadata"]
        self.assertEqual((metadata["x_min_m"], metadata["y_min_m"],
                          metadata["x_max_m"], metadata["y_max_m"]), (-0.25, -0.5, 0.75, 0.75))
        self.assertEqual(result["prediction"][0, -1], 1)
        self.assertEqual(result["prediction"][-1, 0], 0)
        self.assertEqual(metadata["resolution_m_per_pixel"], 0.25)

    def test_majority_independent_target_zero_and_any_covered(self):
        xyz = np.array([[0.1, 0.2, 1], [0.1, 0.2, 4], [0.1, 0.2, 99],
                        [1.1, 0.2, -4], [1.7, 0.8, -2], [1.2, 0.3, 88]], dtype=np.float32)
        prediction = np.array([2, 2, 3, 0, 0, 5], dtype=np.uint8)
        target = np.array([9, 0, 0, 7, 6, 6], dtype=np.uint8)
        covered = np.array([False, False, True, False, True, False])
        raster = self.raster(xyz, prediction, target, covered)
        np.testing.assert_array_equal(raster["prediction"], [[2, 0]])
        np.testing.assert_array_equal(raster["target"], [[0, 6]])
        np.testing.assert_array_equal(raster["point_count"], [[3, 3]])
        np.testing.assert_array_equal(raster["prediction_count"], [[2, 2]])
        np.testing.assert_array_equal(raster["target_count"], [[2, 2]])
        np.testing.assert_array_equal(raster["covered_count"], [[1, 1]])
        np.testing.assert_array_equal(raster["covered"], [[True, True]])
        self.assertEqual(raster["metadata"]["point_count"], 6)
        self.assertEqual(raster["metadata"]["occupied_cells"], 2)
        self.assertEqual(raster["metadata"]["collision_points"], 4)
        self.assertEqual(raster["metadata"]["rule"], "majority_count_then_lowest_class_id")
        self.assertEqual(raster["metadata"]["tie_break"], "lowest_class_id")
        self.assertEqual(raster["metadata"]["zero_class_policy"], "class 0 participates as unassigned/ignored")
        replacement = self.raster(xyz, prediction, target[::-1], ~covered)
        without_gt = self.raster(xyz, prediction, None, covered)
        np.testing.assert_array_equal(replacement["prediction"], raster["prediction"])
        np.testing.assert_array_equal(without_gt["prediction"], raster["prediction"])
        np.testing.assert_array_equal(without_gt["prediction_count"], raster["prediction_count"])

    def test_ties_pick_lowest_class_including_zero_for_prediction_and_target(self):
        xyz = np.zeros((8, 3))
        prediction = np.array([7, 3, 7, 3, 0, 0, 254, 254], dtype=np.uint8)
        target = np.array([254, 254, 1, 1, 0, 0, 7, 7], dtype=np.uint8)
        raster = self.raster(xyz, prediction, target)
        self.assertEqual(raster["prediction"][0, 0], 0)
        self.assertEqual(raster["target"][0, 0], 0)
        self.assertEqual(raster["prediction_count"][0, 0], 2)
        self.assertEqual(raster["target_count"][0, 0], 2)
        raster = self.raster(xyz[:4], prediction[:4], target[:4])
        self.assertEqual(raster["prediction"][0, 0], 3)
        self.assertEqual(raster["target"][0, 0], 1)

    def test_order_and_z_invariance_all_arrays(self):
        rng = np.random.default_rng(23)
        xyz = rng.uniform(-2, 2, size=(1000, 3))
        prediction = rng.integers(0, 255, size=1000, dtype=np.uint8)
        target = rng.integers(0, 255, size=1000, dtype=np.uint8)
        covered = rng.random(1000) > 0.4
        reference = self.raster(xyz, prediction, target, covered)
        order = rng.permutation(len(xyz))
        moved = xyz[order].copy()
        moved[:, 2] = rng.uniform(-1e6, 1e6, len(xyz))
        actual = self.raster(moved, prediction[order], target[order], covered[order])
        for key, value in reference.items():
            if key == "metadata":
                self.assertEqual(actual[key], value)
            else:
                np.testing.assert_array_equal(actual[key], value)
        unsigned = np.array([[0, 0, 0], [0, 0, 255], [0, 0, 255]], dtype=np.uint8)
        self.assertEqual(self.raster(unsigned)["prediction"][0, 0], 0)

    def test_sparse_histograms_reconstruct_counts_and_winners(self):
        rng = np.random.default_rng(37)
        xyz = rng.uniform(-2, 2, size=(2000, 3))
        prediction = rng.integers(0, 8, size=2000, dtype=np.uint8)
        target = rng.integers(0, 9, size=2000, dtype=np.uint8)
        covered = rng.random(2000) > 0.5
        raster = self.raster(xyz, prediction, target, covered)
        self.assertEqual(int(raster["point_count"].sum()), len(xyz))
        self.assertEqual(int(raster["covered_count"].sum()), int(covered.sum()))
        self.assertEqual(raster["occupied"].dtype, np.bool_)
        self.assertEqual(raster["covered"].dtype, np.bool_)
        for name in ("point_count", "prediction_count", "target_count", "covered_count"):
            self.assertEqual(raster[name].dtype, np.uint32)
        width = raster["metadata"]["width"]
        expected_cells = ((int(np.floor(xyz[:, 1].max())) - np.floor(xyz[:, 1])).astype(np.int64) * width
                          + (np.floor(xyz[:, 0]) - int(np.floor(xyz[:, 0].min()))).astype(np.int64))
        for mode, labels in (("prediction", prediction), ("target", target)):
            cells = raster[f"{mode}_vote_cells"]
            classes = raster[f"{mode}_vote_classes"]
            counts = raster[f"{mode}_vote_counts"]
            self.assertEqual(cells.dtype, np.int64)
            self.assertEqual(classes.dtype, np.uint8)
            self.assertEqual(counts.dtype, np.uint32)
            self.assertEqual(int(counts.sum()), len(xyz))
            keys = cells * 256 + classes
            self.assertTrue(np.all(keys[1:] > keys[:-1]))
            histogram = np.zeros((raster["occupied"].size, 256), dtype=np.uint32)
            histogram[cells, classes] = counts
            np.testing.assert_array_equal(histogram.sum(axis=1), raster["point_count"].ravel())
            occupied = raster["occupied"].ravel()
            np.testing.assert_array_equal(histogram.argmax(axis=1)[occupied], raster[mode].ravel()[occupied])
            np.testing.assert_array_equal(histogram.max(axis=1), raster[f"{mode}_count"].ravel())
            original = np.zeros_like(histogram)
            np.add.at(original, (expected_cells, labels), 1)
            np.testing.assert_array_equal(histogram, original)

    def test_empty_cloud_is_single_empty_cell_with_optional_target(self):
        xyz = np.empty((0, 3), dtype=np.float32)
        for target in (None, np.empty(0, dtype=np.uint8)):
            with self.subTest(target=target):
                raster = self.raster(xyz, target=target, resolution=0.2)
                np.testing.assert_array_equal(raster["occupied"], [[False]])
                np.testing.assert_array_equal(raster["covered"], [[False]])
                np.testing.assert_array_equal(raster["prediction"], [[255]])
                for name in ("point_count", "prediction_count", "covered_count"):
                    np.testing.assert_array_equal(raster[name], [[0]])
                for suffix in ("cells", "classes", "counts"):
                    self.assertEqual(len(raster[f"prediction_vote_{suffix}"]), 0)
                if target is None:
                    for key in raster:
                        if key.startswith("target"):
                            self.assertIsNone(raster[key])
                else:
                    np.testing.assert_array_equal(raster["target"], [[255]])
                    np.testing.assert_array_equal(raster["target_count"], [[0]])
                    for suffix in ("cells", "classes", "counts"):
                        self.assertEqual(len(raster[f"target_vote_{suffix}"]), 0)
                metadata = raster["metadata"]
                self.assertEqual((metadata["width"], metadata["height"]), (1, 1))
                self.assertEqual((metadata["x_min_m"], metadata["x_max_m"]), (0, 0.2))
                self.assertEqual(metadata["point_count"], 0)
                self.assertEqual(metadata["collision_points"], 0)
                self.assertIsNone(metadata["valid_extent_m"])

    def test_resolution_validation(self):
        for resolution in (0, -0.2, float("nan"), float("inf"), -float("inf"),
                           True, "0.2", None, 1 + 2j):
            with self.subTest(resolution=resolution):
                with self.assertRaises((TypeError, ValueError)):
                    self.raster(np.empty((0, 3)), resolution=resolution)

    def test_array_validation(self):
        args = dict(xyz=np.zeros((2, 3)), prediction=np.zeros(2, dtype=np.uint8),
                    target=np.zeros(2, dtype=np.uint8), covered=np.ones(2, dtype=bool))
        changes = [
            {"xyz": [[0, 0, 0], [1, 1, 1]]}, {"xyz": np.zeros((2, 2))},
            {"xyz": np.zeros(6)}, {"xyz": np.zeros((2, 3), dtype=complex)},
            {"xyz": np.zeros((2, 3), dtype=bool)}, {"xyz": np.full((2, 3), np.nan)},
            {"xyz": np.full((2, 3), np.inf)}, {"xyz": np.full((2, 3), "0")},
            {"prediction": np.zeros(2, dtype=np.int64)}, {"prediction": [0, 0]},
            {"prediction": np.zeros((2, 1), dtype=np.uint8)},
            {"prediction": np.array([0, 255], dtype=np.uint8)},
            {"prediction": np.zeros(1, dtype=np.uint8)},
            {"target": np.zeros(2, dtype=np.int64)}, {"target": [0, 0]},
            {"target": np.zeros((2, 1), dtype=np.uint8)},
            {"target": np.zeros(1, dtype=np.uint8)},
            {"target": np.array([255, 0], dtype=np.uint8)},
            {"covered": np.ones(2, dtype=np.uint8)}, {"covered": [True, True]},
            {"covered": np.ones((2, 1), dtype=bool)}, {"covered": np.ones(1, dtype=bool)},
        ]
        for change in changes:
            with self.subTest(change=change):
                with self.assertRaises((TypeError, ValueError)):
                    bev.rasterize_bev(**{**args, **change})
        result = bev.rasterize_bev(**{**args, "prediction": np.full(2, 254, dtype=np.uint8),
                                     "target": np.full(2, 254, dtype=np.uint8)})
        self.assertEqual(result["prediction"][0, 0], 254)

    def test_memory_limit_and_overflow_reject_without_sampling(self):
        with self.assertRaisesRegex(ValueError, "increase resolution"):
            self.raster(np.array([[0, 0, 0], [4000, 4000, 1]], dtype=float))
        with self.assertRaisesRegex(ValueError, "increase resolution"):
            self.raster(np.array([[-1e308, 0, 0], [1e308, 0, 1]]), resolution=1e-308)
        with patch.object(bev, "MAX_CELLS", 4):
            accepted = self.raster(np.array([[0, 0, 0], [1, 1, 0]]))
            self.assertEqual(accepted["occupied"].size, 4)
            with self.assertRaisesRegex(ValueError, "increase resolution"):
                self.raster(np.array([[0, 0, 0], [2, 1, 0]]))


class DensityTintTests(unittest.TestCase):
    def test_exact_formula_monotonic_saturation_and_white_zero(self):
        counts = np.array([0, 1, 2, 4, 8, 16, 31, 32, 33, 10000], dtype=np.uint32)
        base = np.array([10, 90, 200], dtype=np.uint8)
        tinted = bev.density_tint(base, counts)
        alpha = 0.25 + 0.75 * np.log1p(np.minimum(counts, 32)) / np.log1p(32)
        alpha[counts == 0] = 0
        expected = np.rint(255 * (1 - alpha[:, None]) + base * alpha[:, None]).astype(np.uint8)
        np.testing.assert_array_equal(tinted, expected)
        np.testing.assert_array_equal(tinted[0], [255, 255, 255])
        self.assertTrue(np.all(np.diff(tinted.astype(int), axis=0) <= 0))
        np.testing.assert_array_equal(tinted[-3:], np.tile(base, (3, 1)))
        self.assertEqual(tinted.dtype, np.uint8)
        self.assertEqual(bev.density_tint(base, np.array(1)).shape, (3,))
        np.testing.assert_array_equal(bev.density_tint(base, np.array([1, 2]), cap=2)[1], base)

    def test_absolute_counts_do_not_normalize_per_frame(self):
        base = (23, 59, 91)
        single = bev.density_tint(base, np.array([2]))[0]
        busy = bev.density_tint(base, np.array([2, 999]))[0]
        np.testing.assert_array_equal(single, busy)

    def test_validation(self):
        for cap in (0, 1, -1, True, np.bool_(False), 2.0, "32", None):
            with self.subTest(cap=cap):
                with self.assertRaises((TypeError, ValueError)):
                    bev.density_tint((10, 20, 30), np.array([1]), cap)
        for base, counts in (((0, 1), [1]), ((-1, 0, 1), [1]), ((256, 0, 1), [1]),
                             ((np.nan, 0, 1), [1]), ((1, 2, 3), [-1]),
                             ((1, 2, 3), [1.5]), ((1, 2, 3), [np.nan])):
            with self.subTest(base=base, counts=counts):
                with self.assertRaises((TypeError, ValueError)):
                    bev.density_tint(base, np.array(counts))


class SaveBevImagesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.xyz = np.array([[0, 0, 1], [1, 0, 2], [2, 0, 0], [0, 1, 0], [2, 1, 0]], dtype=float)
        self.prediction = np.array([1, 2, 0, 2, 1], dtype=np.uint8)
        self.target = np.array([1, 2, 1, 1, 0], dtype=np.uint8)
        self.covered = np.array([True, False, True, True, True])
        self.names = {0: "unlabeled", 1: "road", 2: "sidewalk"}
        self.palette = np.zeros((256, 3), dtype=np.uint8)
        self.palette[0] = (255, 0, 255)
        self.palette[1] = (10, 20, 30)
        self.palette[2] = (80, 90, 100)
        self.palette[255] = (255, 255, 255)
        self.summary = {"covered_points": {"mean_iou": np.float64(0.5)},
                        "full_scan": {"mean_iou": 0.25},
                        "coverage": {"geometric_coverage": 0.8, "total_points": 5,
                                     "covered_points": 4}}

    def arguments(self, **changes):
        args = dict(output_dir=self.root, xyz=self.xyz, prediction=self.prediction,
                    target=self.target, covered=self.covered, class_names=self.names,
                    palette=self.palette, summary=self.summary, resolution=1.0)
        args.update(changes)
        return args

    def image_array(self, name):
        with Image.open(self.root / f"bev_{name}.png") as image:
            image.load()
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, (3, 2))
            return np.asarray(image).copy()

    def tint_one(self, base):
        return bev.density_tint(base, np.array(1))

    def test_native_png_colors_and_white_empty_vs_uncovered_or_ignored(self):
        bev.save_bev_images(**self.arguments())
        background = bev.BACKGROUND_COLOR
        one = self.tint_one
        np.testing.assert_array_equal(self.image_array("prediction"), [
            [one(self.palette[2]), background, one(self.palette[1])],
            [one(self.palette[1]), one(self.palette[2]), one(bev.UNLABELED_COLOR)]])
        np.testing.assert_array_equal(self.image_array("coverage"), [
            [one(bev.COVERED_COLOR), background, one(bev.COVERED_COLOR)],
            [one(bev.COVERED_COLOR), one(bev.UNCOVERED_COLOR), one(bev.COVERED_COLOR)]])
        np.testing.assert_array_equal(self.image_array("ground_truth"), [
            [one(self.palette[1]), background, one(bev.IGNORED_COLOR)],
            [one(self.palette[1]), one(self.palette[2]), one(self.palette[1])]])
        np.testing.assert_array_equal(self.image_array("errors"), [
            [one(bev.ERROR_COLOR), background, one(bev.IGNORED_COLOR)],
            [one(bev.CORRECT_COLOR), one(bev.IGNORED_COLOR), one(bev.ERROR_COLOR)]])
        np.testing.assert_array_equal(self.image_array("density"), [
            [one(bev.DENSITY_COLOR), background, one(bev.DENSITY_COLOR)],
            [one(bev.DENSITY_COLOR), one(bev.DENSITY_COLOR), one(bev.DENSITY_COLOR)]])
        with Image.open(self.root / "bev_comparison.png") as image:
            image.load()
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.width, 1280)
            self.assertGreater(image.height, 1280)
            self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))

    def test_winning_counts_coverage_counts_and_total_density_have_distinct_bases(self):
        xyz = np.array([[0, 0, z] for z in range(8)] + [[2, 0, 0], [2, 0, 1]], dtype=float)
        prediction = np.array([1, 1, 1, 1, 1, 2, 2, 2, 0, 0], dtype=np.uint8)
        target = np.array([2, 2, 2, 2, 2, 2, 1, 1, 0, 0], dtype=np.uint8)
        covered = np.array([True, True, False, False, False, False, False, False, False, False])
        raster = bev.rasterize_bev(xyz, prediction, target, covered, 1)
        colors = bev._colors(raster, self.palette, 8)
        np.testing.assert_array_equal(colors["prediction"][0, 0], bev.density_tint(self.palette[1], np.array(5), 8))
        np.testing.assert_array_equal(colors["ground_truth"][0, 0], bev.density_tint(self.palette[2], np.array(6), 8))
        np.testing.assert_array_equal(colors["coverage"][0, 0], bev.density_tint(bev.COVERED_COLOR, np.array(2), 8))
        np.testing.assert_array_equal(colors["density"][0, 0], bev.DENSITY_COLOR)
        np.testing.assert_array_equal(colors["errors"][0, 0], bev.density_tint(bev.ERROR_COLOR, np.array(5), 8))
        np.testing.assert_array_equal(colors["prediction"][0, 2], bev.density_tint(bev.UNCOVERED_COLOR, np.array(2), 8))
        np.testing.assert_array_equal(colors["coverage"][0, 2], bev.density_tint(bev.UNCOVERED_COLOR, np.array(2), 8))
        for mode in colors:
            np.testing.assert_array_equal(colors[mode][0, 1], [255, 255, 255])
        # An independently selected target majority agrees even when pairs often differ.
        target[:8] = [1, 2, 1, 2, 2, 1, 1, 1]
        raster = bev.rasterize_bev(xyz, prediction, target, covered, 1)
        np.testing.assert_array_equal(bev._colors(raster, self.palette)["errors"][0, 0],
                                      bev.density_tint(bev.CORRECT_COLOR, np.array(5)))

    def test_numeric_npz_and_metadata_preserve_dense_counts_and_sparse_votes(self):
        metadata = bev.save_bev_images(**self.arguments(density_cap=8))
        loaded_metadata = json.loads((self.root / "bev_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(loaded_metadata["summary"]["covered_points"]["mean_iou"], 0.5)
        self.assertEqual(loaded_metadata["point_count"], len(self.xyz))
        self.assertEqual(metadata["metrics_basis"], "original unique points; BEV cells are display-only")
        self.assertEqual(metadata["density_scaling"]["cap"], 8)
        self.assertIn("not a point metric", metadata["error_basis"])
        raster = bev.rasterize_bev(self.xyz, self.prediction, self.target, self.covered, 1.0)
        with np.load(self.root / "bev_raster.npz", allow_pickle=False) as saved:
            self.assertEqual(set(saved.files), set(raster) - {"metadata"})
            for name in saved.files:
                np.testing.assert_array_equal(saved[name], raster[name])
                self.assertEqual(saved[name].dtype, raster[name].dtype)
            self.assertNotIn("point_index", saved.files)
            self.assertEqual(saved["point_count"].dtype, np.uint32)
        one = bev.density_tint(self.palette[1], np.array(1), 8)
        np.testing.assert_array_equal(self.image_array("prediction")[1, 0], one)

    def test_no_gt_comparison_is_two_side_by_side_panels(self):
        bev.save_bev_images(**self.arguments(target=None))
        self.assertFalse((self.root / "bev_ground_truth.png").exists())
        self.assertFalse((self.root / "bev_errors.png").exists())
        with np.load(self.root / "bev_raster.npz", allow_pickle=False) as saved:
            self.assertFalse(any(key.startswith("target") for key in saved.files))
        with Image.open(self.root / "bev_comparison.png") as image:
            image.load()
            self.assertEqual(image.width, 1280)
            self.assertLess(image.height, 1280)
            self.assertGreater(image.height, 640)

    def test_make_images_false_creates_only_numeric_and_metadata_outputs(self):
        with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None}):
            bev.save_bev_images(**self.arguments(make_images=False))
        self.assertEqual({path.name for path in self.root.iterdir()}, {"bev_raster.npz", "bev_metadata.json"})

    def test_all_source_arrays_and_summary_are_unchanged(self):
        arrays = (self.xyz, self.prediction, self.target, self.covered, self.palette)
        snapshots = [array.copy() for array in arrays]
        summary = copy.deepcopy(self.summary)
        names = self.names.copy()
        for array in arrays:
            array.flags.writeable = False
        bev.save_bev_images(**self.arguments())
        for array, snapshot in zip(arrays, snapshots):
            np.testing.assert_array_equal(array, snapshot)
        self.assertEqual(self.summary, summary)
        self.assertEqual(self.names, names)

    def test_empty_cloud_images_can_be_opened(self):
        bev.save_bev_images(**self.arguments(xyz=np.empty((0, 3)),
                            prediction=np.empty(0, dtype=np.uint8),
                            target=np.empty(0, dtype=np.uint8), covered=np.empty(0, dtype=bool)))
        for path in self.root.glob("*.png"):
            with Image.open(path) as image:
                image.load()
                if path.name != "bev_comparison.png":
                    self.assertEqual(image.size, (1, 1))
                    self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))

    def test_export_validation_happens_before_creating_directory(self):
        invalid = [
            {"palette": self.palette.astype(float)}, {"palette": self.palette[:10]},
            {"palette": self.palette.tolist()}, {"class_names": []},
            {"class_names": {255: "bad"}}, {"class_names": {True: "bad"}},
            {"class_names": {1: 3}}, {"summary": []},
            {"summary": {"bad": float("nan")}}, {"make_images": "false"},
            {"resolution": 0}, {"target": self.target.astype(np.int64)},
        ]
        invalid.extend({"density_cap": cap} for cap in (0, 1, -1, True, 32.0, "32", None))
        for changes in invalid:
            output = self.root / "should_not_exist"
            with self.subTest(changes=changes):
                with self.assertRaises((TypeError, ValueError)):
                    bev.save_bev_images(**self.arguments(output_dir=output, **changes))
                self.assertFalse(output.exists())

    def test_summary_labels_explicitly_name_point_metrics(self):
        normalized = json.loads(bev._json(self.summary))
        lines = bev._summary_lines(normalized)
        self.assertIn("Point metric | covered points / mIoU: 50.00%", lines)
        self.assertIn("Point metric | full scan / mIoU: 25.00%", lines)
        self.assertIn("Point metric | coverage / geometric coverage: 80.00%", lines)


if __name__ == "__main__":
    unittest.main()

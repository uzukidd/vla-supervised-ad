"""Synthetic geometry and Pillow BEV checks; no models or nuScenes required."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tasks.nuscenes_sam3.python.instance_hulls import (
    convex_hull_xy, instance_rgb, points_in_convex_polygon, polygon_area, save_hull_bev,
)


def instance(identifier=1, indices=(), class_id=4, score=0.8, camera="CAM_FRONT"):
    return {"global_instance_id": identifier, "camera": camera, "class_id": class_id,
            "class_name": "car" if class_id <= 10 else "driveable_surface",
            "score": score, "point_indices": list(indices)}


class ConvexHullTests(unittest.TestCase):
    def test_known_triangle_square_ccw_duplicates_and_collinear_interior(self):
        triangle = np.array([[0, 0], [2, 0], [0, 3], [0.1, 0.1], [0, 0]])
        np.testing.assert_array_equal(convex_hull_xy(triangle), [[0, 0], [2, 0], [0, 3]])
        self.assertEqual(polygon_area(convex_hull_xy(triangle)), 3)
        square = np.array([[1, 1], [0, 1], [0, 0], [1, 0], [0.5, 0], [0.5, 0.5]])
        hull = convex_hull_xy(square)
        self.assertEqual(hull.dtype, np.float64)
        np.testing.assert_array_equal(hull, [[0, 0], [1, 0], [1, 1], [0, 1]])
        self.assertEqual(polygon_area(hull), 1)
        self.assertEqual(polygon_area(hull[::-1]), 1)

    def test_empty_singleton_and_collinear_endpoints_have_zero_area(self):
        cases = [(np.empty((0, 2)), np.empty((0, 2))),
                 (np.array([[3, 4], [3, 4]]), np.array([[3, 4]])),
                 (np.array([[2, 4], [0, 0], [1, 2], [-1, -2]]), np.array([[-1, -2], [2, 4]]))]
        for points, expected in cases:
            with self.subTest(points=points):
                hull = convex_hull_xy(points)
                np.testing.assert_array_equal(hull, expected)
                self.assertEqual(polygon_area(hull), 0)
                self.assertEqual(hull.shape, expected.shape)

    def test_closed_containment_both_windings_and_metric_tolerance(self):
        square = np.array([[0., 0], [2, 0], [2, 2], [0, 2]])
        points = np.array([[1, 1], [0, 0], [2, 1], [2 + 5e-8, 1], [2 + 2e-7, 1], [-1, 0]])
        for vertices in (square, square[::-1]):
            np.testing.assert_array_equal(points_in_convex_polygon(points, vertices),
                                          [True, True, True, True, False, False])
        big_square = square * 1000
        self.assertFalse(points_in_convex_polygon(np.array([[2000 + 1e-5, 1000]]), big_square)[0])

    def test_degenerate_containment_uses_distance_and_clamped_projection(self):
        points = np.array([[0., 0], [1, 0], [0.5, 5e-8], [1 + 5e-8, 0],
                           [1 + 2e-7, 0], [0.5, 2e-7], [-2e-7, 0]])
        self.assertFalse(points_in_convex_polygon(points, np.empty((0, 2))).any())
        np.testing.assert_array_equal(points_in_convex_polygon(points, np.array([[0., 0]])),
                                      [True, False, False, False, False, False, False])
        np.testing.assert_array_equal(points_in_convex_polygon(points, np.array([[0., 0], [1, 0]])),
                                      [True, True, True, True, False, False, False])
        self.assertTrue(points_in_convex_polygon(np.array([[0., 0]]), np.array([[0., 0], [0, 0]]))[0])

    def test_permutation_translation_determinism_and_outlier_retention(self):
        rng = np.random.default_rng(17)
        points = np.vstack((rng.normal(0, 0.2, (100, 2)), [[30, 25]]))
        original = points.copy()
        points.flags.writeable = False
        expected = convex_hull_xy(points)
        self.assertTrue(np.any(np.all(expected == [30, 25], axis=1)))
        for unused in range(10):
            np.testing.assert_array_equal(convex_hull_xy(points[rng.permutation(len(points))]), expected)
        shift = np.array([12345., -9123.])
        moved = convex_hull_xy(points + shift)
        np.testing.assert_allclose(moved - shift, expected, atol=2e-12)
        self.assertAlmostEqual(polygon_area(moved), polygon_area(expected), places=9)
        self.assertTrue(points_in_convex_polygon(points, expected).all())
        np.testing.assert_array_equal(points, original)

    def test_validation_finite_shape_numeric_and_tolerance(self):
        for invalid in (np.zeros((2, 3)), np.zeros(2), np.array([[np.nan, 0]]),
                        np.array([[0, np.inf]]), np.array([[1j, 0]]), np.array([["a", "b"]])):
            for function in (convex_hull_xy, polygon_area):
                with self.subTest(function=function.__name__, invalid=invalid):
                    with self.assertRaises((ValueError, TypeError)):
                        function(invalid)
            with self.assertRaises((ValueError, TypeError)):
                points_in_convex_polygon(np.zeros((1, 2)), invalid)
        for tolerance in (-1, np.nan, np.inf, True, "x"):
            with self.assertRaises((ValueError, TypeError)):
                points_in_convex_polygon(np.zeros((1, 2)), np.zeros((1, 2)), tolerance)

    def test_instance_colors_support_ids_above_255(self):
        colors = [instance_rgb(identifier) for identifier in range(1, 601)]
        self.assertEqual(len(set(colors)), 600)
        self.assertNotEqual(instance_rgb(1), instance_rgb(257))
        self.assertNotEqual(instance_rgb(2), instance_rgb(258))
        self.assertEqual(instance_rgb(1001), instance_rgb(np.int64(1001)))
        self.assertTrue(all(0 <= value <= 255 for color in colors for value in color))


class HullBevTests(unittest.TestCase):
    def save(self, directory, xyz, instances, **kwargs):
        return save_hull_bev(Path(directory), np.asarray(xyz, dtype=np.float64).reshape(-1, 3),
                             instances, {4: "car", 11: "driveable_surface"}, **kwargs)

    def test_roi_png_dimensions_counts_original_vertices_and_no_mutation(self):
        upper = np.nextafter(50., -np.inf)
        xyz = np.array([[-50., -50, 0], [upper, -50, 0], [0, upper, 0], [0, 0, 0],
                        [49, 20, 0], [50, 0, 0], [0, 50, 0], [-50.001, 0, 0],
                        [1, 1, 0], [2, 1, 0], [3, 1, 0], [4, 4, 0]])
        instances = [instance(indices=[0, 1, 2, 3, 4, 5, 5, 6, 7]),
                     instance(257, [8, 9, 10], class_id=11, camera="CAM_BACK"),
                     instance(999, [11]), instance(1000)]
        snapshot, original = copy.deepcopy(instances), xyz.copy()
        xyz.flags.writeable = False
        with tempfile.TemporaryDirectory() as directory:
            result = save_hull_bev(Path(directory), xyz, instances, {4: "car", 11: "driveable_surface"})
            self.assertEqual(instances, snapshot)
            np.testing.assert_array_equal(xyz, original)
            first = result["instances"][0]
            self.assertEqual(first["point_count"], 8)
            self.assertEqual(first["support_count"], 5)
            self.assertEqual(first["outside_roi_point_count"], 3)
            self.assertEqual(first["roi_point_indices"], [0, 1, 2, 3, 4])
            hull = np.asarray(first["hull_xy"])
            self.assertTrue(points_in_convex_polygon(xyz[first["roi_point_indices"], :2], hull).all())
            self.assertIn(upper, hull[:, 0])
            self.assertEqual(result["instances"][1]["geometry_type"], "segment")
            self.assertEqual(result["instances"][1]["area_m2"], 0)
            self.assertEqual(result["instances"][2]["geometry_type"], "point")
            self.assertEqual(result["instances"][3]["geometry_type"], "empty")
            self.assertEqual(result["metadata"]["roi_point_count"], 9)
            self.assertEqual(result["metadata"]["assigned_roi_point_count"], 9)
            for name in ("bev_instance_points.png", "bev_instance_hulls.png", "bev_all_regions.png",
                         "per_camera/CAM_FRONT_hulls.png", "per_camera/CAM_BACK_hulls.png"):
                with Image.open(Path(directory) / name) as image:
                    self.assertEqual(image.size, (500, 500))
                    self.assertEqual(image.mode, "RGB")
            with Image.open(Path(directory) / "bev_instance_comparison.png") as image:
                self.assertEqual(image.width, 1000)
                self.assertGreater(image.height, 1000)
            data = json.loads((Path(directory) / "instance_hulls.json").read_text(encoding="utf-8"))
            self.assertEqual(data, result)
            self.assertIn("minimum_convex_polygon", data["metadata"]["rule"])
            self.assertIn("bridge gaps", " ".join(data["metadata"]["warnings"]))

    def test_empty_roi_is_white_and_stuff_excluded_from_foreground_hulls(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.save(directory, [[100, 0, 0]], [instance(indices=[0])])
            self.assertEqual(result["instances"][0]["geometry_type"], "empty")
            for name in ("bev_instance_points.png", "bev_instance_hulls.png", "bev_all_regions.png"):
                with Image.open(Path(directory) / name) as image:
                    self.assertTrue(np.all(np.asarray(image) == 255))
        xyz = [[-20, -20, 0], [20, -20, 0], [0, 20, 0]]
        with tempfile.TemporaryDirectory() as directory:
            self.save(directory, xyz, [instance(indices=[0, 1, 2], class_id=11)])
            with Image.open(Path(directory) / "bev_instance_hulls.png") as image:
                self.assertEqual(image.getpixel((250, 280)), (255, 255, 255))
            with Image.open(Path(directory) / "bev_all_regions.png") as image:
                self.assertNotEqual(image.getpixel((250, 280)), (255, 255, 255))

    def test_point_colors_id_257_and_clamped_roi_corner_mapping(self):
        upper = np.nextafter(50., -np.inf)
        xyz = [[-50, -50, 0], [upper, upper, 0]]
        with tempfile.TemporaryDirectory() as directory:
            self.save(directory, xyz, [instance(1, [0]), instance(257, [1])])
            with Image.open(Path(directory) / "bev_instance_points.png") as image:
                self.assertEqual(image.getpixel((0, 499)), instance_rgb(1))
                self.assertEqual(image.getpixel((499, 0)), instance_rgb(257))

    def test_score_priority_and_input_order_are_deterministic(self):
        xyz = [[-20, -20, 0], [20, -20, 0], [0, 20, 0],
               [-21, -21, 0], [21, -21, 0], [0, 21, 0]]
        low, high = instance(1, [0, 1, 2], score=.2), instance(2, [3, 4, 5], score=.9)
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a", Path(directory) / "b"
            result_a = self.save(a, xyz, [high, low])
            result_b = self.save(b, xyz, [low, high])
            self.assertEqual(result_a, result_b)
            for name in ("bev_instance_points.png", "bev_instance_hulls.png", "bev_all_regions.png",
                         "bev_instance_comparison.png"):
                self.assertEqual((a / name).read_bytes(), (b / name).read_bytes())
            expected = np.full(3, 255.)
            for identifier in (1, 2):
                expected = np.round(expected * (1 - 46 / 255) + np.array(instance_rgb(identifier)) * (46 / 255))
            with Image.open(a / "bev_instance_hulls.png") as image:
                np.testing.assert_allclose(image.getpixel((230, 270)), expected, atol=1)

    def test_cross_camera_point_observations_are_retained_without_merging(self):
        xyz = [[0, 0, 0], [2, 0, 0], [0, 2, 0], [100, 0, 0]]
        instances = [instance(1, [0, 1, 2, 3], camera="CAM_FRONT"),
                     instance(2, [0, 1, 2, 3], camera="CAM_FRONT_LEFT")]
        with tempfile.TemporaryDirectory() as directory:
            result = self.save(directory, xyz, instances)
            self.assertEqual(len(result["instances"]), 2)
            self.assertEqual(result["instances"][0]["hull_xy"], result["instances"][1]["hull_xy"])
            self.assertEqual(result["instances"][0]["roi_point_indices"], [0, 1, 2])
            self.assertEqual(result["instances"][1]["roi_point_indices"], [0, 1, 2])
            metadata = result["metadata"]
            self.assertEqual(metadata["assigned_point_count"], 4)
            self.assertEqual(metadata["assigned_roi_point_count"], 3)
            self.assertEqual(metadata["assigned_point_observation_count"], 8)
            self.assertEqual(metadata["assigned_roi_point_observation_count"], 6)
            self.assertEqual(metadata["summary"]["num_points"], 6)
            self.assertEqual(metadata["summary"]["num_unique_points"], 3)
            self.assertIn("across cameras retained", metadata["assignment_policy"])

    def test_invalid_indices_exclusivity_geometry_and_finite_inputs(self):
        invalid = [instance(indices=[-1]), instance(indices=[1]), instance(indices=[0.5]),
                   instance(indices=[True]), {**instance(indices=[0]), "class_id": 17},
                   {**instance(indices=[0]), "score": np.nan},
                   {**instance(indices=[0]), "camera": "../oops"}]
        with tempfile.TemporaryDirectory() as directory:
            for index, item in enumerate(invalid):
                output = Path(directory) / str(index)
                with self.subTest(item=item), self.assertRaises((ValueError, TypeError)):
                    self.save(output, [[0, 0, 0]], [item])
                self.assertFalse(output.exists())
            with self.assertRaises(ValueError):
                self.save(directory, [[0, 0, 0]], [instance(1, [0]), instance(2, [0])])
            with self.assertRaises(ValueError):
                self.save(directory, [[np.nan, 0, 0]], [])
            with self.assertRaises(ValueError):
                self.save(directory, [], [], resolution=.3)
            with self.assertRaises(ValueError):
                self.save(directory, [], [], roi_size=-1)


if __name__ == "__main__":
    unittest.main()

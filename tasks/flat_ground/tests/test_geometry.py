"""Synthetic NumPy-only ground-plane geometry tests; no models or dataset."""

from __future__ import annotations

import unittest

import numpy as np

from tasks.flat_ground.python.geometry import (
    bev_ground_grid,
    ego_yaw_frame,
    pixels_to_ground,
    pose_matrix,
    project_ground_to_camera,
)


def _axis_quaternion(axis: int, angle: float) -> np.ndarray:
    quaternion = np.zeros(4)
    quaternion[0] = np.cos(angle / 2)
    quaternion[axis + 1] = np.sin(angle / 2)
    return quaternion


def _quaternion_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.concatenate(([a[0] * b[0] - np.dot(a[1:], b[1:])],
                           a[0] * b[1:] + b[0] * a[1:] + np.cross(a[1:], b[1:])))


def _camera(*, tilt: float = 0.25, translation=(4.0, -3.0, 2.0)) -> np.ndarray:
    """Camera x right, y down and z forward, tilted down from global +x."""
    cosine, sine = np.cos(tilt), np.sin(tilt)
    transform = np.eye(4)
    transform[:3, :3] = [[0, -sine, cosine], [-1, 0, 0], [0, -cosine, -sine]]
    transform[:3, 3] = translation
    return transform


class PoseTests(unittest.TestCase):
    def test_pose_uses_wxyz_and_normalizes_quaternion(self):
        matrix = pose_matrix({"rotation": 3 * _axis_quaternion(2, np.pi / 2),
                              "translation": [1, 2, 3]})
        self.assertEqual(matrix.dtype, np.float64)
        np.testing.assert_allclose(matrix @ [1, 0, 0, 1], [1, 3, 3, 1], atol=1e-14)
        np.testing.assert_array_equal(matrix[3], [0, 0, 0, 1])

    def test_ego_frame_removes_roll_pitch_and_altitude(self):
        yaw, pitch, roll = 0.8, -0.3, 0.4
        quaternion = _quaternion_product(
            _axis_quaternion(2, yaw),
            _quaternion_product(_axis_quaternion(1, pitch), _axis_quaternion(0, roll)),
        )
        pose = {"rotation": quaternion, "translation": [20, -7, 12]}
        for plane_z in (0.0, -1.5):
            with self.subTest(plane_z=plane_z):
                transform = ego_yaw_frame(pose, plane_z)
                expected = [[np.cos(yaw), -np.sin(yaw), 0],
                            [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
                np.testing.assert_allclose(transform[:3, :3], expected, atol=1e-14)
                np.testing.assert_array_equal(transform[:3, 3], [20, -7, plane_z])

    def test_pose_validation(self):
        bad_records = [{}, {"rotation": [0, 0, 0, 0], "translation": [0, 0, 0]},
                       {"rotation": [1, 0, 0, np.nan], "translation": [0, 0, 0]},
                       {"rotation": [1, 0, 0], "translation": [0, 0, 0]},
                       {"rotation": [1, 0, 0, 0], "translation": [0, 0]},
                       {"rotation": [1, 0, 0, 0], "translation": [0, np.inf, 0]}]
        for record in bad_records:
            with self.subTest(record=record), self.assertRaises(ValueError):
                pose_matrix(record)
        with self.assertRaises(ValueError):
            ego_yaw_frame({"rotation": [1, 0, 0, 0], "translation": [0, 0, 0]}, np.nan)


class GroundGridTests(unittest.TestCase):
    def test_default_centres_and_orientation(self):
        grid, metadata = bev_ground_grid()
        self.assertEqual(grid.shape, (500, 500, 2))
        self.assertEqual(grid.dtype, np.float64)
        self.assertEqual(metadata["shape"], (500, 500))
        self.assertEqual(metadata["resolution"], 0.2)
        self.assertEqual(metadata["roi_size"], 100)
        np.testing.assert_allclose(grid[0, 0], [-49.9, 49.9])
        np.testing.assert_allclose(grid[-1, -1], [49.9, -49.9])
        np.testing.assert_allclose(np.diff(grid[0, :, 0]), 0.2)
        np.testing.assert_allclose(np.diff(grid[:, 0, 1]), -0.2)
        self.assertEqual(metadata["column_axis"], "+x")
        self.assertEqual(metadata["row_axis"], "-y")

    def test_small_grid(self):
        grid, _ = bev_ground_grid(1, 0.5)
        np.testing.assert_array_equal(grid, [[[-0.25, 0.25], [0.25, 0.25]],
                                           [[-0.25, -0.25], [0.25, -0.25]]])

    def test_malformed_and_excessive_grids(self):
        pairs = [(0, 0.2), (-1, 0.2), (100, 0), (100, -1), (np.nan, 1),
                 (1, np.inf), (1, 0.3), (0.1, 1), (4001, 1), (1e308, 1e-308)]
        for roi_size, resolution in pairs:
            with self.subTest(roi_size=roi_size, resolution=resolution):
                with self.assertRaises(ValueError):
                    bev_ground_grid(roi_size, resolution)


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.camera = _camera()
        self.K = np.array([[120, 0, 100], [0, 120, 60], [0, 0, 1]], dtype=float)

    def test_known_tilted_camera_exact_ground_and_roundtrip(self):
        pixels = np.array([[100, 60], [100, 80], [80, 90], [130, 100]], dtype=float)
        inverse = pixels_to_ground(pixels, self.camera, self.K)
        self.assertTrue(np.all(inverse["valid"]))
        self.assertEqual(inverse["points_global"].dtype, np.float64)
        np.testing.assert_array_equal(inverse["points_global"][:, 2], 0)
        np.testing.assert_allclose(inverse["points_global"][0],
                                   [4 + 2 / np.tan(0.25), -3, 0], atol=1e-12)
        transform = ego_yaw_frame({"rotation": _axis_quaternion(2, 0.7),
                                   "translation": [2, -5, 10]})
        local = (inverse["points_global"] - transform[:3, 3]) @ transform[:3, :3]
        projected = project_ground_to_camera(local[:, :2][None], transform,
                                              self.camera, self.K, (201, 121))
        self.assertTrue(np.all(projected["valid"]))
        np.testing.assert_allclose(projected["uv"][0], pixels, atol=1e-12)
        np.testing.assert_array_equal(projected["pixel_xy"][0], pixels.astype(np.int32))
        np.testing.assert_allclose(projected["depth"][0], inverse["depth"], rtol=1e-7)
        np.testing.assert_allclose(projected["slant_distance"][0],
                                   inverse["slant_distance"], rtol=1e-7)
        np.testing.assert_allclose(projected["incidence"][0],
                                   2 / inverse["slant_distance"], rtol=1e-7)
        self.assertEqual(projected["pixel_xy"].dtype, np.int32)
        for key in ("depth", "slant_distance", "incidence"):
            self.assertEqual(projected[key].dtype, np.float32)
        self.assertEqual(projected["uv"].dtype, np.float64)
        self.assertEqual(projected["valid"].dtype, np.bool_)

    def test_world_plane_height_and_camera_height_incidence(self):
        camera = _camera(translation=(4, -3, 7))
        inverse = pixels_to_ground([[100, 60]], camera, self.K, plane_z=5)
        transform = np.eye(4)
        transform[2, 3] = 5
        result = project_ground_to_camera(inverse["points_global"][:, :2][None],
                                          transform, camera, self.K, (201, 121))
        self.assertTrue(result["valid"].item())
        self.assertEqual(inverse["points_global"][0, 2], 5)
        self.assertAlmostEqual(result["incidence"].item(),
                               2 / inverse["slant_distance"][0], places=7)

    def test_max_distance_is_slant_distance_not_depth(self):
        camera = _camera(tilt=np.pi / 2, translation=(0, 0, 2))
        K = np.eye(3)
        # Camera z is 2 but slant distance is sqrt(20), beyond a 4m limit.
        pixels = np.array([[2.0, 0]])
        accepted = pixels_to_ground(pixels, camera, K, max_distance=5)
        self.assertTrue(accepted["valid"].item())
        self.assertAlmostEqual(accepted["depth"].item(), 2)
        self.assertAlmostEqual(accepted["slant_distance"].item(), np.sqrt(20))
        rejected = pixels_to_ground(pixels, camera, K, max_distance=4)
        self.assertFalse(rejected["valid"].item())
        self.assertTrue(np.all(np.isnan(rejected["points_global"])))
        result = project_ground_to_camera(accepted["points_global"][:, :2][None],
                                          np.eye(4), camera, K, (10, 10), max_distance=4)
        self.assertFalse(result["valid"].item())
        np.testing.assert_array_equal(result["pixel_xy"], -np.ones((1, 1, 2)))
        self.assertTrue(np.all(np.isnan(result["uv"])))

    def test_parallel_upward_and_behind_rays(self):
        camera = _camera(tilt=0, translation=(0, 0, 2))
        result = pixels_to_ground([[0, 0], [0, -1], [0, 1], [0, 1e-11]], camera, np.eye(3))
        np.testing.assert_array_equal(result["valid"], [False, False, True, False])
        self.assertTrue(np.all(np.isnan(result["points_global"][[0, 1, 3]])))
        self.assertTrue(np.all(np.isnan(result["depth"][[0, 1, 3]])))
        # A camera below the plane also rejects rays pointing away from it.
        below = _camera(tilt=0, translation=(0, 0, -2))
        inverse = pixels_to_ground([[0, 1], [0, -1]], below, np.eye(3))
        np.testing.assert_array_equal(inverse["valid"], [False, True])

    def test_min_depth_and_behind_camera(self):
        camera = _camera(tilt=np.pi / 2, translation=(0, 0, 0.4))
        result = pixels_to_ground([[0, 0]], camera, np.eye(3))
        self.assertFalse(result["valid"].item())
        result = project_ground_to_camera(np.zeros((1, 1, 2)), np.eye(4), camera,
                                          np.eye(3), (1, 1))
        self.assertFalse(result["valid"].item())
        camera = _camera(tilt=-np.pi / 2, translation=(0, 0, 2))
        result = project_ground_to_camera(np.zeros((1, 1, 2)), np.eye(4), camera,
                                          np.eye(3), (1, 1))
        self.assertLess(result["depth"].item(), 0)
        self.assertFalse(result["valid"].item())

    def test_nearest_integer_centres_and_image_boundaries(self):
        camera = _camera(tilt=np.pi / 2, translation=(0, 0, 2))
        pixels = np.array([[-0.49, 0], [0.49, 0], [0.5, 0], [1.49, 0], [1.5, 0],
                           [-0.51, 0], [0, 1.5]])
        inverse = pixels_to_ground(pixels, camera, np.eye(3))
        result = project_ground_to_camera(inverse["points_global"][:, :2][None],
                                          np.eye(4), camera, np.eye(3), (2, 2))
        np.testing.assert_array_equal(result["valid"][0],
                                      [True, True, True, True, False, False, False])
        np.testing.assert_array_equal(result["pixel_xy"][0, :4], [[0, 0], [0, 0], [1, 0], [1, 0]])
        np.testing.assert_allclose(result["uv"][0, :4], pixels[:4], atol=1e-14)

    def test_empty_inverse_pixels(self):
        result = pixels_to_ground(np.empty((0, 2)), self.camera, self.K)
        self.assertEqual(result["points_global"].shape, (0, 3))
        self.assertEqual(result["valid"].shape, (0,))

    def test_parameter_validation(self):
        grid = np.zeros((1, 1, 2))
        invalid_transforms = []
        for index, value in [((3, 0), 1), ((0, 0), 2), ((0, 0), -1), ((0, 3), np.nan)]:
            transform = np.eye(4)
            transform[index] = value
            invalid_transforms.append(transform)
        invalid_transforms.extend([np.eye(3), np.zeros((4, 4))])
        for transform in invalid_transforms:
            with self.subTest(transform=transform):
                with self.assertRaises(ValueError):
                    pixels_to_ground([[0, 0]], transform, self.K)
                with self.assertRaises(ValueError):
                    project_ground_to_camera(grid, transform, self.camera, self.K, (100, 100))
        tilted_bev = pose_matrix({"rotation": _axis_quaternion(0, 0.1), "translation": [0, 0, 0]})
        with self.assertRaises(ValueError):
            project_ground_to_camera(grid, tilted_bev, self.camera, self.K, (100, 100))
        for K in [np.zeros((3, 3)), np.eye(2), np.full((3, 3), np.nan)]:
            with self.subTest(K=K):
                with self.assertRaises(ValueError):
                    pixels_to_ground([[0, 0]], self.camera, K)
                with self.assertRaises(ValueError):
                    project_ground_to_camera(grid, np.eye(4), self.camera, K, (100, 100))
        for size in [(0, 1), (-1, 3), (1.5, 2), (2,), (np.inf, 2), (True, False)]:
            with self.subTest(size=size), self.assertRaises(ValueError):
                project_ground_to_camera(grid, np.eye(4), self.camera, self.K, size)
        for name in ("min_depth", "max_distance"):
            for value in (0, -1, np.inf, np.nan, True, [1]):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        pixels_to_ground([[0, 0]], self.camera, self.K, **{name: value})
                    with self.assertRaises(ValueError):
                        project_ground_to_camera(grid, np.eye(4), self.camera, self.K,
                                                 (100, 100), **{name: value})
        for pixels in [np.zeros((2, 3)), [1, 2], [[np.nan, 0]]]:
            with self.subTest(pixels=pixels), self.assertRaises(ValueError):
                pixels_to_ground(pixels, self.camera, self.K)
        for bad_grid in [np.zeros((2, 2)), np.zeros((0, 1, 2)), np.full((1, 1, 2), np.nan)]:
            with self.subTest(grid=bad_grid), self.assertRaises(ValueError):
                project_ground_to_camera(bad_grid, np.eye(4), self.camera, self.K, (100, 100))
        with self.assertRaises(ValueError):
            pixels_to_ground([[0, 0]], self.camera, self.K, plane_z=np.nan)


if __name__ == "__main__":
    unittest.main()

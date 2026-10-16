"""Synthetic exact occupancy projection regressions; no dataset or model needed.

Run in the existing vla-supervised environment:
    python -B -m unittest discover -s tests -p test_effocc_projection.py -v
"""
import json
import unittest

import numpy as np
import torch

from tasks.effocc.python.effocc_projection import raycast_occupancy


def pose_at(origin=(.5, .5, -1), rotation=None):
    result = np.eye(4)
    result[:3, 3] = origin
    if rotation is not None:
        result[:3, :3] = rotation
    return result


K_CENTER = np.array([[1., 0, .5], [0, 1., .5], [0, 0, 1.]])


def cast(grid, pose=None, intrinsic=None, **kwargs):
    options = dict(camera_to_grid=pose_at() if pose is None else pose,
                   intrinsic=K_CENTER if intrinsic is None else intrinsic,
                   image_size=(1, 1), point_cloud_range=np.r_[np.zeros(3), getattr(grid, "shape", (2, 2, 2))],
                   voxel_size=np.ones(3))
    options.update(kwargs)
    return raycast_occupancy(grid, **options)


def brute_force(grid, pose, intrinsic, image_size, bounds, spacing, min_depth):
    """Independent small-grid oracle: intersect every solid cell's AABB."""
    width, height = image_size
    labels = np.full((height, width), 255, np.uint8)
    depths = np.zeros((height, width), np.float32)
    indices = np.full((height, width), -1, np.int64)
    origin = pose[:3, 3]
    inverse = np.linalg.inv(intrinsic)
    for v in range(height):
        for u in range(width):
            direction = pose[:3, :3] @ inverse @ [u + .5, v + .5, 1]
            best = np.inf
            for xyz in np.ndindex(grid.shape):
                if grid[xyz] == 17:
                    continue
                lower = bounds[:3] + np.array(xyz) * spacing
                upper = lower + spacing
                near, far = min_depth, np.inf
                for axis in range(3):
                    if direction[axis] == 0:
                        if not lower[axis] <= origin[axis] < upper[axis]:
                            far = -np.inf
                            break
                    else:
                        a = (lower[axis] - origin[axis]) / direction[axis]
                        b = (upper[axis] - origin[axis]) / direction[axis]
                        near, far = max(near, min(a, b)), min(far, max(a, b))
                if far > near and near < best:
                    best = near
                    labels[v, u] = grid[xyz]
                    depths[v, u] = near
                    indices[v, u] = np.ravel_multi_index(xyz, grid.shape)
    return dict(labels=labels, depth=depths, voxel_index=indices)


class RaycastTests(unittest.TestCase):
    def setUp(self):
        self.grid = np.full((2, 2, 2), 17, np.uint8)

    def assert_hit(self, result, label, depth, index):
        self.assertEqual(result["labels"][0, 0], label)
        self.assertAlmostEqual(float(result["depth"][0, 0]), depth, places=6)
        self.assertEqual(result["voxel_index"][0, 0], index)

    def assert_empty(self, result):
        self.assertTrue(np.all(result["labels"] == 255))
        self.assertFalse(result["depth"].any())
        self.assertTrue(np.all(result["voxel_index"] == -1))

    def test_first_solid_occludes_later_solid(self):
        self.grid[0, 0, :] = [4, 8]
        self.assert_hit(cast(self.grid), 4, 1., 0)

    def test_free_transparent_and_others_zero_opaque(self):
        self.grid[0, 0, 1] = 0
        self.assert_hit(cast(self.grid), 0, 2., 1)
        self.grid[0, 0, 0] = 0
        self.assert_hit(cast(self.grid), 0, 1., 0)

    def test_empty_all_free_and_output_dtypes(self):
        result = cast(self.grid, image_size=(4, 3), chunk_size=5)
        self.assert_empty(result)
        for key, dtype in (("labels", np.uint8), ("depth", np.float32), ("voxel_index", np.int64)):
            self.assertEqual(result[key].dtype, dtype)
            self.assertEqual(result[key].shape, (3, 4))
        json.dumps(result["metadata"])

    def test_min_depth_crops_already_occupied_voxel_without_bias(self):
        self.grid[0, 0, :] = [4, 8]
        self.assert_hit(cast(self.grid, min_depth=1.25), 4, 1.25, 0)
        self.assert_hit(cast(self.grid, min_depth=2.), 8, 2., 1)
        self.assert_hit(cast(self.grid, min_depth=2.7), 8, 2.7, 1)
        self.assert_empty(cast(self.grid, min_depth=3.))
        self.assert_empty(cast(self.grid, min_depth=4.))

    def test_camera_inside_and_min_depth_point(self):
        self.grid[0, 0, :] = 4
        self.assert_hit(cast(self.grid, pose_at((.5, .5, .5))), 4, .1, 0)
        self.assert_hit(cast(self.grid, pose_at((.5, .5, .5)), min_depth=1.), 4, 1., 1)

    def test_camera_z_depth_is_not_euclidean_distance(self):
        self.grid[1, 0, 1] = 6
        intrinsic = np.array([[1., 0, 0], [0, 1, .5], [0, 0, 1.]])
        self.assert_hit(cast(self.grid, pose_at((0, .5, -1)), intrinsic), 6, 2., 5)

    def test_pixel_centers_and_xyz_flat_index(self):
        grid = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
        result = cast(grid, pose_at((0, 0, -1)), np.eye(3), image_size=(2, 2))
        np.testing.assert_array_equal(result["labels"], [[0, 4], [2, 6]])
        np.testing.assert_array_equal(result["voxel_index"], [[0, 4], [2, 6]])
        np.testing.assert_array_equal(result["depth"], np.ones((2, 2)))

    def test_translated_rotated_camera_and_nonuniform_voxels(self):
        # camera forward +Z becomes grid -Y; camera +X remains grid +X.
        rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        self.grid[1, 1, 1] = 9
        bounds = np.array([-4., -3, -2, 0, 3, 6])
        self.assert_hit(cast(self.grid, pose_at((-1, 6, 4), rotation),
                             point_cloud_range=bounds, voxel_size=np.array([2., 3., 4.])),
                        9, 3., 7)

    def test_negative_direction_and_grid_max_entry(self):
        rotation = np.diag([-1., 1., -1.])
        self.grid[0, 0, :] = [3, 5]
        self.assert_hit(cast(self.grid, pose_at((.5, .5, 3), rotation)), 5, 1., 1)
        self.assert_hit(cast(self.grid, pose_at((.5, .5, 3), rotation), min_depth=2.), 3, 2., 0)
        self.assert_hit(cast(self.grid, pose_at((.5, .5, 2.1), rotation)), 5, .1, 1)

    def test_parallel_rays_outside_or_on_maximum_face_miss(self):
        self.grid.fill(4)
        for origin in ((-.1, .5, -1), (2., .5, -1), (.5, 3, -1), (.5, .5, 3)):
            with self.subTest(origin=origin):
                self.assert_empty(cast(self.grid, pose_at(origin)))
        self.assert_hit(cast(self.grid, pose_at((0, .5, -1))), 4, 1., 0)

    def test_ray_only_touches_outer_corner(self):
        self.grid.fill(5)
        # x=t, z=t-2 reaches x=max and z=min simultaneously.
        intrinsic = np.array([[1., 0, -.5], [0, 1., .5], [0, 0, 1.]])
        self.assert_empty(cast(self.grid, pose_at((0, .5, -2)), intrinsic))

    def test_simultaneous_xyz_boundaries_skip_point_touch_cells(self):
        # Direction [1,1,1] passes only (0,0,0) and (1,1,1).
        self.grid.fill(5)
        self.grid[0, 0, 0] = 17
        self.grid[1, 1, 1] = 9
        intrinsic = np.array([[1., 0, -.5], [0, 1., -.5], [0, 0, 1.]])
        self.assert_hit(cast(self.grid, pose_at((-1, -1, -1)), intrinsic), 9, 2., 7)
        self.grid[1, 1, 1] = 17
        self.assert_empty(cast(self.grid, pose_at((-1, -1, -1)), intrinsic))

    def test_negative_simultaneous_boundaries(self):
        self.grid.fill(5)
        self.grid[1, 1, 1] = 17
        self.grid[0, 0, 0] = 9
        # Proper rotation mapping [1,-1,1] to [-1,-1,-1].
        rotation = np.diag([-1., 1., -1.])
        intrinsic = np.array([[1., 0, -.5], [0, 1., 1.5], [0, 0, 1.]])
        self.assert_hit(cast(self.grid, pose_at((3, 3, 3), rotation), intrinsic), 9, 2., 0)

    def test_uses_all_16_z_cells_in_original_effocc_range(self):
        grid = np.full((200, 200, 16), 17, np.uint8)
        grid[100, 100, 15] = 6
        self.assert_hit(cast(grid, pose_at((.2, .2, -2)),
                             point_cloud_range=np.array([-40., -40., -1., 40., 40., 5.4]),
                             voxel_size=np.full(3, .4)), 6, 7., (100 * 200 + 100) * 16 + 15)

    def test_custom_free_id_and_no_input_mutation(self):
        self.grid[0, 0, :] = [0, 17]
        pose, intrinsic = pose_at(), K_CENTER.copy()
        bounds, spacing = np.array([0., 0, 0, 2, 2, 2]), np.ones(3)
        arrays = [self.grid, pose, intrinsic, bounds, spacing]
        before = [array.copy() for array in arrays]
        result = cast(self.grid, pose, intrinsic, free_id=0, point_cloud_range=bounds, voxel_size=spacing)
        self.assert_hit(result, 17, 2., 1)
        for original, previous in zip(arrays, before):
            np.testing.assert_array_equal(original, previous)

    def test_noncontiguous_input_and_chunk_size_independence(self):
        grid = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)[:, ::-1]
        pose = pose_at((0, 0, -1))
        a = cast(grid, pose, np.eye(3), image_size=(5, 3), chunk_size=1)
        b = cast(grid.copy(), pose, np.eye(3), image_size=(5, 3), chunk_size=262144)
        for key in ("labels", "depth", "voxel_index"):
            np.testing.assert_array_equal(a[key], b[key])
        self.assertFalse(np.any(a["labels"] == 17))

    def test_random_camera_poses_match_independent_voxel_aabb_oracle(self):
        rng = np.random.default_rng(42)
        grid = rng.integers(0, 18, size=(3, 4, 2), dtype=np.uint8)
        grid[rng.random(grid.shape) < .55] = 17
        spacing = np.array([.7, 1.1, .4])
        bounds = np.r_[[-1.3, -2., -.8], np.array([-1.3, -2., -.8]) + np.array(grid.shape) * spacing]
        intrinsic = np.array([[3.1, .1, 3.5], [0, 2.6, 2.5], [0, 0, 1.]])
        center = (bounds[:3] + bounds[3:]) / 2
        for i in range(12):
            forward = rng.normal(size=3)
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, [0, 0, 1])
            right /= np.linalg.norm(right)
            down = np.cross(forward, right)
            pose = pose_at(center - forward * (0 if i % 3 == 0 else 4), np.stack([right, down, forward], axis=1))
            options = dict(image_size=(7, 5), point_cloud_range=bounds, voxel_size=spacing, min_depth=.3)
            expected = brute_force(grid, pose, intrinsic, (7, 5), bounds, spacing, .3)
            result = cast(grid, pose, intrinsic, chunk_size=8, **options)
            for key in ("labels", "voxel_index"):
                np.testing.assert_array_equal(result[key], expected[key], err_msg=f"pose {i}: {key}")
            np.testing.assert_allclose(result["depth"], expected["depth"], atol=1e-6, rtol=1e-6)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_matches_cpu_with_autocast_disabled_internally(self):
        rng = np.random.default_rng(14)
        grid = rng.integers(0, 18, size=(6, 5, 4), dtype=np.uint8)
        grid[rng.random(grid.shape) < .5] = 17
        pose = pose_at((3, 2.5, -1))
        intrinsic = np.array([[12., 0, 9], [0, 12., 7], [0, 0, 1.]])
        cpu = cast(grid, pose, intrinsic, image_size=(18, 14), chunk_size=67)
        with torch.autocast("cuda", dtype=torch.float16):
            cuda = cast(grid, pose, intrinsic, image_size=(18, 14), chunk_size=83, device="cuda")
        for key in ("labels", "voxel_index"):
            np.testing.assert_array_equal(cuda[key], cpu[key])
        np.testing.assert_allclose(cuda["depth"], cpu["depth"], atol=1e-5, rtol=1e-5)
        self.assertEqual(cuda["metadata"]["geometry_dtype"], "float32")
        self.assertFalse(np.any(cuda["labels"] == 17))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_tied_boundaries_and_maximum_entry(self):
        self.grid.fill(5)
        self.grid[0, 0, 0] = 17
        self.grid[1, 1, 1] = 9
        intrinsic = np.array([[1., 0, -.5], [0, 1., -.5], [0, 0, 1.]])
        self.assert_hit(cast(self.grid, pose_at((-1, -1, -1)), intrinsic, device="cuda"), 9, 2., 7)
        self.grid.fill(17)
        self.grid[0, 0, 1] = 4
        self.assert_hit(cast(self.grid, pose_at((.5, .5, 3), np.diag([-1., 1., -1.])), device="cuda"), 4, 1., 1)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.grid = np.full((2, 2, 2), 17, np.uint8)

    def test_invalid_semantics(self):
        for value in (self.grid.astype(np.int64), self.grid[:, :, 0], np.zeros((2, 0, 2), np.uint8),
                      np.full((2, 2, 2), 18, np.uint8), self.grid.tolist()):
            with self.subTest(value=type(value)), self.assertRaisesRegex(ValueError, "semantics"):
                cast(value)

    def test_invalid_geometry_and_options(self):
        invalid = [
            {"camera_to_grid": np.eye(3)},
            {"camera_to_grid": np.full((4, 4), np.nan)},
            {"camera_to_grid": np.diag([1., 1., 1., 2.])},
            {"camera_to_grid": np.diag([0., 1., 1., 1.])},
            {"intrinsic": np.zeros((3, 3))},
            {"intrinsic": np.full((3, 3), np.inf)},
            {"intrinsic": np.diag([1., 1., 2.])},
            {"point_cloud_range": np.array([0., 0, 0, 2, 2, 3])},
            {"point_cloud_range": np.array([2., 0, 0, 0, 2, 2])},
            {"voxel_size": np.array([1., 0, 1])},
            {"voxel_size": np.array([1., 1])},
            {"voxel_size": np.array([1., np.nan, 1])},
            {"image_size": (0, 2)}, {"image_size": (2., 2)}, {"image_size": (True, 2)},
            {"image_size": (1, 2, 3)},
            {"min_depth": -1}, {"min_depth": 0}, {"min_depth": np.nan}, {"min_depth": True},
            {"chunk_size": 0}, {"chunk_size": 1.5}, {"chunk_size": True},
            {"free_id": 18}, {"free_id": -1}, {"free_id": True},
            {"device": "meta"},
        ]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                cast(self.grid, **options)


if __name__ == "__main__":
    unittest.main()

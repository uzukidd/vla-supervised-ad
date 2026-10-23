"""Analytical camera-only tests / 仅使用解析相机模型的测试。"""
import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from tasks.nuscenes_sam3.python.grid_projection import project_ground_cells


def camera(name="CAM_FRONT", focal=9., centre=10., height=3.):
    transform = np.diag([1., -1., -1., 1.])
    transform[2, 3] = height
    return {"camera": name, "ego_pose": {"rotation": [1., 0., 0., 0.],
                                         "translation": [0., 0., 0.]},
            "camera_to_global": transform,
            "intrinsic": np.array([[focal, 0., centre], [0., focal, centre], [0., 0., 1.]])}


class GroundCellProjectionTests(unittest.TestCase):
    def project(self, cameras=None, masks=None, **kwargs):
        cameras = [camera()] if cameras is None else cameras
        masks = [np.ones((21, 21), np.uint8)] if masks is None else masks
        return project_ground_cells(cameras, masks, roi_size=kwargs.pop("roi_size", 3.),
                                    resolution=kwargs.pop("resolution", 3.), **kwargs)

    def test_known_nine_subcell_projections_and_majority_not_center(self):
        mask = np.zeros((21, 21), np.uint8)
        mask[np.ix_([7, 10, 13], [7, 10, 13])] = 1
        mask[10, 10] = 2
        result = self.project(masks=[mask])
        self.assertEqual(result["labels"].item(), 1)
        self.assertEqual(result["sample_count"].item(), 9)
        self.assertEqual(result["support_count"].item(), 9)
        self.assertEqual(result["camera_count"].item(), 1)
        self.assertTrue(result["conflict"].item())
        self.assertEqual(self.project(masks=[mask], samples_per_axis=1)["labels"].item(), 2)
        for key, dtype in (("labels", np.uint8), ("camera_count", np.uint8),
                           ("sample_count", np.uint32), ("support_count", np.uint32),
                           ("conflict", np.bool_)):
            self.assertEqual(result[key].dtype, dtype)
        json.dumps(result["metadata"], allow_nan=False)

    def test_even_sample_count_uses_subcell_centers(self):
        mask = np.zeros((21, 21), np.uint8)
        # S=2, cell side 4: sample positions +/-1, pixel positions 7 and 13.
        mask[np.ix_([7, 13], [7, 13])] = 2
        result = self.project(masks=[mask], roi_size=4., resolution=4., samples_per_axis=2)
        self.assertEqual(result["labels"].item(), 2)
        self.assertEqual(result["support_count"].item(), 4)

    def test_pixel_deduplication_is_per_cell_and_camera(self):
        cameras = [camera(focal=.01), camera("CAM_BACK", focal=.01)]
        masks = [np.ones((21, 21), np.uint8)] * 2
        result = self.project(cameras, masks, roi_size=2., resolution=1., samples_per_axis=8)
        np.testing.assert_array_equal(result["sample_count"], np.full((2, 2), 2))
        np.testing.assert_array_equal(result["support_count"], np.full((2, 2), 2))
        np.testing.assert_array_equal(result["camera_count"], np.full((2, 2), 2))
        # Both cameras share one source pixel across all four cells. Every cell
        # still receives one observation per camera, never 64 repeated votes.
        np.testing.assert_array_equal(result["labels"], np.ones((2, 2)))

    def test_six_camera_conflict_tie_and_camera_order_independence(self):
        names = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK",
                 "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
        cameras = [camera(name, focal=.01) for name in names]
        masks = [np.full((21, 21), label, np.uint8) for label in [2, 1, 2, 1, 2, 1]]
        result = self.project(cameras, masks)
        self.assertEqual(result["labels"].item(), 1)
        self.assertEqual(result["sample_count"].item(), 6)
        self.assertEqual(result["camera_count"].item(), 6)
        self.assertEqual(result["support_count"].item(), 6)
        self.assertTrue(result["conflict"].item())
        reversed_result = self.project(cameras[::-1], masks[::-1])
        for key in ("labels", "camera_count", "sample_count", "support_count", "conflict"):
            np.testing.assert_array_equal(result[key], reversed_result[key])
        self.assertEqual(reversed_result["metadata"]["reference_camera"], "CAM_FRONT")

    def test_majority_counts_samples_across_cameras(self):
        result = self.project([camera(), camera("CAM_BACK", focal=.01)],
                              [np.full((21, 21), 2, np.uint8), np.ones((21, 21), np.uint8)])
        self.assertEqual(result["labels"].item(), 2)
        self.assertEqual(result["support_count"].item(), 10)
        self.assertTrue(result["conflict"].item())

    def test_abstentions_and_configured_plane_ids(self):
        mask = np.zeros((21, 21), np.uint8)
        mask[np.ix_([7, 10, 13], [7, 10, 13])] = 5
        mask[10, 10] = 2
        default = self.project(masks=[mask])
        self.assertEqual(default["labels"].item(), 2)
        self.assertEqual(default["support_count"].item(), 1)
        self.assertFalse(default["conflict"].item())
        configured = self.project(masks=[mask], plane_ids=(5,))
        self.assertEqual(configured["labels"].item(), 5)
        self.assertEqual(configured["support_count"].item(), 8)
        self.assertEqual(configured["sample_count"].item(), 9)

    def test_no_semantic_or_geometric_evidence(self):
        for label in (0, 6):
            result = self.project(masks=[np.full((21, 21), label, np.uint8)])
            self.assertEqual(result["labels"].item(), 0)
            self.assertEqual(result["support_count"].item(), 0)
            self.assertEqual(result["sample_count"].item(), 9)
            self.assertEqual(result["camera_count"].item(), 1)
            self.assertFalse(result["conflict"].item())
        upward = camera()
        upward["camera_to_global"][:3, :3] = np.eye(3)
        for result in (self.project([upward]), self.project(max_distance=1.)):
            for key in ("labels", "camera_count", "sample_count", "support_count", "conflict"):
                self.assertFalse(result[key].any())

    def test_yaw_translation_and_global_zero_height(self):
        cam = camera(focal=6., centre=2.)
        cam["ego_pose"] = {"rotation": [np.sqrt(.5), 0., 0., np.sqrt(.5)],
                           "translation": [10., 20., 9.]}
        yaw = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        cam["camera_to_global"][:3, :3] = yaw @ cam["camera_to_global"][:3, :3]
        cam["camera_to_global"][:3, 3] = [10., 20., 3.]
        mask = np.zeros((5, 5), np.uint8)
        mask[1, 1], mask[1, 3], mask[3, 1], mask[3, 3] = 1, 2, 2, 1
        result = self.project([cam], [mask], roi_size=2., resolution=1., samples_per_axis=1)
        np.testing.assert_array_equal(result["labels"], [[1, 2], [2, 1]])
        np.testing.assert_allclose(result["global_from_bev"][:3, :3], yaw, atol=1e-14)
        np.testing.assert_array_equal(result["global_from_bev"][:3, 3], [10., 20., 0.])
        self.assertEqual(result["metadata"]["plane_z_global_m"], 0.)

    def test_horizon_behind_camera_and_distance(self):
        cam = camera(focal=1., centre=4., height=1.)
        cam["camera_to_global"][:3, :3] = [[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]]
        result = self.project([cam], [np.ones((9, 9), np.uint8)], roi_size=4., resolution=1., samples_per_axis=1)
        np.testing.assert_array_equal(result["camera_count"], np.tile([0, 0, 1, 1], (4, 1)))
        # The x=0 column is exactly at the camera horizon and cannot project.
        result = self.project([cam], [np.ones((9, 9), np.uint8)], roi_size=3., resolution=1., samples_per_axis=1)
        self.assertFalse(result["camera_count"][:, :2].any())

    def test_nonfront_reference_and_front_precedence(self):
        back = camera("CAM_BACK")
        back["ego_pose"]["translation"] = [100., 100., 20.]
        front = camera()
        mask = np.ones((21, 21), np.uint8)
        result = self.project([back, front], [mask, mask])
        np.testing.assert_array_equal(result["global_from_bev"][:3, 3], [0., 0., 0.])
        result = self.project([back], [mask])
        np.testing.assert_array_equal(result["global_from_bev"][:3, 3], [100., 100., 0.])
        self.assertEqual(result["metadata"]["reference_camera"], "CAM_BACK")

    def test_row_chunks_equal_single_chunk_and_inputs_unchanged(self):
        cameras = [camera()]
        masks = [np.indices((21, 21)).sum(axis=0).astype(np.uint8) % 3]
        before_cameras, before_masks = copy.deepcopy(cameras), [mask.copy() for mask in masks]
        expected = self.project(cameras, masks, roi_size=4., resolution=.5)
        with patch("tasks.nuscenes_sam3.python.grid_projection._MAX_CHUNK_SAMPLES", 1):
            actual = self.project(cameras, masks, roi_size=4., resolution=.5)
        for key in ("labels", "camera_count", "sample_count", "support_count", "conflict", "global_from_bev"):
            np.testing.assert_array_equal(actual[key], expected[key])
        self.assertEqual(cameras[0]["ego_pose"], before_cameras[0]["ego_pose"])
        for key in ("camera_to_global", "intrinsic"):
            np.testing.assert_array_equal(cameras[0][key], before_cameras[0][key])
        np.testing.assert_array_equal(masks[0], before_masks[0])

    def test_sparse_high_class_id(self):
        result = self.project(masks=[np.full((21, 21), 254, np.uint8)],
                              plane_ids=(254,), num_classes=255)
        self.assertEqual(result["labels"].item(), 254)

    def test_invalid_parameters(self):
        cases = [{"samples_per_axis": value} for value in (0, 9, True, 2., "3")]
        cases += [{"max_distance": value} for value in (0, -1, np.inf, np.nan, True)]
        cases += [{"num_classes": value} for value in (1, 256, True, 7.)]
        cases += [{"plane_ids": value} for value in ((), (0,), (7,), (1, 1), (True,), (1.,), None)]
        cases += [{"roi_size": value} for value in (0., -1., np.nan)]
        cases += [{"resolution": value} for value in (0., -1., np.inf, .7)]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                self.project(**kwargs)

    def test_invalid_camera_masks_and_calibration(self):
        mask = np.ones((21, 21), np.uint8)
        bad_cameras = [[], [camera(), camera()], [camera()]]
        bad_masks = [[], [mask, mask], []]
        for cameras, masks in zip(bad_cameras, bad_masks):
            with self.assertRaises(ValueError):
                self.project(cameras, masks)
        for invalid in (np.zeros((21, 21, 1), np.uint8), np.zeros((0, 1), np.uint8),
                        np.ones((21, 21), float), np.full((21, 21), -1),
                        np.full((21, 21), 7, np.uint8)):
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype), self.assertRaises(ValueError):
                self.project(masks=[invalid])
        changes = [("intrinsic", np.zeros((3, 3))), ("intrinsic", np.full((3, 3), np.nan)),
                   ("camera_to_global", np.zeros((4, 4))),
                   ("image_size_wh", (22, 21)), ("width", 22), ("height", 22),
                   ("camera", ""), ("ego_pose", {"rotation": [0, 0, 0, 0], "translation": [0, 0, 0]})]
        for key, value in changes:
            cam = camera()
            cam[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.project([cam], [mask])
        for key in ("ego_pose", "intrinsic", "camera_to_global"):
            cam = camera()
            del cam[key]
            with self.assertRaises(ValueError):
                self.project([cam], [mask])


if __name__ == "__main__":
    unittest.main()

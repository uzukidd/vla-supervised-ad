"""EFFOcc test-pipeline regressions; NumPy, Pillow and OpenCV are required.

Run with WSL vla-supervised Python:
    python -B -m unittest discover -s tests -p test_effocc_data.py -v
"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tasks.effocc.python.effocc_data import (
    CAMERAS, IMAGE_MEAN, IMAGE_STD, _points_to_depth_map, prepare_sample,
)
from shared.python.nuscenes_data import NuScenesMini


IDENTITY = [1, 0, 0, 0]
Z90 = [2 ** -0.5, 0, 0, 2 ** -0.5]


class DepthRasterizationTests(unittest.TestCase):
    def test_rounding_nearest_collisions_and_inclusive_exclusive_depth_bounds(self):
        projected = np.array([
            [1.49, 2.49, 8], [1.4, 2.4, 3],  # same rounded pixel: closest wins
            [2.5, 4.5, 1],                   # banker's rounding -> (2,4)
            [3.5, 5.5, 44.999],              # -> (4,6)
            [0, 0, .999], [0, 1, 45],
            [-.51, 3, 4], [703.51, 3, 4], [2, 255.51, 4],
            [-.49, 3, 2], [703.49, 255.49, 6],
            [np.nan, 0, 4], [0, np.inf, 4], [0, 0, np.nan],
        ], dtype=np.float32)
        depth = _points_to_depth_map(projected)
        self.assertEqual(depth.dtype, np.float32)
        self.assertEqual(depth.shape, (256, 704))
        self.assertEqual(np.count_nonzero(depth), 5)
        self.assertEqual(depth[2, 1], 3)
        self.assertEqual(depth[4, 2], 1)
        self.assertEqual(depth[6, 4], np.float32(44.999))
        self.assertEqual(depth[3, 0], 2)
        self.assertEqual(depth[255, 703], 6)

    def test_empty_and_all_invalid_clouds(self):
        self.assertFalse(_points_to_depth_map(np.empty((0, 3), np.float32)).any())
        self.assertFalse(_points_to_depth_map(np.array([[0, 0, -1]], np.float32)).any())


class PrepareSampleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "v1.0-mini").mkdir()
        (self.root / "media").mkdir()
        self.tables = {
            "scene": [{"token": "scene", "name": "scene-0061"}],
            "sample": [{"token": "sample", "scene_token": "scene", "timestamp": 2000000}],
            "sample_data": [], "sensor": [], "calibrated_sensor": [], "ego_pose": [],
            "lidarseg": [{"sample_data_token": "lidar", "filename": "media/labels.bin"}],
        }
        self.add_sensor("lidar", "LIDAR_TOP", "lidar", 2000000, "media/lidar.bin")
        self.points = np.array([[0, 0, 5, 71, 13], [2, 0, 5, 83, 14]], dtype=np.float32)
        self.points.tofile(self.root / "media/lidar.bin")
        np.array([1, 2], dtype=np.uint8).tofile(self.root / "media/labels.bin")
        # Fine pattern distinguishes PIL's default bicubic from bilinear.
        yy, xx = np.indices((900, 1600))
        self.rgb = np.stack(((xx * 31 + yy * 17) % 256,
                             (xx * 7 + yy * 3) % 256,
                             (xx * 11 + yy * 29) % 256), axis=-1).astype(np.uint8)
        Image.fromarray(self.rgb).save(self.root / "media/camera.png")
        for index, name in enumerate(CAMERAS):
            self.add_sensor(name, name, name, 2000000 + index * 1000, "media/camera.png")

    def add_sensor(self, token, channel, calibration, timestamp, filename,
                   translation=(0, 0, 0), rotation=IDENTITY, ego_translation=(0, 0, 0),
                   ego_rotation=IDENTITY, prev="", keyframe=True):
        sensor = f"{calibration}-sensor"
        if not any(record["token"] == sensor for record in self.tables["sensor"]):
            self.tables["sensor"].append({"token": sensor, "channel": channel})
            self.tables["calibrated_sensor"].append({
                "token": calibration, "sensor_token": sensor,
                "rotation": rotation, "translation": list(translation),
                "camera_intrinsic": [[100, 0, 800], [0, 100, 600], [0, 0, 1]],
            })
        self.tables["ego_pose"].append({
            "token": f"{token}-ego", "rotation": ego_rotation,
            "translation": list(ego_translation), "timestamp": timestamp,
        })
        self.tables["sample_data"].append({
            "token": token, "sample_token": "sample", "is_key_frame": keyframe,
            "calibrated_sensor_token": calibration, "ego_pose_token": f"{token}-ego",
            "timestamp": timestamp, "filename": filename, "prev": prev,
        })

    def record(self, table, token):
        return next(record for record in self.tables[table] if record["token"] == token)

    def load(self):
        for name, records in self.tables.items():
            (self.root / "v1.0-mini" / f"{name}.json").write_text(json.dumps(records))
        dataset = NuScenesMini(self.root)
        return dataset, dataset.select_views(list(CAMERAS))

    def prepare(self, sweeps_num=0):
        dataset, views = self.load()
        return prepare_sample(dataset, views, sweeps_num)

    def test_images_default_pil_resize_bottom_crop_bgr_normalization_and_post_transform(self):
        result = self.prepare()
        expected_pixels = np.asarray(Image.fromarray(self.rgb).resize((704, 396)).crop((0, 140, 704, 396)))
        expected = ((expected_pixels[..., ::-1].astype(np.float32) - IMAGE_MEAN).astype(np.float64)
                    * (1 / IMAGE_STD.astype(np.float64))).astype(np.float32)
        for image in result["images"]:
            np.testing.assert_array_equal(image, expected.transpose(2, 0, 1))
        np.testing.assert_allclose(result["post_rots"], np.tile(np.diag([.44, .44, 1]), (6, 1, 1)))
        np.testing.assert_array_equal(result["post_trans"], np.tile([0, -140, 0], (6, 1)))
        self.assertEqual(result["info"]["image_augmentation"][0]["resize_dims_wh"], [704, 396])
        self.assertEqual(result["info"]["image_augmentation"][0]["crop_xyxy"], [0, 140, 704, 396])
        self.assertEqual(result["info"]["normalization"]["color_order"], "BGR")
        for key in ("images", "depth_maps", "camera2keyego", "intrinsics", "post_rots", "post_trans", "points"):
            self.assertEqual(result[key].dtype, np.float32, key)
            self.assertTrue(result[key].flags.c_contiguous, key)
        json.dumps(result["info"])

    def test_six_views_are_sorted_and_must_be_one_complete_canonical_sample(self):
        dataset, views = self.load()
        result = prepare_sample(dataset, views[::-1], 0)
        self.assertEqual(result["info"]["camera_names"], list(CAMERAS))
        for invalid in (views[:5], views[:5] + [views[0]],
                        [dict(view, sample_token="other") if index == 1 else view
                         for index, view in enumerate(views)],
                        [dict(view, lidar_token="other") if index == 1 else view
                         for index, view in enumerate(views)]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                prepare_sample(dataset, invalid, 0)
        wrong = [dict(view) for view in views]
        wrong[0]["camera_token"] = wrong[1]["camera_token"]
        with self.assertRaises(ValueError):
            prepare_sample(dataset, wrong, 0)
        wrong = [dict(view) for view in views]
        wrong[0]["image_path"] = self.root / "other.png"
        with self.assertRaisesRegex(ValueError, "image_path"):
            prepare_sample(dataset, wrong, 0)
        for count in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                prepare_sample(dataset, views, count)

    def test_empty_history_pads_nine_copies_after_square_close_removal(self):
        result = self.prepare(9)
        self.assertEqual(result["points"].shape, (11, 5))
        np.testing.assert_array_equal(result["points"][:2, :4], self.points[:, :4])
        np.testing.assert_array_equal(result["points"][2:, :4], np.tile(self.points[1, :4], (9, 1)))
        np.testing.assert_array_equal(result["points"][:, 4], 0)
        self.assertEqual(result["info"]["used_sweep_count"], 0)
        self.assertEqual(result["info"]["padded_sweep_count"], 9)

    def test_sweep_alignment_lag_intensity_and_no_padding_for_short_nonempty_history(self):
        self.record("sample_data", "lidar")["prev"] = "sweep"
        self.record("calibrated_sensor", "lidar")["translation"] = [1, 0, 0]
        self.record("ego_pose", "lidar-ego")["translation"] = [10, 0, 0]
        self.record("ego_pose", "CAM_FRONT-ego")["translation"] = [10, 0, 0]
        self.add_sensor("sweep", "LIDAR_TOP", "sweep-cal", 1750000, "media/sweep.bin",
                        translation=[1, 0, 0], ego_translation=[8, 0, 0],
                        ego_rotation=Z90, keyframe=False)
        sweep = np.array([[2, 0, 5, 17, 99], [.5, .5, 5, 19, 99],
                          [1, .5, 5, 23, 99], [.9, -.99, 5, 29, 99]], np.float32)
        sweep.tofile(self.root / "media/sweep.bin")
        result = self.prepare(9)
        expected = np.array([[1, 0, 5, 71, 0], [3, 0, 5, 83, 0],
                             [-2, 3, 5, 17, .25], [-2.5, 2, 5, 23, .25]], np.float32)
        np.testing.assert_allclose(result["points"], expected, atol=2e-6)
        self.assertEqual(result["info"]["used_sweep_count"], 1)
        self.assertEqual(result["info"]["padded_sweep_count"], 0)
        self.assertEqual(result["info"]["sweep_timestamps"], [1750000])

    def test_selects_nearest_nine_sweeps_in_prev_order(self):
        self.record("sample_data", "lidar")["prev"] = "sweep-0"
        for index in range(11):
            self.add_sensor(f"sweep-{index}", "LIDAR_TOP", "sweep-cal", 1900000 - index * 50000,
                            f"media/sweep-{index}.bin", keyframe=False,
                            prev=f"sweep-{index + 1}" if index < 10 else "")
            np.array([[2, 0, 5, index, 88]], np.float32).tofile(self.root / f"media/sweep-{index}.bin")
        result = self.prepare(9)
        self.assertEqual(result["info"]["sweep_tokens"], [f"sweep-{index}" for index in range(9)])
        np.testing.assert_array_equal(result["points"][2:, 3], np.arange(9))
        np.testing.assert_allclose(result["points"][2:, 4], .1 + np.arange(9) * .05)

    def test_camera_keyego_is_first_left_camera_but_toego_is_front_camera(self):
        self.record("ego_pose", "CAM_FRONT_LEFT-ego")["translation"] = [10, 0, 0]
        self.record("ego_pose", "CAM_FRONT-ego")["translation"] = [12, 0, 0]
        self.record("ego_pose", "CAM_FRONT_RIGHT-ego")["translation"] = [13, 0, 0]
        self.record("calibrated_sensor", "CAM_FRONT_RIGHT")["translation"] = [2, 0, 0]
        result = self.prepare()
        np.testing.assert_allclose(result["camera2keyego"][:3, :3, 3], [[0, 0, 0], [2, 0, 0], [5, 0, 0]])
        np.testing.assert_allclose(result["points"][:, 0], self.points[:, 0] - 12)
        self.assertEqual(result["info"]["points_reference_camera"], "CAM_FRONT")
        self.assertEqual(result["info"]["camera_keyego_reference"], "CAM_FRONT_LEFT")

    def test_fusion_depth_uses_all_points_inverse_calibration_and_ignores_other_camera_pose(self):
        self.record("ego_pose", "CAM_FRONT_RIGHT-ego")["translation"] = [100, 0, 0]
        self.record("calibrated_sensor", "CAM_FRONT_RIGHT")["translation"] = [1, 0, 0]
        # Key and sweep collide at (352,124); sweep's depth 2 must win.
        self.points = np.array([[0, 0, 5, 71, 13]], np.float32)
        self.points.tofile(self.root / "media/lidar.bin")
        np.array([1], np.uint8).tofile(self.root / "media/labels.bin")
        self.record("sample_data", "lidar")["prev"] = "sweep"
        self.add_sensor("sweep", "LIDAR_TOP", "sweep-cal", 1900000, "media/sweep.bin",
                        translation=[-2, 0, 0], keyframe=False)
        np.array([[2, 0, 2, 17, 99]], np.float32).tofile(self.root / "media/sweep.bin")
        result = self.prepare(9)
        self.assertEqual(result["depth_maps"][0, 124, 352], 2)
        self.assertEqual(result["depth_maps"][2, 124, 330], 2)
        self.assertEqual(result["depth_maps"][2, 124, 343], 5)
        self.assertEqual(result["camera2keyego"][2, 0, 3], 101)

    def test_malformed_sweep_file_and_cyclic_or_non_decreasing_history_fail(self):
        self.record("sample_data", "lidar")["prev"] = "sweep"
        self.add_sensor("sweep", "LIDAR_TOP", "sweep-cal", 1900000, "media/sweep.bin",
                        prev="lidar", keyframe=False)
        dataset, views = self.load()
        with self.assertRaisesRegex(ValueError, "cycle"):
            prepare_sample(dataset, views)
        self.record("sample_data", "sweep")["prev"] = ""
        self.record("sample_data", "sweep")["timestamp"] = 2000000
        dataset, views = self.load()
        with self.assertRaisesRegex(ValueError, "timestamps"):
            prepare_sample(dataset, views)
        self.record("sample_data", "sweep")["timestamp"] = 1900000
        (self.root / "media/sweep.bin").write_bytes(b"broken")
        dataset, views = self.load()
        with self.assertRaisesRegex(ValueError, "float32 Nx5"):
            prepare_sample(dataset, views)


if __name__ == "__main__":
    unittest.main()

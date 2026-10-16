"""Synthetic tests; run with the project's WSL vla-supervised Python.

    python -B -m unittest discover -s tests -p test_nuscenes_data.py -v
"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from shared.python.nuscenes_data import (
    CAMERAS, COARSE_CLASS_NAMES, RAW_CLASS_NAMES, RAW_TO_COARSE,
    NuScenesMini, _quaternion_matrix, _transform,
)


_IDENTITY = [1, 0, 0, 0]
_Z90 = [2 ** -0.5, 0, 0, 2 ** -0.5]
_MINI_TRAIN = [
    "scene-0061", "scene-0553", "scene-0655", "scene-0757",
    "scene-0796", "scene-1077", "scene-1094", "scene-1100",
]
_MINI_VAL = ["scene-0103", "scene-0916"]


class TransformTests(unittest.TestCase):
    def test_wxyz_quaternion_known_axes_and_normalization(self):
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        np.testing.assert_allclose(_quaternion_matrix(_Z90), expected, atol=1e-15)
        np.testing.assert_allclose(_quaternion_matrix(np.array(_Z90) * 3), expected, atol=1e-15)
        np.testing.assert_allclose(_quaternion_matrix([0, 1, 0, 0]), np.diag([1, -1, -1]))
        np.testing.assert_allclose(_quaternion_matrix(_IDENTITY), np.eye(3))

    def test_forward_and_inverse_have_known_coordinates(self):
        points = np.array([[2., 0., 3.], [0., 1., 4.]])
        pose = {"rotation": _Z90, "translation": [10, 20, 30]}
        forward = _transform(points, pose)
        np.testing.assert_allclose(forward, [[10, 22, 33], [9, 20, 34]], atol=1e-14)
        np.testing.assert_allclose(_transform(forward, pose, inverse=True), points, atol=1e-14)
        np.testing.assert_array_equal(points, [[2, 0, 3], [0, 1, 4]])

    def test_invalid_quaternion_fails(self):
        for rotation in ([0, 0, 0, 0], [1, 0, 0], [np.nan, 0, 0, 0]):
            with self.subTest(rotation=rotation), self.assertRaisesRegex(ValueError, "quaternion"):
                _quaternion_matrix(rotation)


class NuScenesDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "v1.0-mini").mkdir()
        (self.root / "samples").mkdir()
        (self.root / "lidarseg").mkdir()
        self.point_path = self.root / "samples" / "lidar.bin"
        self.label_path = self.root / "lidarseg" / "labels.bin"
        self.image_path = self.root / "samples" / "camera.png"
        Image.new("RGB", (10, 8)).save(self.image_path)
        self.tables = {
            "scene": [{"token": "scene", "name": "scene-0061"}],
            # Like the raw dataset, this table has no devkit-added 'data' field.
            "sample": [{"token": "sample", "scene_token": "scene", "timestamp": 100}],
            "sensor": [
                {"token": "lidar-sensor", "channel": "LIDAR_TOP", "modality": "lidar"},
                {"token": "front-sensor", "channel": "CAM_FRONT", "modality": "camera"},
                {"token": "left-sensor", "channel": "CAM_FRONT_LEFT", "modality": "camera"},
            ],
            "calibrated_sensor": [
                {"token": "lidar-cal", "sensor_token": "lidar-sensor", "rotation": _IDENTITY,
                 "translation": [0, 0, 0], "camera_intrinsic": []},
                {"token": "front-cal", "sensor_token": "front-sensor", "rotation": _IDENTITY,
                 "translation": [0, 0, 0], "camera_intrinsic": [[2, 0, 5], [0, 2, 4], [0, 0, 1]]},
                {"token": "left-cal", "sensor_token": "left-sensor", "rotation": _IDENTITY,
                 "translation": [0, 0, 0], "camera_intrinsic": [[2, 0, 5], [0, 2, 4], [0, 0, 1]]},
            ],
            "ego_pose": [
                {"token": "lidar-ego", "rotation": _IDENTITY, "translation": [0, 0, 0], "timestamp": 100},
                {"token": "camera-ego", "rotation": _IDENTITY, "translation": [0, 0, 0], "timestamp": 120},
            ],
            "sample_data": [],
            "lidarseg": [{"token": "different-label-token", "sample_data_token": "lidar",
                          "filename": "lidarseg/labels.bin"}],
        }
        self.add_sample_data("sample", "lidar", "front", "left")
        self.write_points([[0, 0, 2], [1, 1, 2]], [17, 26])

    def add_sample_data(self, sample, lidar, front, left):
        for token, calibration, ego, filename, timestamp in (
            (lidar, "lidar-cal", "lidar-ego", "samples/lidar.bin", 100),
            (front, "front-cal", "camera-ego", "samples/camera.png", 120),
            (left, "left-cal", "camera-ego", "samples/camera.png", 125),
        ):
            self.tables["sample_data"].append({
                "token": token, "sample_token": sample, "calibrated_sensor_token": calibration,
                "ego_pose_token": ego, "is_key_frame": True, "timestamp": timestamp,
                "filename": filename,
                # Size comes from the image, not metadata and never a resize.
                "width": 100, "height": 80,
            })

    def write_points(self, xyz, labels=None):
        xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
        points = np.column_stack((xyz, np.zeros((len(xyz), 2), dtype=np.float32)))
        points.astype(np.float32).tofile(self.point_path)
        if labels is None:
            labels = np.arange(len(xyz), dtype=np.uint8) % 32
        np.asarray(labels, dtype=np.uint8).tofile(self.label_path)

    def write_uv_depth(self, entries, labels=None):
        self.write_points([[(u - 5) * z / 2, (v - 4) * z / 2, z] for u, v, z in entries], labels)

    def load(self):
        for table, records in self.tables.items():
            (self.root / "v1.0-mini" / f"{table}.json").write_text(json.dumps(records), encoding="utf-8")
        return NuScenesMini(self.root)

    def project(self, min_depth=1.0):
        nusc = self.load()
        view = nusc.select_views(["CAM_FRONT"])[0]
        return nusc.projection(view, min_depth=min_depth)

    def test_official_taxonomy_and_merge(self):
        self.assertIsInstance(CAMERAS, tuple)
        self.assertEqual(set(CAMERAS), {
            "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
            "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
        })
        self.assertEqual(set(RAW_CLASS_NAMES), set(range(32)))
        self.assertEqual(set(COARSE_CLASS_NAMES), set(range(17)))
        self.assertEqual(RAW_TO_COARSE.shape, (32,))
        self.assertEqual(RAW_TO_COARSE.dtype, np.uint8)
        expected_groups = {
            "ignore": [0, 1, 5, 7, 8, 10, 11, 13, 19, 20, 29, 31],
            "barrier": [9], "bicycle": [14], "bus": [15, 16], "car": [17],
            "construction_vehicle": [18], "motorcycle": [21], "pedestrian": [2, 3, 4, 6],
            "traffic_cone": [12], "trailer": [22], "truck": [23],
            "driveable_surface": [24], "other_flat": [25], "sidewalk": [26],
            "terrain": [27], "manmade": [28], "vegetation": [30],
        }
        for name, raw_indices in expected_groups.items():
            coarse_id = next(index for index, value in COARSE_CLASS_NAMES.items() if value == name)
            np.testing.assert_array_equal(np.flatnonzero(RAW_TO_COARSE == coarse_id), raw_indices)
        self.assertEqual(RAW_CLASS_NAMES[25], "flat.other")
        self.assertEqual(RAW_CLASS_NAMES[26], "flat.sidewalk")
        self.assertEqual(RAW_CLASS_NAMES[29], "static.other")
        self.assertEqual(RAW_CLASS_NAMES[30], "static.vegetation")

    def test_projection_coordinates_labels_types_and_native_image_size(self):
        result = self.project()
        np.testing.assert_array_equal(result["point_indices"], [0, 1])
        np.testing.assert_allclose(result["uv"], [[5, 4], [6, 5]])
        np.testing.assert_array_equal(result["pixels"], [[5, 4], [6, 5]])
        np.testing.assert_array_equal(result["depth"], [2, 2])
        np.testing.assert_array_equal(result["raw_labels"], [17, 26])
        self.assertEqual(result["point_indices"].dtype, np.int64)
        self.assertEqual(result["pixels"].dtype, np.int64)
        self.assertEqual(result["raw_labels"].dtype, np.uint8)
        self.assertEqual(result["uv"].dtype, np.float64)
        self.assertEqual(result["depth"].dtype, np.float64)
        self.assertEqual(result["image_size_wh"], [10, 8])
        self.assertEqual(result["counts"], {
            "total_points": 2, "in_front_points": 2, "in_image_points": 2,
            "retained_points": 2, "same_pixel_discarded": 0,
        })

    def test_full_transform_chain_uses_distinct_ego_timestamps(self):
        self.write_points([[2, 0, 10]], [30])
        lidar_cal, camera_cal = self.tables["calibrated_sensor"][:2]
        lidar_cal.update(rotation=_Z90, translation=[1, 0, 0])
        camera_cal.update(rotation=_Z90, translation=[1, 1, 0])
        lidar_ego, camera_ego = self.tables["ego_pose"]
        lidar_ego.update(rotation=_Z90, translation=[10, 20, 0])
        camera_ego.update(translation=[6, 20, 0])
        # Manual coordinates: lidar [2,0,10] -> [1,2,10] -> [8,21,10]
        # -> camera ego [2,1,10] -> camera [0,-1,10].
        result = self.project()
        np.testing.assert_allclose(result["uv"], [[5, 3.8]], atol=1e-14)
        np.testing.assert_allclose(result["depth"], [10])
        np.testing.assert_array_equal(result["raw_labels"], [30])
        # Moving only camera ego changes the projected position.
        camera_ego["translation"] = [7, 20, 0]
        moved = self.project()
        np.testing.assert_allclose(moved["uv"], [[5, 4]], atol=1e-14)

    def test_front_depth_and_strict_margin_filters(self):
        self.write_uv_depth([
            (5, 4, -2), (5, 4, 0), (5, 4, 1),  # behind/on threshold
            (1, 4, 2), (9, 4, 2), (5, 1, 2), (5, 7, 2),  # exact boundaries
            (0.5, 4, 2), (9.5, 4, 2), (5, 0.5, 2), (5, 7.5, 2),
            (1.25, 1.25, 2), (8.75, 6.75, 2),
        ])
        result = self.project()
        np.testing.assert_array_equal(result["point_indices"], [11, 12])
        np.testing.assert_array_equal(result["pixels"], [[1, 1], [8, 6]])
        self.assertEqual(result["counts"], {
            "total_points": 13, "in_front_points": 10, "in_image_points": 2,
            "retained_points": 2, "same_pixel_discarded": 0,
        })

    def test_nearest_depth_exact_ties_original_index_and_floor_pixels(self):
        self.write_uv_depth([
            (4.5, 3.5, 8), (4.25, 3.25, 2), (4.75, 3.75, 2),
            (5.1, 3.2, 3), (4.1, 4.2, 1.5),
        ], [10, 17, 26, 30, 0])
        result = self.project()
        np.testing.assert_array_equal(result["point_indices"], [1, 3, 4])
        np.testing.assert_array_equal(result["pixels"], [[4, 3], [5, 3], [4, 4]])
        np.testing.assert_allclose(result["uv"], [[4.25, 3.25], [5.1, 3.2], [4.1, 4.2]], atol=1e-6)
        np.testing.assert_allclose(result["depth"], [2, 3, 1.5])
        np.testing.assert_array_equal(result["raw_labels"], [17, 30, 0])
        self.assertEqual(result["counts"]["in_image_points"], 5)
        self.assertEqual(result["counts"]["retained_points"], 3)
        self.assertEqual(result["counts"]["same_pixel_discarded"], 2)

    def test_empty_points_and_all_filtered_have_consistent_shapes(self):
        for points in ([], [[0, 0, -1]], [[100, 100, 2]]):
            with self.subTest(points=points):
                self.write_points(points)
                result = self.project()
                for field in ("uv", "pixels"):
                    self.assertEqual(result[field].shape, (0, 2))
                for field in ("point_indices", "depth", "raw_labels"):
                    self.assertEqual(result[field].shape, (0,))
                self.assertEqual(result["counts"]["retained_points"], 0)

    def test_label_point_mismatch_and_invalid_label_even_if_filtered(self):
        for labels in ([1], [1, 2, 3]):
            with self.subTest(labels=labels):
                np.asarray(labels, dtype=np.uint8).tofile(self.label_path)
                with self.assertRaisesRegex(ValueError, "label/point count mismatch"):
                    self.project()
        self.write_points([[0, 0, -2], [0, 0, 2]], [32, 17])
        with self.assertRaisesRegex(ValueError, "0..31"):
            self.project()

    def test_malformed_point_file_and_nonfinite_coordinates(self):
        self.point_path.write_bytes(b"123")
        with self.assertRaisesRegex(ValueError, "float32 Nx5"):
            self.project()
        self.write_points([[np.nan, 0, 2]])
        with self.assertRaisesRegex(ValueError, "coordinates must be finite"):
            self.project()

    def test_custom_min_depth_and_invalid_values(self):
        self.write_uv_depth([(5, 4, 2), (6, 5, 3)])
        result = self.project(min_depth=2)
        np.testing.assert_array_equal(result["point_indices"], [1])
        for depth in (-1, np.nan, np.inf):
            with self.subTest(depth=depth), self.assertRaisesRegex(ValueError, "min_depth"):
                self.project(min_depth=depth)

    def test_keyframe_selection_and_stable_scene_timestamp_camera_order(self):
        self.tables["scene"].append({"token": "scene-b", "name": "scene-0103"})
        self.tables["sample"].extend([
            {"token": "early", "scene_token": "scene", "timestamp": 50},
            {"token": "other-scene", "scene_token": "scene-b", "timestamp": 1},
        ])
        for sample in ("early", "other-scene"):
            self.add_sample_data(sample, sample + "-lidar", sample + "-front", sample + "-left")
            self.tables["lidarseg"].append({"token": sample + "-label", "sample_data_token": sample + "-lidar",
                                           "filename": "lidarseg/labels.bin"})
        sweep = dict(self.tables["sample_data"][1])
        sweep.update(token="sweep", is_key_frame=False, timestamp=0, filename="missing-sweep.png")
        self.tables["sample_data"].append(sweep)
        # JSON ordering and sensor timestamps must not affect selected ordering.
        self.tables["sample_data"].reverse()
        self.tables["sample"].reverse()
        self.tables["scene"].reverse()
        nusc = self.load()
        views = nusc.select_views(["CAM_FRONT_LEFT", "CAM_FRONT"])
        self.assertEqual([(v["sample_token"], v["camera"]) for v in views], [
            ("early", "CAM_FRONT_LEFT"), ("early", "CAM_FRONT"),
            ("sample", "CAM_FRONT_LEFT"), ("sample", "CAM_FRONT"),
            ("other-scene", "CAM_FRONT_LEFT"), ("other-scene", "CAM_FRONT"),
        ])
        self.assertEqual(set(views[0]), {
            "sample_token", "scene_name", "camera", "camera_token", "lidar_token", "image_path",
        })
        self.assertIsInstance(views[0]["image_path"], Path)
        self.assertNotIn("sweep", [v["camera_token"] for v in views])

    def test_official_mini_splits_scene_filter_and_reject_trainval(self):
        self.tables["scene"] = []
        self.tables["sample"] = []
        self.tables["sample_data"] = []
        self.tables["lidarseg"] = []
        # scene-0001 is a trainval scene and must not leak into mini_train.
        for i, name in enumerate(reversed(_MINI_TRAIN + _MINI_VAL + ["scene-0001"])):
            token = f"sample-{i}"
            self.tables["scene"].append({"token": name, "name": name})
            self.tables["sample"].append({"token": token, "scene_token": name, "timestamp": 100})
            self.add_sample_data(token, token + "-lidar", token + "-front", token + "-left")
            self.tables["lidarseg"].append({"sample_data_token": token + "-lidar", "filename": "lidarseg/labels.bin"})
        nusc = self.load()
        for split, expected in (("mini_train", _MINI_TRAIN), ("mini_val", _MINI_VAL)):
            views = nusc.select_views(["CAM_FRONT"], split=split)
            self.assertEqual([view["scene_name"] for view in views], sorted(expected))
        selected = nusc.select_views(["CAM_FRONT"], split="mini_train", scenes=["scene-0103", "scene-1100"])
        self.assertEqual([view["scene_name"] for view in selected], ["scene-1100"])
        self.assertEqual(nusc.select_views(["CAM_FRONT"], scenes=[]), [])
        for split in ("train", "val", "test", "trainval"):
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "unsupported split"):
                nusc.select_views(["CAM_FRONT"], split=split)
        with self.assertRaisesRegex(ValueError, "unknown scenes"):
            nusc.select_views(["CAM_FRONT"], scenes=["scene-9999"])

    def test_missing_camera_lidar_and_label_records_raise_explicit_errors(self):
        records = list(self.tables["sample_data"])
        for token, expected in (("front", "missing CAM_FRONT keyframe"), ("lidar", "missing LIDAR_TOP keyframe")):
            with self.subTest(token=token):
                self.tables["sample_data"] = [record for record in records if record["token"] != token]
                nusc = self.load()
                with self.assertRaisesRegex(ValueError, expected):
                    nusc.select_views(["CAM_FRONT"])
        self.tables["sample_data"] = records
        self.tables["lidarseg"] = []
        with self.assertRaisesRegex(ValueError, "missing lidarseg record for token lidar"):
            self.load().select_views(["CAM_FRONT"])

    def test_missing_media_and_unknown_or_repeated_cameras(self):
        nusc = self.load()
        with self.assertRaisesRegex(ValueError, "unknown camera"):
            nusc.select_views(["CAM_UNKNOWN"])
        with self.assertRaisesRegex(ValueError, "must not be repeated"):
            nusc.select_views(["CAM_FRONT", "CAM_FRONT"])
        self.assertEqual(nusc.select_views([]), [])
        # Modifying record paths exercises missing files without deleting fixtures.
        self.tables["lidarseg"][0]["filename"] = "lidarseg/missing.bin"
        with self.assertRaisesRegex(FileNotFoundError, "missing lidarseg labels"):
            self.load().select_views(["CAM_FRONT"])
        self.tables["lidarseg"][0]["filename"] = "lidarseg/labels.bin"
        self.tables["sample_data"][1]["filename"] = "samples/missing-camera.png"
        with self.assertRaisesRegex(FileNotFoundError, "missing CAM_FRONT image"):
            self.load().select_views(["CAM_FRONT"])


if __name__ == "__main__":
    unittest.main()

"""Synthetic nuImages tests; run with unittest discovery in vla-supervised."""
import base64
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

from tasks.nuimages.python.nuimages_data import NuImagesData


EXPECTED_NAMES = {
    0: "unlabeled", 1: "animal", 2: "human.pedestrian.adult",
    3: "human.pedestrian.child", 4: "human.pedestrian.construction_worker",
    5: "human.pedestrian.personal_mobility", 6: "human.pedestrian.police_officer",
    7: "human.pedestrian.stroller", 8: "human.pedestrian.wheelchair",
    9: "movable_object.barrier", 10: "movable_object.debris",
    11: "movable_object.pushable_pullable", 12: "movable_object.trafficcone",
    13: "static_object.bicycle_rack", 14: "vehicle.bicycle",
    15: "vehicle.bus.bendy", 16: "vehicle.bus.rigid", 17: "vehicle.car",
    18: "vehicle.construction", 19: "vehicle.emergency.ambulance",
    20: "vehicle.emergency.police", 21: "vehicle.motorcycle",
    22: "vehicle.trailer", 23: "vehicle.truck", 24: "flat.driveable_surface",
    25: "unused_25", 26: "unused_26", 27: "unused_27", 28: "unused_28",
    29: "unused_29", 30: "unused_30", 31: "vehicle.ego",
}


def encode_mask(array):
    encoded = coco_mask.encode(np.asfortranarray(array, dtype=np.uint8))
    return {"size": encoded["size"], "counts": base64.b64encode(encoded["counts"]).decode("ascii")}


class NuImagesDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.version = "v1.0-mini"
        (self.root / self.version).mkdir()
        (self.root / "samples").mkdir()
        (self.root / "sweeps").mkdir()
        for name in ("key-a", "key-b", "other-camera"):
            Image.new("RGB", (4, 3)).save(self.root / "samples" / f"{name}.png")
        Image.new("RGB", (4, 3)).save(self.root / "sweeps" / "sweep.png")
        self.tables = {
            "category": [
                {"token": f"category-{i}", "name": name}
                for i, name in reversed(list(EXPECTED_NAMES.items()))
                if i in set(range(1, 25)) | {31}
            ],
            "sample": [
                {"token": "sample-b", "key_camera_token": "key-b", "timestamp": 20},
                {"token": "sample-a", "key_camera_token": "key-a", "timestamp": 10},
            ],
            "sample_data": [
                self.image_record("key-b", "sample-b", "cal-back", 20),
                self.image_record("key-a", "sample-a", "cal-front", 10),
                self.image_record("other-camera", "sample-a", "cal-back", 10),
                self.image_record("sweep", "sample-a", "cal-front", 1, keyframe=False),
            ],
            "calibrated_sensor": [
                {"token": "cal-front", "sensor_token": "sensor-front"},
                {"token": "cal-back", "sensor_token": "sensor-back"},
            ],
            "sensor": [
                {"token": "sensor-front", "channel": "CAM_FRONT", "modality": "camera"},
                {"token": "sensor-back", "channel": "CAM_BACK", "modality": "camera"},
            ],
            "object_ann": [], "surface_ann": [],
        }

    @staticmethod
    def image_record(token, sample, calibration, timestamp, keyframe=True):
        return {
            "token": token, "sample_token": sample, "calibrated_sensor_token": calibration,
            "is_key_frame": keyframe, "timestamp": timestamp, "width": 4, "height": 3,
            "filename": f"{'samples' if keyframe else 'sweeps'}/{token}.png",
        }

    @staticmethod
    def annotation(token, class_id, mask, image="key-a"):
        return {
            "token": token, "category_token": f"category-{class_id}",
            "sample_data_token": image, "mask": None if mask is None else encode_mask(mask),
            "bbox": [0, 0, 4, 3],
        }

    def load(self, with_annotations=True, omit=(), version=None):
        for name, table in self.tables.items():
            if name not in omit:
                (self.root / self.version / f"{name}.json").write_text(json.dumps(table), encoding="utf-8")
        return NuImagesData(self.root, version or self.version, with_annotations)

    def mask(self, data):
        return data.semantic_mask(data.select_images()[0])

    def test_official_complete_mapping_and_active_ids(self):
        data = self.load()
        self.assertEqual(data.class_names, EXPECTED_NAMES)
        self.assertEqual(sorted(data.class_names), list(range(32)))
        self.assertEqual(data.active_class_ids, set(range(1, 25)) | {31})
        self.assertEqual(data.class_names[17], "vehicle.car")
        self.assertEqual(data.class_names[24], "flat.driveable_surface")
        self.assertEqual(data.class_names[31], "vehicle.ego")
        self.assertTrue(all(i not in data.active_class_ids for i in range(25, 31)))

    def test_subset_category_does_not_renumber(self):
        self.tables["category"] = [item for item in self.tables["category"] if item["token"] == "category-17"]
        data = self.load()
        self.assertEqual(data.active_class_ids, {17})
        self.assertEqual(data.class_names, EXPECTED_NAMES)
        self.tables["object_ann"] = [self.annotation("car", 17, np.ones((3, 4)))]
        mask, _ = self.mask(self.load())
        np.testing.assert_array_equal(mask, np.full((3, 4), 17, dtype=np.uint8))

    def test_base64_compressed_rle_decode_orientation(self):
        expected = np.array([[0, 1, 0, 1], [1, 0, 0, 0], [1, 1, 0, 0]], dtype=np.uint8)
        self.tables["object_ann"] = [self.annotation("animal", 1, expected)]
        data = self.load()
        real_decode = coco_mask.decode
        with patch.object(coco_mask, "decode", wraps=real_decode) as decode:
            mask, stats = self.mask(data)
        self.assertIsInstance(decode.call_args.args[0]["counts"], bytes)
        np.testing.assert_array_equal(mask, expected)
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(mask.shape, (3, 4))
        self.assertEqual(stats, {
            "object_annotations": 1, "surface_annotations": 0,
            "decoded_object_masks": 1, "decoded_surface_masks": 0,
            "null_masks": 0, "annotated_pixels": 5,
        })

    def test_surfaces_in_table_order_objects_in_token_order(self):
        surface = np.array([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0]])
        ego = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        early = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
        late = np.array([[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
        # Surface token sorting would wrongly erase ego. Object table order
        # would wrongly let early overwrite late on the overlapping pixels.
        self.tables["surface_ann"] = [self.annotation("z-road", 24, surface), self.annotation("a-ego", 31, ego)]
        self.tables["object_ann"] = [self.annotation("z-car", 17, late), self.annotation("a-adult", 2, early)]
        mask, stats = self.mask(self.load())
        np.testing.assert_array_equal(mask, [[2, 17, 24, 0], [24, 17, 24, 0], [24, 24, 24, 0]])
        self.assertEqual(stats["decoded_object_masks"], 2)
        self.assertEqual(stats["decoded_surface_masks"], 2)
        self.assertEqual(stats["annotated_pixels"], 9)

    def test_ego_surface_is_31(self):
        self.tables["surface_ann"] = [self.annotation("ego", 31, np.ones((3, 4)))]
        mask, _ = self.mask(self.load())
        np.testing.assert_array_equal(mask, np.full((3, 4), 31, dtype=np.uint8))

    def test_null_masks_skip_bbox_and_count_both_tables(self):
        self.tables["object_ann"] = [self.annotation("object", 17, None)]
        self.tables["surface_ann"] = [self.annotation("surface", 24, None)]
        mask, stats = self.mask(self.load())
        np.testing.assert_array_equal(mask, np.zeros((3, 4), dtype=np.uint8))
        self.assertEqual(stats, {
            "object_annotations": 1, "surface_annotations": 1,
            "decoded_object_masks": 0, "decoded_surface_masks": 0,
            "null_masks": 2, "annotated_pixels": 0,
        })

    def test_empty_annotations_background_zero(self):
        mask, stats = self.mask(self.load())
        self.assertFalse(mask.any())
        self.assertTrue(all(value == 0 for value in stats.values()))

    def test_records_and_json_unchanged_after_repeated_decode(self):
        self.tables["object_ann"] = [self.annotation("object", 17, np.ones((3, 4)))]
        self.tables["surface_ann"] = [self.annotation("surface", 24, np.ones((3, 4)))]
        data = self.load()
        original = copy.deepcopy(data._tables)
        files_before = {path.name: path.read_bytes() for path in (self.root / self.version).glob("*.json")}
        first, first_stats = self.mask(data)
        second, second_stats = self.mask(data)
        self.assertEqual(data._tables, original)
        self.assertEqual(files_before, {path.name: path.read_bytes() for path in (self.root / self.version).glob("*.json")})
        self.assertEqual(first_stats, second_stats)
        np.testing.assert_array_equal(first, second)

    def test_only_key_camera_one_per_sample_sorted(self):
        views = self.load().select_images()
        self.assertEqual([view["camera_token"] for view in views], ["key-a", "key-b"])
        self.assertEqual(len(views), len(self.tables["sample"]))
        self.assertEqual(set(views[0]), {"sample_token", "camera_token", "camera", "image_path", "width", "height", "timestamp"})
        self.assertEqual(views[0]["camera"], "CAM_FRONT")
        self.assertEqual(views[0]["image_path"], self.root / "samples" / "key-a.png")
        self.assertEqual((views[0]["width"], views[0]["height"]), (4, 3))
        self.tables["sample_data"][0]["timestamp"] = 10
        self.assertEqual([v["sample_token"] for v in self.load().select_images()], ["sample-a", "sample-b"])

    def test_six_camera_filter_and_empty_selection(self):
        data = self.load()
        self.assertEqual([v["camera_token"] for v in data.select_images(["CAM_BACK"])], ["key-b"])
        self.assertEqual(data.select_images(["CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]), [])
        self.assertEqual(data.select_images([]), [])
        self.assertEqual(len(data.select_images(["CAM_FRONT", "CAM_FRONT"])), 1)
        with self.assertRaisesRegex(ValueError, "Unknown cameras"):
            data.select_images(["FRONT"])
        with self.assertRaises(TypeError):
            data.select_images("CAM_FRONT")
        with self.assertRaises(TypeError):
            data.select_images([None])

    def test_annotations_disabled_does_not_open_or_require_annotation_tables(self):
        data = self.load(with_annotations=False, omit=("object_ann", "surface_ann"))
        self.assertEqual(len(data.select_images()), 2)
        with self.assertRaisesRegex(RuntimeError, "with_annotations"):
            self.mask(data)
        (self.root / self.version / "object_ann.json").write_text("not JSON", encoding="utf-8")
        self.assertEqual(len(NuImagesData(self.root, with_annotations=False).select_images()), 2)

    def test_missing_tables_and_missing_split(self):
        for name in self.tables:
            with self.subTest(table=name):
                self.load()
                path = self.root / self.version / f"{name}.json"
                path.unlink()
                with self.assertRaisesRegex(FileNotFoundError, name):
                    NuImagesData(self.root)
        self.load()
        with self.assertRaisesRegex(FileNotFoundError, "v1.0-test"):
            NuImagesData(self.root, version="v1.0-test")

    def test_selected_actual_image_dimensions_match_metadata(self):
        data = self.load()
        Image.new("RGB", (8, 6)).save(self.root / "samples" / "key-a.png")
        with self.assertRaisesRegex(ValueError, "dimensions disagree"):
            data.select_images()
        self.assertEqual(len(data.select_images(["CAM_BACK"])), 1)
        with self.assertRaisesRegex(ValueError, "dimensions disagree"):
            data.semantic_mask(dict(data._views_by_token["key-a"]))

    def test_rle_size_mismatch(self):
        self.tables["object_ann"] = [self.annotation("bad-size", 17, np.ones((4, 3)))]
        with self.assertRaisesRegex(ValueError, "mask size"):
            self.mask(self.load())

    def test_invalid_base64_and_rle_types(self):
        for rle in ({"size": [3, 4], "counts": "!!!"}, {"size": [3, 4], "counts": [12]}, {"size": [3.0, 4], "counts": "MA=="}, []):
            with self.subTest(rle=rle):
                item = self.annotation("bad-rle", 17, None)
                item["mask"] = rle
                self.tables["object_ann"] = [item]
                with self.assertRaises(ValueError):
                    self.mask(self.load())

    def test_decoded_shape_must_match_image(self):
        self.tables["object_ann"] = [self.annotation("object", 17, np.ones((3, 4)))]
        data = self.load()
        with patch.object(coco_mask, "decode", return_value=np.ones((3, 4, 1), dtype=np.uint8)):
            with self.assertRaisesRegex(ValueError, "decoded mask shape"):
                self.mask(data)

    def test_unknown_category_name_and_annotation_category(self):
        self.tables["category"][0]["name"] = "vehicle.spaceship"
        with self.assertRaisesRegex(ValueError, "Unknown nuImages category"):
            self.load()
        self.tables["category"][0]["name"] = "vehicle.ego"
        annotation = self.annotation("bad-category", 17, None)
        annotation["category_token"] = "missing-category"
        self.tables["object_ann"] = [annotation]
        with self.assertRaisesRegex(ValueError, "unknown category_token"):
            self.load()

    def test_duplicate_tokens_each_table(self):
        for name in self.tables:
            with self.subTest(table=name):
                original = copy.deepcopy(self.tables[name])
                if not original:
                    self.tables[name] = [self.annotation("annotation", 17, None)]
                self.tables[name].append(copy.deepcopy(self.tables[name][0]))
                with self.assertRaisesRegex(ValueError, "duplicate token"):
                    self.load()
                self.tables[name] = original

    def test_non_keyframe_sample_annotation_and_view_rejected(self):
        self.tables["sample"][1]["key_camera_token"] = "sweep"
        with self.assertRaisesRegex(ValueError, "non-keyframe"):
            self.load()
        self.tables["sample"][1]["key_camera_token"] = "key-a"
        for name in ("object_ann", "surface_ann"):
            self.tables[name] = [self.annotation("bad-key", 17, None, image="other-camera")]
            with self.assertRaisesRegex(ValueError, "non-keyframe"):
                self.load()
            self.tables[name] = []
        data = self.load()
        view = data.select_images()[0]
        view["camera_token"] = "sweep"
        with self.assertRaisesRegex(ValueError, "non-keyframe"):
            data.semantic_mask(view)

    def test_view_cannot_override_image_or_dimensions(self):
        data = self.load()
        for key, value in (("width", 8), ("image_path", self.root / "samples" / "key-b.png"), ("sample_token", "sample-b")):
            with self.subTest(key=key):
                view = data.select_images()[0]
                view[key] = value
                with self.assertRaisesRegex(ValueError, "disagrees with keyframe metadata"):
                    data.semantic_mask(view)

    def test_missing_mask_does_not_fallback_to_bbox(self):
        item = self.annotation("bbox-only", 17, None)
        del item["mask"]
        self.tables["object_ann"] = [item]
        with self.assertRaisesRegex(ValueError, "missing mask"):
            self.load()

    def test_missing_reference_and_sample_mismatch(self):
        self.tables["sample_data"][1]["sample_token"] = "sample-b"
        with self.assertRaisesRegex(ValueError, "mismatched sample_token"):
            self.load()
        self.tables["sample_data"][1]["sample_token"] = "sample-a"
        self.tables["calibrated_sensor"][0]["sensor_token"] = "missing"
        with self.assertRaisesRegex(ValueError, "unknown referenced token"):
            self.load()


if __name__ == "__main__":
    unittest.main()

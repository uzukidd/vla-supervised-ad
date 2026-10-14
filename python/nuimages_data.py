"""Read local nuImages keyframes and official semantic masks without the devkit."""
from __future__ import annotations

import base64
import binascii
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image


# nuImages' alphabetical object IDs are different from nuScenes lidarseg IDs.
_OBJECT_NAMES = (
    "animal",
    "human.pedestrian.adult",
    "human.pedestrian.child",
    "human.pedestrian.construction_worker",
    "human.pedestrian.personal_mobility",
    "human.pedestrian.police_officer",
    "human.pedestrian.stroller",
    "human.pedestrian.wheelchair",
    "movable_object.barrier",
    "movable_object.debris",
    "movable_object.pushable_pullable",
    "movable_object.trafficcone",
    "static_object.bicycle_rack",
    "vehicle.bicycle",
    "vehicle.bus.bendy",
    "vehicle.bus.rigid",
    "vehicle.car",
    "vehicle.construction",
    "vehicle.emergency.ambulance",
    "vehicle.emergency.police",
    "vehicle.motorcycle",
    "vehicle.trailer",
    "vehicle.truck",
)
_CLASS_NAMES = {
    0: "unlabeled",
    **dict(enumerate(_OBJECT_NAMES, 1)),
    24: "flat.driveable_surface",
    **{i: f"unused_{i}" for i in range(25, 31)},
    31: "vehicle.ego",
}
CAMERAS = (
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
)


class NuImagesData:
    """Load ``root/version/*.json`` and images relative to ``root``.

    ``select_images`` returns one key camera per sample, ordered by image
    timestamp then sample token. ``semantic_mask`` returns a uint8 HxW mask
    and integer annotation/coverage counts. With annotations disabled, the
    annotation tables are not opened and semantic masks cannot be requested.
    IDs 25..30 are reserved; ``active_class_ids`` contains only category-table
    classes, keeping their official IDs even for a subset taxonomy.
    """

    def __init__(self, root: Path, version: str = "v1.0-mini", with_annotations: bool = True):
        self.root = Path(root)
        self.version = version
        self.with_annotations = with_annotations
        self.class_names = dict(_CLASS_NAMES)
        self._tables = {}
        self._indices = {}
        table_names = ["category", "sample", "sample_data", "calibrated_sensor", "sensor"]
        if with_annotations:
            table_names += ["object_ann", "surface_ann"]
        for name in table_names:
            path = self.root / version / f"{name}.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing nuImages table for split {version!r}: {path}")
            records = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                raise ValueError(f"{name}: table must contain a JSON list")
            index = {}
            for record in records:
                if not isinstance(record, dict) or not isinstance(record.get("token"), str) or not record["token"]:
                    raise ValueError(f"{name}: every record must have a nonempty string token")
                token = record["token"]
                if token in index:
                    raise ValueError(f"{name}: duplicate token {token!r}")
                index[token] = record
            self._tables[name] = records
            self._indices[name] = index

        name_to_id = {name: i for i, name in self.class_names.items() if i in set(range(1, 25)) | {31}}
        self._category_ids = {}
        seen_names = set()
        for category in self._tables["category"]:
            name = category.get("name")
            if name not in name_to_id:
                raise ValueError(f"Unknown nuImages category name: {name!r}")
            if name in seen_names:
                raise ValueError(f"Duplicate nuImages category name: {name!r}")
            seen_names.add(name)
            self._category_ids[category["token"]] = name_to_id[name]
        self.active_class_ids = set(self._category_ids.values())

        self._key_samples = {}
        self._views = []
        for sample in self._tables["sample"]:
            token = sample.get("key_camera_token")
            record = self._get("sample_data", token)
            if record.get("is_key_frame") is not True:
                raise ValueError(f"Sample {sample['token']!r} references non-keyframe {token!r}")
            if record.get("sample_token") != sample["token"]:
                raise ValueError(f"Keyframe {token!r} has mismatched sample_token")
            if token in self._key_samples:
                raise ValueError(f"Duplicate key_camera_token {token!r}")
            self._key_samples[token] = sample["token"]
            calibration = self._get("calibrated_sensor", record.get("calibrated_sensor_token"))
            sensor = self._get("sensor", calibration.get("sensor_token"))
            camera = sensor.get("channel")
            if sensor.get("modality") != "camera" or camera not in CAMERAS:
                raise ValueError(f"Keyframe {token!r} does not reference a supported camera")
            for dimension in ("width", "height"):
                if type(record.get(dimension)) is not int or record[dimension] <= 0:
                    raise ValueError(f"Keyframe {token!r}: invalid {dimension}")
            if type(record.get("timestamp")) is not int:
                raise ValueError(f"Keyframe {token!r}: invalid timestamp")
            filename = record.get("filename")
            if not isinstance(filename, str) or not filename:
                raise ValueError(f"Keyframe {token!r}: missing image filename")
            self._views.append({
                "sample_token": sample["token"], "camera_token": token, "camera": camera,
                "image_path": self.root / filename, "width": record["width"],
                "height": record["height"], "timestamp": record["timestamp"],
            })
        self._views.sort(key=lambda view: (view["timestamp"], view["sample_token"]))
        self._views_by_token = {view["camera_token"]: view for view in self._views}
        self._annotations = {"object_ann": defaultdict(list), "surface_ann": defaultdict(list)}
        if with_annotations:
            for name in self._annotations:
                for annotation in self._tables[name]:
                    token = annotation.get("sample_data_token")
                    if token not in self._key_samples:
                        raise ValueError(f"{name} {annotation['token']!r}: annotation references non-keyframe or unknown sample_data {token!r}")
                    if annotation.get("category_token") not in self._category_ids:
                        raise ValueError(f"{name} {annotation['token']!r}: unknown category_token {annotation.get('category_token')!r}")
                    if "mask" not in annotation:
                        raise ValueError(f"{name} {annotation['token']!r}: missing mask (use null for unavailable masks)")
                    self._annotations[name][token].append(annotation)
            for annotations in self._annotations["object_ann"].values():
                annotations.sort(key=lambda annotation: annotation["token"])

    def _get(self, table: str, token: str) -> dict:
        try:
            return self._indices[table][token]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{table}: unknown referenced token {token!r}") from exc

    @staticmethod
    def _image_size(view: dict) -> tuple[int, int]:
        with Image.open(view["image_path"]) as image:
            width, height = image.size
        if (width, height) != (view["width"], view["height"]):
            raise ValueError(
                f"Image dimensions disagree with metadata for {view['camera_token']!r}: "
                f"actual {(width, height)}, metadata {(view['width'], view['height'])}"
            )
        return height, width

    def select_images(self, cameras: list[str] | None = None) -> list[dict]:
        """Select annotated keyframe cameras; sweeps and other cameras are excluded."""
        if cameras is not None:
            if not isinstance(cameras, list) or any(not isinstance(camera, str) for camera in cameras):
                raise TypeError("cameras must be a list of camera channel strings or None")
            unknown = set(cameras) - set(CAMERAS)
            if unknown:
                raise ValueError(f"Unknown cameras: {sorted(unknown)}")
        selected = [dict(view) for view in self._views if cameras is None or view["camera"] in cameras]
        for view in selected:
            self._image_size(view)
        return selected

    @staticmethod
    def _decode_mask(record: dict, shape: tuple[int, int], token: str) -> np.ndarray:
        if not isinstance(record, dict):
            raise ValueError(f"Annotation {token!r}: mask must be an RLE dictionary or null")
        size = record.get("size")
        if not isinstance(size, list) or len(size) != 2 or any(type(value) is not int for value in size) or tuple(size) != shape:
            raise ValueError(f"Annotation {token!r}: mask size {size!r} disagrees with image dimensions {shape}")
        counts = record.get("counts")
        if not isinstance(counts, str):
            raise ValueError(f"Annotation {token!r}: counts must be base64 encoded COCO compressed RLE")
        try:
            compressed = base64.b64decode(counts, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError(f"Annotation {token!r}: invalid base64 mask counts") from exc
        from pycocotools import mask as coco_mask
        try:
            decoded = coco_mask.decode({"size": list(size), "counts": compressed})
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"Annotation {token!r}: invalid COCO compressed RLE") from exc
        if decoded.shape != shape:
            raise ValueError(f"Annotation {token!r}: decoded mask shape {decoded.shape} disagrees with image dimensions {shape}")
        return decoded.astype(bool)

    def semantic_mask(self, view: dict) -> tuple[np.ndarray, dict]:
        """Paint surfaces in table order, then objects in ascending token order.

        Null masks contribute to annotation and null counts, but paint nothing.
        ``annotated_pixels`` counts nonzero pixels in the final composed mask.
        RLE decoding uses a fresh dictionary and never modifies loaded records.
        """
        if not self.with_annotations:
            raise RuntimeError("semantic_mask requires with_annotations=True")
        token = view.get("camera_token")
        if token not in self._views_by_token:
            raise ValueError(f"View references non-keyframe or unknown camera_token {token!r}")
        canonical = self._views_by_token[token]
        for key, value in canonical.items():
            if key not in view or view[key] != value:
                raise ValueError(f"View {token!r}: {key} disagrees with keyframe metadata")
        shape = self._image_size(canonical)
        result = np.zeros(shape, dtype=np.uint8)
        objects = self._annotations["object_ann"][token]
        surfaces = self._annotations["surface_ann"][token]
        stats = {
            "object_annotations": len(objects), "surface_annotations": len(surfaces),
            "decoded_object_masks": 0, "decoded_surface_masks": 0,
            "null_masks": 0, "annotated_pixels": 0,
        }
        for annotations, decoded_key in ((surfaces, "decoded_surface_masks"), (objects, "decoded_object_masks")):
            for annotation in annotations:
                if annotation["mask"] is None:
                    stats["null_masks"] += 1
                    continue
                decoded = self._decode_mask(annotation["mask"], shape, annotation["token"])
                result[decoded] = self._category_ids[annotation["category_token"]]
                stats[decoded_key] += 1
        stats["annotated_pixels"] = int(np.count_nonzero(result))
        return result, stats

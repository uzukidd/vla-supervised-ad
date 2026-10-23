"""Read nuScenes versions and project lidar keyframes without the devkit.

Official taxonomy/index ordering (also asserted by the official lidarseg test):
https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/color_map.py
https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/tests/test_lidarseg.py
Official benchmark merge and mini splits:
https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/lidarseg/utils.py
https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/splits.py

The box-only category.json is deliberately unnecessary. Projection uses both
sensor timestamps, native image dimensions and only a per-pixel nearest-point
rule. It makes no additional visibility/occlusion assumptions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


CAMERAS = (
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
)
RAW_CLASS_NAMES: dict[int, str] = dict(enumerate((
    "noise", "animal", "human.pedestrian.adult", "human.pedestrian.child",
    "human.pedestrian.construction_worker", "human.pedestrian.personal_mobility",
    "human.pedestrian.police_officer", "human.pedestrian.stroller",
    "human.pedestrian.wheelchair", "movable_object.barrier", "movable_object.debris",
    "movable_object.pushable_pullable", "movable_object.trafficcone",
    "static_object.bicycle_rack", "vehicle.bicycle", "vehicle.bus.bendy",
    "vehicle.bus.rigid", "vehicle.car", "vehicle.construction",
    "vehicle.emergency.ambulance", "vehicle.emergency.police", "vehicle.motorcycle",
    "vehicle.trailer", "vehicle.truck", "flat.driveable_surface", "flat.other",
    "flat.sidewalk", "flat.terrain", "static.manmade", "static.other",
    "static.vegetation", "vehicle.ego",
)))
COARSE_CLASS_NAMES: dict[int, str] = dict(enumerate((
    "ignore", "barrier", "bicycle", "bus", "car", "construction_vehicle",
    "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck",
    "driveable_surface", "other_flat", "sidewalk", "terrain", "manmade", "vegetation",
)))
RAW_TO_COARSE = np.array([
    0, 0, 7, 7, 7, 0, 7, 0, 0, 1, 0, 0, 8, 0, 2, 3,
    3, 4, 5, 0, 0, 6, 9, 10, 11, 12, 13, 14, 15, 0, 16, 0,
], dtype=np.uint8)
_MINI_SPLITS = {
    "mini_train": (
        "scene-0061", "scene-0553", "scene-0655", "scene-0757",
        "scene-0796", "scene-1077", "scene-1094", "scene-1100",
    ),
    "mini_val": ("scene-0103", "scene-0916"),
}


def _quaternion_matrix(rotation: list[float]) -> np.ndarray:
    """Return the active rotation matrix for a quaternion in w, x, y, z order."""
    quaternion = np.asarray(rotation, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("rotation must be a finite wxyz quaternion of length 4")
    norm = np.linalg.norm(quaternion)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("rotation quaternion must have a finite nonzero norm")
    w, x, y, z = quaternion / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _transform(points: np.ndarray, pose: dict, inverse: bool = False) -> np.ndarray:
    """Apply R p + t (or R.T (p - t)) to row-major Nx3 points."""
    rotation = _quaternion_matrix(pose["rotation"])
    translation = np.asarray(pose["translation"], dtype=np.float64)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("translation must contain three finite coordinates")
    if inverse:
        return (points - translation) @ rotation
    return points @ rotation.T + translation


class NuScenesMini:
    """Index nuScenes metadata; require ground-truth labels by default."""

    def __init__(
        self, root: Path, version: str = "v1.0-mini", require_lidarseg: bool = True,
    ) -> None:
        if version not in ("v1.0-mini", "v1.0-trainval", "v1.0-test"):
            raise ValueError(f"unsupported nuScenes version {version!r}; use v1.0-mini, v1.0-trainval or v1.0-test")
        self.root = Path(root)
        self.version = version
        self.require_lidarseg = require_lidarseg
        self._tables: dict[str, dict[str, dict]] = {}
        for name in (
            "scene", "sample", "sample_data", "calibrated_sensor",
            "sensor", "ego_pose", "lidarseg",
        ):
            path = self.root / version / f"{name}.json"
            if name == "lidarseg" and not require_lidarseg and not path.is_file():
                self._tables[name] = {}
                continue
            with path.open(encoding="utf-8") as stream:
                records = json.load(stream)
            if not isinstance(records, list):
                raise ValueError(f"{path}: expected a JSON array")
            index = {}
            for record in records:
                token = record["sample_data_token"] if name == "lidarseg" else record["token"]
                if token in index:
                    raise ValueError(f"{path}: duplicate token {token}")
                index[token] = record
            self._tables[name] = index
        self._scene_by_name = {
            scene["name"]: scene for scene in self._tables["scene"].values()
        }
        if len(self._scene_by_name) != len(self._tables["scene"]):
            raise ValueError("scene table contains duplicate scene names")
        self._keyframes: dict[str, dict[str, dict]] = {}
        for record in self._tables["sample_data"].values():
            if not record["is_key_frame"]:
                continue
            calibration = self._get("calibrated_sensor", record["calibrated_sensor_token"])
            sensor = self._get("sensor", calibration["sensor_token"])
            channel = sensor["channel"]
            if channel not in CAMERAS and channel != "LIDAR_TOP":
                continue
            channels = self._keyframes.setdefault(record["sample_token"], {})
            if channel in channels:
                raise ValueError(f"sample {record['sample_token']}: duplicate {channel} keyframe")
            channels[channel] = record

    def _get(self, table: str, token: str) -> dict:
        try:
            return self._tables[table][token]
        except KeyError as error:
            raise ValueError(f"missing {table} record for token {token}") from error

    def _media_path(self, filename: str, description: str) -> Path:
        path = self.root / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing {description}: {path}")
        return path

    def select_views(
        self, cameras: list[str], split: str = "all", scenes: list[str] | None = None,
    ) -> list[dict]:
        """Select keyframes sorted by scene, sample timestamp, then camera order.

        An explicit scene list filters the split (intersection). Unknown names
        and unsupported splits fail rather than silently producing no views.
        Every selected sample must have LIDAR_TOP and all requested cameras.
        Label records/files are required only when require_lidarseg is True.
        """
        if split not in ("all", "mini_train", "mini_val"):
            raise ValueError(f"unsupported split {split!r}; use all, mini_train or mini_val")
        if split != "all" and self.version != "v1.0-mini":
            raise ValueError(f"{split} requires version v1.0-mini")
        if any(camera not in CAMERAS for camera in cameras):
            raise ValueError(f"unknown camera channel; supported channels: {CAMERAS}")
        if len(set(cameras)) != len(cameras):
            raise ValueError("camera channels must not be repeated")
        available = set(self._scene_by_name)
        selected = available if split == "all" else available.intersection(_MINI_SPLITS[split])
        if scenes is not None:
            unknown = set(scenes) - available
            if unknown:
                raise ValueError(f"unknown scenes: {sorted(unknown)}")
            selected = selected.intersection(scenes)
        if not cameras:
            return []
        samples = []
        for sample in self._tables["sample"].values():
            name = self._get("scene", sample["scene_token"])["name"]
            if name in selected:
                samples.append((name, sample))
        samples.sort(key=lambda item: (item[0], item[1]["timestamp"], item[1]["token"]))
        views = []
        for scene_name, sample in samples:
            token = sample["token"]
            channels = self._keyframes.get(token, {})
            if "LIDAR_TOP" not in channels:
                raise ValueError(f"sample {token} ({scene_name}): missing LIDAR_TOP keyframe")
            lidar = channels["LIDAR_TOP"]
            if self.require_lidarseg:
                label = self._get("lidarseg", lidar["token"])
                self._media_path(label["filename"], f"lidarseg labels for {lidar['token']}")
            self._media_path(lidar["filename"], f"LIDAR_TOP points for {lidar['token']}")
            for camera in cameras:
                if camera not in channels:
                    raise ValueError(f"sample {token} ({scene_name}): missing {camera} keyframe")
                camera_record = channels[camera]
                views.append({
                    "sample_token": token,
                    "scene_name": scene_name,
                    "camera": camera,
                    "camera_token": camera_record["token"],
                    "lidar_token": lidar["token"],
                    "image_path": self._media_path(camera_record["filename"], f"{camera} image"),
                })
        return views

    def load_lidar(self, lidar_token: str) -> dict:
        """Load native LiDAR coordinates; raw_labels is None in optional-label mode."""
        lidar = self._get("sample_data", lidar_token)
        sensor = self._get("sensor", self._get("calibrated_sensor", lidar["calibrated_sensor_token"])["sensor_token"])
        if not lidar["is_key_frame"] or sensor["channel"] != "LIDAR_TOP":
            raise ValueError("load_lidar requires a LIDAR_TOP keyframe token")
        point_path = self._media_path(lidar["filename"], "LIDAR_TOP points")
        if point_path.stat().st_size % (5 * np.dtype(np.float32).itemsize):
            raise ValueError(f"{point_path}: point file must be float32 Nx5 (20 bytes per point)")
        points = np.fromfile(point_path, dtype=np.float32).reshape(-1, 5)
        labels = None
        if self.require_lidarseg:
            label_path = self._media_path(self._get("lidarseg", lidar_token)["filename"], "lidarseg labels")
            labels = np.fromfile(label_path, dtype=np.uint8)
            if len(labels) != len(points):
                raise ValueError(f"label/point count mismatch: {len(labels)} labels vs {len(points)} points ({label_path})")
            if np.any(labels > 31):
                raise ValueError(f"{label_path}: raw labels must be in 0..31")
        if not np.all(np.isfinite(points[:, :3])):
            raise ValueError(f"{point_path}: point coordinates must be finite")
        return {"xyz": points[:, :3], "raw_labels": labels, "intensity": points[:, 3], "ring": points[:, 4]}

    def projection(self, view: dict, min_depth: float = 1.0) -> dict:
        """Project a keyframe and retain the nearest point per floor pixel.

        Output arrays are aligned and sorted by original lidar point index.
        Counts describe each successive filter; in_image_points is counted
        before pixel deduplication. Strict mode keeps all labels, including
        ignore classes; optional-label mode returns raw_labels=None.
        """
        if not np.isfinite(min_depth) or min_depth < 0:
            raise ValueError("min_depth must be finite and nonnegative")
        lidar = self._get("sample_data", view["lidar_token"])
        camera = self._get("sample_data", view["camera_token"])
        for record in (lidar, camera):
            if not record["is_key_frame"] or record["sample_token"] != view["sample_token"]:
                raise ValueError("view must refer to camera/lidar keyframes of its sample")
        cloud = self.load_lidar(view["lidar_token"])
        points = cloud["xyz"].astype(np.float64)
        labels = cloud["raw_labels"]
        lidar_calibration = self._get("calibrated_sensor", lidar["calibrated_sensor_token"])
        camera_calibration = self._get("calibrated_sensor", camera["calibrated_sensor_token"])
        points = _transform(points, lidar_calibration)
        points = _transform(points, self._get("ego_pose", lidar["ego_pose_token"]))
        points = _transform(points, self._get("ego_pose", camera["ego_pose_token"]), inverse=True)
        points = _transform(points, camera_calibration, inverse=True)
        image_path = self._media_path(camera["filename"], "camera image")
        with Image.open(image_path) as image:
            width, height = image.size
        intrinsic = np.asarray(camera_calibration["camera_intrinsic"], dtype=np.float64)
        if intrinsic.shape != (3, 3) or not np.all(np.isfinite(intrinsic)):
            raise ValueError("camera_intrinsic must be a finite 3x3 matrix")
        depth = points[:, 2]
        indices = np.flatnonzero(depth > min_depth).astype(np.int64)
        in_front_points = len(indices)
        projected = points[indices] @ intrinsic.T
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = projected[:, :2] / projected[:, 2:3]
        inside = (
            np.all(np.isfinite(uv), axis=1)
            & (uv[:, 0] > 1) & (uv[:, 0] < width - 1)
            & (uv[:, 1] > 1) & (uv[:, 1] < height - 1)
        )
        indices, uv = indices[inside], uv[inside]
        pixels = np.floor(uv).astype(np.int64)
        in_image_points = len(indices)
        # Pixel is the primary key, then depth, then original index for exact ties.
        pixel_ids = pixels[:, 1] * width + pixels[:, 0]
        order = np.lexsort((indices, depth[indices], pixel_ids))
        sorted_pixels = pixel_ids[order]
        first = np.ones(len(order), dtype=bool)
        first[1:] = sorted_pixels[1:] != sorted_pixels[:-1]
        winners = order[first]
        winners = winners[np.argsort(indices[winners])]
        indices, uv, pixels = indices[winners], uv[winners], pixels[winners]
        retained = len(indices)
        return {
            "point_indices": indices,
            "uv": uv,
            "pixels": pixels,
            "depth": depth[indices],
            "raw_labels": None if labels is None else labels[indices],
            "image_size_wh": [width, height],
            "counts": {
                "total_points": len(points),
                "in_front_points": in_front_points,
                "in_image_points": in_image_points,
                "retained_points": retained,
                "same_pixel_discarded": in_image_points - retained,
            },
        }

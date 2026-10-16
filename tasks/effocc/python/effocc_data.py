# Inference adaptation of EFFOcc/BEVDet input processing (Apache-2.0).
# Original pipeline components: Copyright (c) OpenMMLab. All rights reserved.
"""Dependency-free EFFOcc FusionR18 test data preparation.

Geometry follows PrepareImageInputs, LoadPointsFromMultiSweeps, ToEgo,
PointToMultiViewDepthFusion and BEVDet.prepare_inputs in models/EFFOcc.
The original pipeline has two different ego references: points use CAM_FRONT,
while camera2keyego uses the first configured camera, CAM_FRONT_LEFT. Fusion
projects the former with sensor2ego calibration only (no camera-time motion
compensation). These details are deliberate, not corrected here.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from shared.python.nuscenes_data import NuScenesMini, _quaternion_matrix


CAMERAS = (
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
)
CAMERA_NAMES = CAMERAS
INPUT_SIZE = (256, 704)
IMAGE_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMAGE_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def _pose_matrix(record: dict, dtype=np.float64) -> np.ndarray:
    matrix = np.eye(4, dtype=dtype)
    matrix[:3, :3] = _quaternion_matrix(record["rotation"])
    translation = np.asarray(record["translation"], dtype=dtype)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("translation must contain three finite coordinates")
    matrix[:3, 3] = translation
    return matrix


def _sensor_records(dataset: NuScenesMini, record: dict, channel: str):
    calibration = dataset._get("calibrated_sensor", record["calibrated_sensor_token"])
    sensor = dataset._get("sensor", calibration["sensor_token"])
    if sensor["channel"] != channel:
        raise ValueError(f"sample_data {record['token']} must belong to {channel}")
    return calibration, dataset._get("ego_pose", record["ego_pose_token"])


def _read_points(dataset: NuScenesMini, record: dict) -> tuple[np.ndarray, Path]:
    path = dataset._media_path(record["filename"], "EFFOcc LiDAR points")
    if path.stat().st_size % 20:
        raise ValueError(f"{path}: point file must be float32 Nx5 (20 bytes per point)")
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{path}: point features must be finite")
    return points, path


def _remove_close(points: np.ndarray) -> np.ndarray:
    # Official loader uses a square, not Euclidean distance, before alignment.
    return points[~((np.abs(points[:, 0]) < 1) & (np.abs(points[:, 1]) < 1))]


def _prepare_image(path: Path):
    height, width = INPUT_SIZE
    with Image.open(path) as source:
        source = source.convert("RGB")
        original_size = source.size
        resize = width / source.width
        resize_dims = (int(source.width * resize), int(source.height * resize))
        crop_h = int(resize_dims[1]) - height
        crop_w = int(max(0, resize_dims[0] - width) / 2)
        crop = (crop_w, crop_h, crop_w + width, crop_h + height)
        # RGB Pillow's default is BICUBIC; source calls resize without resample.
        image = source.resize(resize_dims).crop(crop).rotate(0)
        pixels = np.asarray(image, dtype=np.float32)
    # Match mmcv.imnormalize_ including its double-precision scale constants.
    # to_rgb=True reverses the already RGB PIL array to BGR.
    normalized = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
    cv2.subtract(normalized, IMAGE_MEAN.astype(np.float64).reshape(1, -1), normalized)
    cv2.multiply(normalized, (1 / IMAGE_STD.astype(np.float64)).reshape(1, -1), normalized)
    post_rot = np.diag([resize, resize, 1]).astype(np.float32)
    post_trans = np.array([-crop_w, -crop_h, 0], dtype=np.float32)
    metadata = {
        "original_size_wh": list(original_size), "resize": resize,
        "resize_dims_wh": list(resize_dims), "crop_xyxy": list(crop),
        "flip": False, "rotate_degrees": 0, "interpolation": "PIL default BICUBIC",
    }
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)), post_rot, post_trans, metadata


def _points_to_depth_map(projected: np.ndarray, height=256, width=704) -> np.ndarray:
    """Round (ties to even) augmented pixels; keep closest depth in [1,45).

    Use a stable minimum instead of upstream's float32 rank+depth/100 sort:
    large pixel ranks can quantize distinct depths to the same sorting key.
    Geometry follows the original, but backend bitwise equality is not promised.
    """
    pixels = np.rint(projected[:, :2])
    depth = projected[:, 2]
    keep = (
        np.isfinite(pixels).all(axis=1) & np.isfinite(depth)
        & (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        & (depth >= 1) & (depth < 45)
    )
    pixels, depth = pixels[keep].astype(np.int64), depth[keep]
    buffer = np.full(height * width, np.inf, dtype=np.float32)
    np.minimum.at(buffer, pixels[:, 1] * width + pixels[:, 0], depth)
    buffer[~np.isfinite(buffer)] = 0
    return buffer.reshape(height, width)


def prepare_sample(dataset: NuScenesMini, views: list[dict], sweeps_num: int = 9) -> dict:
    """Prepare exactly six views of one keyframe, in the fixed EFFOcc order.

    Return unbatched float32 NumPy images, depth_maps, camera2keyego,
    intrinsics, post_rots, post_trans and Nx5 points, plus JSON-friendly info.
    Points retain intensity; the fifth feature is lag in seconds (keyframe 0).
    Previous sweeps are selected nearest first. Only a completely empty sweep
    list is padded; a shorter nonempty history is not padded, as in upstream.

    Images reproduce upstream mmlabNormalize exactly: PIL RGB is passed into
    mmcv imnormalize(to_rgb=True), so channels are reversed to BGR before
    subtracting the configured mean and dividing by the configured std.
    """
    if isinstance(sweeps_num, bool) or not isinstance(sweeps_num, int) or sweeps_num < 0:
        raise ValueError("sweeps_num must be a nonnegative integer")
    if len(views) != 6 or {view.get("camera") for view in views} != set(CAMERAS):
        raise ValueError("views must contain exactly all six EFFOcc cameras without duplicates")
    if len({view.get("sample_token") for view in views}) != 1:
        raise ValueError("all camera views must refer to the same sample")
    if len({view.get("lidar_token") for view in views}) != 1:
        raise ValueError("all camera views must refer to the same LiDAR keyframe")
    sample_token, lidar_token = views[0]["sample_token"], views[0]["lidar_token"]
    sample = dataset._get("sample", sample_token)
    canonical = dataset._keyframes.get(sample_token, {})
    lidar = dataset._get("sample_data", lidar_token)
    if (not lidar["is_key_frame"] or lidar["sample_token"] != sample_token
            or canonical.get("LIDAR_TOP", {}).get("token") != lidar_token):
        raise ValueError("views must refer to the sample's LIDAR_TOP keyframe")
    lidar_cal, lidar_pose = _sensor_records(dataset, lidar, "LIDAR_TOP")
    by_camera = {view["camera"]: view for view in views}
    camera_records, calibrations, camera_poses, image_paths = [], [], [], []
    for name in CAMERAS:
        view = by_camera[name]
        camera = dataset._get("sample_data", view["camera_token"])
        if (not camera["is_key_frame"] or camera["sample_token"] != sample_token
                or canonical.get(name, {}).get("token") != camera["token"]):
            raise ValueError(f"{name}: camera must be the same sample's canonical keyframe")
        calibration, pose = _sensor_records(dataset, camera, name)
        path = dataset._media_path(camera["filename"], f"{name} image")
        if "image_path" in view and Path(view["image_path"]).resolve() != path.resolve():
            raise ValueError(f"{name}: image_path does not match camera metadata")
        camera_records.append(camera)
        calibrations.append(calibration)
        camera_poses.append(pose)
        image_paths.append(path)

    images, post_rots, post_trans, image_metadata, intrinsics = [], [], [], [], []
    for path, calibration in zip(image_paths, calibrations):
        image, post_rot, post_tran, metadata = _prepare_image(path)
        intrinsic = np.asarray(calibration["camera_intrinsic"], dtype=np.float32)
        if intrinsic.shape != (3, 3) or not np.all(np.isfinite(intrinsic)):
            raise ValueError("camera_intrinsic must be a finite 3x3 matrix")
        images.append(image)
        post_rots.append(post_rot)
        post_trans.append(post_tran)
        image_metadata.append(metadata)
        intrinsics.append(intrinsic)
    # prepare_inputs first rounds poses/calibration to float32, then uses double.
    sensor2egos = np.stack([_pose_matrix(cal, np.float32) for cal in calibrations])
    ego2globals = np.stack([_pose_matrix(pose, np.float32) for pose in camera_poses])
    camera2keyego = (
        np.linalg.inv(ego2globals[0].astype(np.float64))[None]
        @ ego2globals.astype(np.float64) @ sensor2egos.astype(np.float64)
    ).astype(np.float32)

    points, key_path = _read_points(dataset, lidar)
    points[:, 4] = 0
    clouds = [points]
    sweep_records, sweep_paths, sweep_timestamps = [], [], []
    previous = lidar.get("prev", "")
    seen = {lidar_token}
    while previous and len(sweep_records) < sweeps_num:
        if previous in seen:
            raise ValueError("cycle in LiDAR previous sweep chain")
        seen.add(previous)
        sweep = dataset._get("sample_data", previous)
        _sensor_records(dataset, sweep, "LIDAR_TOP")
        if sweep["timestamp"] >= (sweep_records[-1] if sweep_records else lidar)["timestamp"]:
            raise ValueError("LiDAR previous sweep timestamps must decrease")
        sweep_records.append(sweep)
        previous = sweep.get("prev", "")
    key_lidar2global = _pose_matrix(lidar_pose) @ _pose_matrix(lidar_cal)
    global2key_lidar = np.linalg.inv(key_lidar2global)
    for sweep in sweep_records:
        sweep_cal, sweep_pose = _sensor_records(dataset, sweep, "LIDAR_TOP")
        sweep_points, path = _read_points(dataset, sweep)
        sweep_points = _remove_close(sweep_points)
        transform = global2key_lidar @ _pose_matrix(sweep_pose) @ _pose_matrix(sweep_cal)
        # Upstream separately assigns rotation and then adds translation to f32.
        sweep_points[:, :3] = sweep_points[:, :3] @ transform[:3, :3].T
        sweep_points[:, :3] += transform[:3, 3]
        sweep_points[:, 4] = lidar["timestamp"] / 1e6 - sweep["timestamp"] / 1e6
        clouds.append(sweep_points)
        sweep_paths.append(str(path))
        sweep_timestamps.append(sweep["timestamp"])
    padded = not sweep_records and sweeps_num > 0
    if padded:
        clouds.extend([_remove_close(points)] * sweeps_num)
    points = np.concatenate(clouds).astype(np.float32, copy=False)
    # ToEgo performs these products entirely in float32, using CAM_FRONT pose.
    front_index = CAMERAS.index("CAM_FRONT")
    lidar2front_ego = (
        np.linalg.inv(_pose_matrix(camera_poses[front_index], np.float32))
        @ _pose_matrix(lidar_pose, np.float32) @ _pose_matrix(lidar_cal, np.float32)
    )
    points[:, :3] = points[:, :3] @ lidar2front_ego[:3, :3].T + lidar2front_ego[:3, 3]

    depth_maps = []
    for sensor2ego, intrinsic, post_rot, post_tran in zip(
            sensor2egos, intrinsics, post_rots, post_trans):
        # Fusion intentionally ignores each camera's ego2global here.
        ego2image = intrinsic @ np.linalg.inv(sensor2ego)[:3]
        projected = points[:, :3] @ ego2image[:, :3].T + ego2image[:, 3]
        with np.errstate(divide="ignore", invalid="ignore"):
            projected[:, :2] /= projected[:, 2:3]
            projected = projected @ post_rot.T + post_tran
        depth_maps.append(_points_to_depth_map(projected))

    return {
        "images": np.stack(images), "depth_maps": np.stack(depth_maps),
        "camera2keyego": camera2keyego, "intrinsics": np.stack(intrinsics),
        "post_rots": np.stack(post_rots), "post_trans": np.stack(post_trans),
        "points": np.ascontiguousarray(points),
        "info": {
            "camera_names": list(CAMERAS), "sample_token": sample_token,
            "lidar_token": lidar_token, "sample_timestamp": sample["timestamp"],
            "lidar_timestamp": lidar["timestamp"],
            "camera_timestamps": [record["timestamp"] for record in camera_records],
            "timestamp_units": "microseconds", "point_lag_units": "seconds",
            "lidar_path": str(key_path), "image_paths": [str(path) for path in image_paths],
            "sweep_paths": sweep_paths, "sweep_timestamps": sweep_timestamps,
            "sweep_tokens": [record["token"] for record in sweep_records],
            "requested_sweeps": sweeps_num, "used_sweep_count": len(sweep_records),
            "padded_sweep_count": sweeps_num if padded else 0,
            "sweep_selection": "deterministic nearest previous first; upstream config omits test_mode and randomly chooses when history exceeds nine; nonempty short history is not padded",
            "points_reference_camera": "CAM_FRONT", "camera_keyego_reference": CAMERAS[0],
            "depth_projection": "upstream Fusion sensor2ego inverse; no camera-time ego compensation",
            "depth_range": [1.0, 45.0], "depth_pixel_rounding": "nearest, ties to even",
            "depth_collision_rule": "numerically stable minimum depth per rounded pixel; does not reproduce upstream float32 rank sort quantization",
            "bitwise_original_backend_equivalence": False,
            "norm_channel_order": "BGR converted from PIL RGB source to_rgb=True",
            "normalization": {
                "color_order": "BGR", "mean": IMAGE_MEAN.tolist(), "std": IMAGE_STD.tolist(),
                "source_behavior": "upstream mmcv to_rgb=True reverses PIL RGB to BGR",
            },
            "image_augmentation": image_metadata,
        },
    }

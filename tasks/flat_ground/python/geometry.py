"""NumPy geometry for a horizontal ground plane; no dataset or LiDAR access.

BEV coordinates are x forward, y left, z up. Image coordinates use integer
pixel centres, matching the camera calibration (there is no half-pixel shift).
"""

from __future__ import annotations

import numpy as np

from shared.python.nuscenes_data import _quaternion_matrix as quaternion_matrix


_MAX_GRID_CELLS = 16_000_000
_PARALLEL_EPS = 1e-10


def _scalar(value: float, name: str, *, positive: bool = False) -> float:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite scalar")
    result = float(array)
    if not np.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


def _rigid_matrix(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-8):
        raise ValueError(f"{name} must have homogeneous last row [0, 0, 0, 1]")
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-7):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-7):
        raise ValueError(f"{name} rotation must be proper (determinant +1)")
    return matrix


def _intrinsics(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("K must be a finite invertible 3x3 matrix")
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("K must be invertible") from error
    if not np.all(np.isfinite(inverse)):
        raise ValueError("K inverse must be finite")
    return matrix, inverse


def _limits(min_depth: float, max_distance: float) -> tuple[float, float]:
    return (
        _scalar(min_depth, "min_depth", positive=True),
        _scalar(max_distance, "max_distance", positive=True),
    )


def pose_matrix(record: dict) -> np.ndarray:
    """Return float64 T_parent_from_local from a wxyz rotation and translation."""
    if not isinstance(record, dict) or "rotation" not in record or "translation" not in record:
        raise ValueError("pose record must contain rotation and translation")
    rotation = quaternion_matrix(record["rotation"])
    translation = np.asarray(record["translation"], dtype=np.float64)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("translation must contain three finite coordinates")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def ego_yaw_frame(ego_pose: dict, plane_z: float = 0.0) -> np.ndarray:
    """Return T_global_from_bev with ego xy, global plane_z and only ego yaw.

    Yaw is atan2(R[1, 0], R[0, 0]); ego roll, pitch and altitude do not tilt
    or elevate the horizontal reference plane.
    """
    matrix = pose_matrix(ego_pose)
    plane_z = _scalar(plane_z, "plane_z")
    yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    cosine, sine = np.cos(yaw), np.sin(yaw)
    result = np.eye(4, dtype=np.float64)
    result[:2, :2] = [[cosine, -sine], [sine, cosine]]
    result[:2, 3] = matrix[:2, 3]
    result[2, 3] = plane_z
    return result


def bev_ground_grid(
    roi_size: float = 100.0, resolution: float = 0.2,
) -> tuple[np.ndarray, dict]:
    """Return HxWx2 float64 cell centres and square-grid metadata.

    Columns increase x; rows decrease y. The default grid is 500x500 with
    centre coordinates from -49.9 to +49.9 metres on both axes.
    """
    roi_size = _scalar(roi_size, "roi_size", positive=True)
    resolution = _scalar(resolution, "resolution", positive=True)
    with np.errstate(over="ignore"):
        ratio = roi_size / resolution
    if not np.isfinite(ratio) or ratio > np.sqrt(_MAX_GRID_CELLS) + 1e-7:
        raise ValueError("grid must contain at most 16,000,000 cells")
    side = int(round(ratio))
    if side < 1 or not np.isclose(ratio, side, rtol=1e-10, atol=1e-8):
        raise ValueError("roi_size / resolution must be a positive integer")
    if side * side > _MAX_GRID_CELLS:
        raise ValueError("grid must contain at most 16,000,000 cells")
    centres = (np.arange(side, dtype=np.float64) + 0.5) * resolution - roi_size / 2
    x, y = np.meshgrid(centres, -centres)
    local_xy = np.stack((x, y), axis=-1)
    metadata = {
        "roi_size": roi_size,
        "resolution": resolution,
        "shape": (side, side),
        "height": side,
        "width": side,
        "x_min": -roi_size / 2,
        "x_max": roi_size / 2,
        "y_min": -roi_size / 2,
        "y_max": roi_size / 2,
        "column_axis": "+x",
        "row_axis": "-y",
        "coordinate_convention": "x forward, y left, z up; cell centres",
    }
    return local_xy, metadata


def project_ground_to_camera(
    local_xy: np.ndarray,
    global_from_bev: np.ndarray,
    camera_to_global: np.ndarray,
    K: np.ndarray,
    image_size: tuple[int, int],
    min_depth: float = 0.5,
    max_distance: float = 80.0,
) -> dict[str, np.ndarray]:
    """Project a horizontal BEV grid and select nearest camera pixel centres.

    image_size is (width, height). Invalid pixel_xy is -1 and invalid uv is
    NaN. Depth (camera z), world-space slant distance and incidence cosine
    are supplied for every cell, including cells outside the image.
    """
    xy = np.asarray(local_xy, dtype=np.float64)
    if xy.ndim != 3 or xy.shape[-1] != 2 or min(xy.shape[:2]) < 1:
        raise ValueError("local_xy must have nonempty shape HxWx2")
    if not np.all(np.isfinite(xy)):
        raise ValueError("local_xy must contain finite coordinates")
    bev = _rigid_matrix(global_from_bev, "global_from_bev")
    if not np.allclose(bev[2, :3], [0, 0, 1], rtol=0, atol=1e-8):
        raise ValueError("global_from_bev must define a horizontal z-up plane")
    camera = _rigid_matrix(camera_to_global, "camera_to_global")
    intrinsics, _ = _intrinsics(K)
    size = np.asarray(image_size)
    if size.shape != (2,) or size.dtype.kind not in "iu" or np.any(size <= 0):
        raise ValueError("image_size must contain positive integer (width, height)")
    if np.any(size > np.iinfo(np.int32).max):
        raise ValueError("image dimensions must fit int32 pixel coordinates")
    min_depth, max_distance = _limits(min_depth, max_distance)

    local = np.concatenate((xy, np.zeros((*xy.shape[:2], 1))), axis=-1)
    world = local @ bev[:3, :3].T + bev[:3, 3]
    displacement = world - camera[:3, 3]
    camera_points = displacement @ camera[:3, :3]
    depth = camera_points[..., 2]
    distance = np.linalg.norm(displacement, axis=-1)
    homogeneous = camera_points @ intrinsics.T
    uv = np.full(xy.shape, np.nan, dtype=np.float64)
    projectable = np.abs(homogeneous[..., 2]) > _PARALLEL_EPS
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.divide(homogeneous[..., :2], homogeneous[..., 2, None], out=uv,
                  where=projectable[..., None])
        rounded = np.floor(uv + 0.5)
    valid = (
        (depth >= min_depth) & (distance <= max_distance)
        & np.isfinite(depth) & np.isfinite(distance)
        & np.all(np.isfinite(uv), axis=-1) & projectable
        & (rounded[..., 0] >= 0) & (rounded[..., 0] < size[0])
        & (rounded[..., 1] >= 0) & (rounded[..., 1] < size[1])
    )
    pixel_xy = np.full(xy.shape, -1, dtype=np.int32)
    pixel_xy[valid] = rounded[valid].astype(np.int32)
    uv[~valid] = np.nan
    incidence = np.zeros(xy.shape[:2], dtype=np.float64)
    np.divide(np.abs(displacement[..., 2]), distance, out=incidence, where=distance > 0)
    return {
        "pixel_xy": pixel_xy,
        "uv": uv,
        "depth": depth.astype(np.float32),
        "slant_distance": distance.astype(np.float32),
        "valid": valid,
        "incidence": incidence.astype(np.float32),
    }


def pixels_to_ground(
    pixels: np.ndarray,
    camera_to_global: np.ndarray,
    K: np.ndarray,
    plane_z: float = 0.0,
    min_depth: float = 0.5,
    max_distance: float = 80.0,
) -> dict[str, np.ndarray]:
    """Intersect Nx2 calibrated pixel-centre rays with global z = plane_z.

    Parallel, upward/nonintersecting, near and distant rays are invalid.
    points_global, depth and slant_distance are NaN for invalid rays. Depth is
    camera z, rather than Euclidean ray length. All numeric outputs are float64.
    """
    pixels = np.asarray(pixels, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.all(np.isfinite(pixels)):
        raise ValueError("pixels must have finite shape Nx2")
    camera = _rigid_matrix(camera_to_global, "camera_to_global")
    _, inverse = _intrinsics(K)
    plane_z = _scalar(plane_z, "plane_z")
    min_depth, max_distance = _limits(min_depth, max_distance)
    camera_rays = np.column_stack((pixels, np.ones(len(pixels)))) @ inverse.T
    ray_z = camera_rays[:, 2].copy()
    ray_valid = np.isfinite(ray_z) & (np.abs(ray_z) > _PARALLEL_EPS)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.divide(camera_rays, ray_z[:, None], out=camera_rays, where=ray_valid[:, None])
    world_rays = camera_rays @ camera[:3, :3].T
    denominator = world_rays[:, 2]
    nonparallel = np.isfinite(denominator) & (np.abs(denominator) > _PARALLEL_EPS)
    depth = np.full(len(pixels), np.nan, dtype=np.float64)
    np.divide(plane_z - camera[2, 3], denominator, out=depth, where=ray_valid & nonparallel)
    with np.errstate(invalid="ignore", over="ignore"):
        offsets = depth[:, None] * world_rays
        distance = np.linalg.norm(offsets, axis=-1)
    valid = (ray_valid & nonparallel & np.isfinite(depth) & (depth >= min_depth)
             & np.isfinite(distance) & (distance <= max_distance))
    points = np.full((len(pixels), 3), np.nan, dtype=np.float64)
    points[valid] = camera[:3, 3] + offsets[valid]
    points[valid, 2] = plane_z
    depth[~valid] = np.nan
    distance[~valid] = np.nan
    return {"points_global": points, "valid": valid, "depth": depth,
            "slant_distance": distance}

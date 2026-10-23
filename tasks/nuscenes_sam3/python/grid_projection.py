"""Sample camera semantics on BEV cells of the assumed global Z=0 plane.

在假定的全局 Z=0 平面上，对 BEV 单元格内的相机语义像素进行采样。
"""
from __future__ import annotations

from numbers import Integral, Real

import numpy as np

from tasks.flat_ground.python.geometry import (
    bev_ground_grid, ego_yaw_frame, project_ground_to_camera,
)

# Bound temporary projected points, including all subcell samples, per row chunk.
_MAX_CHUNK_SAMPLES = 131_072


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    if not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return int(value)


def project_ground_cells(cameras, masks, roi_size=100., resolution=.2,
                         plane_ids=(1, 2), num_classes=7, samples_per_axis=3,
                         max_distance=80.):
    """Return semantic and evidence grids from aligned camera records and masks.

    Records use ``camera``, ``ego_pose``, ``camera_to_global`` and ``intrinsic``
    as returned by CameraFrames.cameras. Masks are full-resolution 2D integer
    arrays in the same order. CAM_FRONT supplies the ego yaw/XY reference when
    present; otherwise the first camera does. Plane height is always global 0.
    Each cell receives S*S uniformly spaced subcell-centre samples. Repeated
    pixels within a cell/camera contribute once; distinct cameras vote separately.
    Only nonzero plane_ids vote; ties choose the smallest class ID. Conflict
    means at least two eligible class IDs were observed, even without a tie.

    输入为对齐的相机记录和二维整数语义图。每格均匀采样 S*S 个子格中心；
    同一格、同一相机的重复像素只计一次。仅指定的非零平面类别投票，平票取
    最小类别编号；冲突表示出现多个有效类别。固定使用全局 Z=0，不使用雷达或真值。
    """
    num_classes = _integer(num_classes, "num_classes", 2, 255)
    samples_per_axis = _integer(samples_per_axis, "samples_per_axis", 1, 8)
    if (isinstance(max_distance, (bool, np.bool_)) or not isinstance(max_distance, Real)
            or not np.isfinite(max_distance) or max_distance <= 0):
        raise ValueError("max_distance must be finite and positive")
    max_distance = float(max_distance)
    try:
        plane_ids = tuple(plane_ids)
    except TypeError as error:
        raise ValueError("plane_ids must be a nonempty sequence of class IDs") from error
    if not plane_ids:
        raise ValueError("plane_ids must not be empty")
    plane_ids = tuple(_integer(value, "plane_ids", 1, num_classes - 1) for value in plane_ids)
    if len(set(plane_ids)) != len(plane_ids):
        raise ValueError("plane_ids must be unique")
    plane_ids = tuple(sorted(plane_ids))
    if not isinstance(cameras, (list, tuple)) or not 1 <= len(cameras) <= 255:
        raise ValueError("cameras must be a nonempty list of at most 255 camera records")
    if not isinstance(masks, (list, tuple)) or len(cameras) != len(masks):
        raise ValueError("masks must be aligned with cameras")
    names = []
    for camera, mask in zip(cameras, masks):
        if not isinstance(camera, dict):
            raise ValueError("camera records must be dictionaries")
        name = camera.get("camera")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("camera names must be nonempty and unique")
        names.append(name)
        if not isinstance(mask, np.ndarray) or mask.ndim != 2 or not mask.size:
            raise ValueError("masks must be nonempty 2D numpy arrays")
        if mask.dtype.kind not in "ui" or np.any(mask < 0) or np.any(mask >= num_classes):
            raise ValueError("masks must contain integer class IDs in [0, num_classes)")
        height, width = mask.shape
        if "image_size_wh" in camera and tuple(camera["image_size_wh"]) != (width, height):
            raise ValueError("mask dimensions do not match image_size_wh")
        if (("width" in camera and camera["width"] != width)
                or ("height" in camera and camera["height"] != height)):
            raise ValueError("mask dimensions do not match camera width/height")
        for key in ("ego_pose", "camera_to_global", "intrinsic"):
            if key not in camera:
                raise ValueError(f"camera record requires {key}")
        # Validate every calibration, even if it cannot see the ground at all.
        ego_yaw_frame(camera["ego_pose"], 0.)
        project_ground_to_camera(np.zeros((1, 1, 2)), np.eye(4),
                                 camera["camera_to_global"], camera["intrinsic"],
                                 (width, height), max_distance=max_distance)
    reference_index = names.index("CAM_FRONT") if "CAM_FRONT" in names else 0
    global_from_bev = ego_yaw_frame(cameras[reference_index]["ego_pose"], 0.)
    xy, grid = bev_ground_grid(roi_size, resolution)
    height, width = xy.shape[:2]
    shape = (height, width)
    result = {"labels": np.zeros(shape, dtype=np.uint8),
              "camera_count": np.zeros(shape, dtype=np.uint8),
              "sample_count": np.zeros(shape, dtype=np.uint32),
              "support_count": np.zeros(shape, dtype=np.uint32),
              "conflict": np.zeros(shape, dtype=np.bool_),
              "global_from_bev": global_from_bev}
    offsets = ((np.arange(samples_per_axis) + .5) / samples_per_axis - .5) * float(resolution)
    dx, dy = np.meshgrid(offsets, offsets)
    offsets = np.column_stack((dx.ravel(), dy.ravel()))
    rows_per_chunk = max(1, _MAX_CHUNK_SAMPLES // (width * len(offsets)))
    eligible = np.zeros(256, dtype=np.bool_)
    eligible[list(plane_ids)] = True
    for start in range(0, height, rows_per_chunk):
        stop = min(height, start + rows_per_chunk)
        cell_count = (stop - start) * width
        # The geometry helper accepts HxWx2: here axes are cells and subpoints.
        points = xy[start:stop].reshape(cell_count, 1, 2) + offsets[None, :, :]
        accepted_votes = []
        for camera, mask in zip(cameras, masks):
            geometry = project_ground_to_camera(
                points, global_from_bev, camera["camera_to_global"], camera["intrinsic"],
                (mask.shape[1], mask.shape[0]), max_distance=max_distance)
            valid = geometry["valid"]
            result["camera_count"][start:stop] += valid.any(axis=1).reshape(stop - start, width)
            cells, _ = np.nonzero(valid)
            pixels = geometry["pixel_xy"][valid]
            if not len(cells):
                continue
            # Sort the pair instead of multiplying by image area (which could
            # overflow for large dimensions). Deduplication is local to a camera.
            pixel_ids = pixels[:, 1].astype(np.int64) * mask.shape[1] + pixels[:, 0]
            order = np.lexsort((pixel_ids, cells))
            cells, pixel_ids, pixels = cells[order], pixel_ids[order], pixels[order]
            unique = np.r_[True, (cells[1:] != cells[:-1]) | (pixel_ids[1:] != pixel_ids[:-1])]
            cells, pixels = cells[unique], pixels[unique]
            counts = np.bincount(cells, minlength=cell_count).astype(np.uint32)
            result["sample_count"][start:stop] += counts.reshape(stop - start, width)
            classes = mask[pixels[:, 1], pixels[:, 0]].astype(np.uint8)
            accepted = eligible[classes]
            cells, classes = cells[accepted], classes[accepted]
            counts = np.bincount(cells, minlength=cell_count).astype(np.uint32)
            result["support_count"][start:stop] += counts.reshape(stop - start, width)
            accepted_votes.append(cells * 256 + classes)
        if accepted_votes:
            encoded, counts = np.unique(np.concatenate(accepted_votes), return_counts=True)
            cells, classes = encoded // 256, encoded % 256
            order = np.lexsort((classes, -counts, cells))
            ordered_cells = cells[order]
            first = np.r_[True, ordered_cells[1:] != ordered_cells[:-1]] if len(order) else np.zeros(0, bool)
            chosen = order[first]
            result["labels"][start:stop].flat[cells[chosen]] = classes[chosen].astype(np.uint8)
            distinct = np.bincount(cells, minlength=cell_count)
            result["conflict"][start:stop] = (distinct > 1).reshape(stop - start, width)
    result["metadata"] = {
        **grid, "plane_z_global_m": 0., "reference_camera": names[reference_index],
        "bev_frame": "ego XY and yaw; horizontal global Z=0; columns +x, rows -y",
        "sampling": "uniform subcell centres; nearest calibrated pixel centre",
        "samples_per_axis": samples_per_axis, "samples_per_cell": samples_per_axis ** 2,
        "deduplication": "unique pixel within each cell and camera; cameras remain independent",
        "plane_ids": list(plane_ids), "num_classes": num_classes,
        "fusion": "majority over eligible unique pixel samples; ties smallest class ID",
        "abstention": "zero and classes outside plane_ids; no evidence yields label 0",
        "conflict": "two or more distinct eligible classes in a cell",
        "camera_count": "cameras with at least one geometrically valid sample",
        "sample_count": "unique geometrically valid (camera, pixel) samples per cell",
        "support_count": "unique samples with labels in plane_ids per cell",
        "max_camera_distance_m": max_distance, "min_camera_depth_m": .5,
        "cameras": names, "source": "camera semantic masks only; no LiDAR or ground truth",
        "limitations": "assumed flat plane; no occlusion, slope or surface-height reconstruction",
    }
    return result

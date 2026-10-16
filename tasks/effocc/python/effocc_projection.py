"""Exact pixel-centre raycasting of an XYZ occupancy grid, using batched torch DDA.

No image augmentation or ego-frame selection is performed here. ``camera_to_grid``
transforms camera coordinates into the metric frame of ``point_cloud_range``;
the caller must compose the appropriate camera-time and grid-reference poses.
Only ``free_id`` is transparent (class 0, ``others``, is an ordinary solid).
"""
from __future__ import annotations

from numbers import Integral

import numpy as np


def _finite_array(value, shape, name):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric array of shape {shape}") from error
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite numeric array of shape {shape}")
    return result


def raycast_occupancy(
    semantics: np.ndarray,
    camera_to_grid: np.ndarray,
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
    point_cloud_range: np.ndarray,
    voxel_size: np.ndarray,
    min_depth: float = .1,
    device: str = "cpu",
    chunk_size: int = 65536,
    free_id: int = 17,
) -> dict:
    """Return the first occupied voxel intersected by every original-image pixel.

    Rays are ``K^-1 @ [u + .5, v + .5, 1]``. They are *not* unit-normalized:
    the ray parameter, and hence returned depth, is camera Z in metres. K must
    therefore have the usual intrinsic last row [0, 0, 1]. Slab intersection
    clips the ray to the grid and ``min_depth`` before Amanatides-Woo traversal.
    All axes tied at a boundary advance together; zero-length intersections
    with edges/corners never count as occupied intervals. At an exact boundary
    the cell in the ray's forward direction owns the sample.

    ``semantics`` is uint8 [X,Y,Z], with IDs 0..17, of any positive shape.
    Grid extents must match shape * voxel_size. Outputs are NumPy HxW arrays:
    labels uint8 (255 for no hit), depth float32 (0 for no hit), voxel_index
    int64 (-1 for no hit, otherwise ``(x * ny + y) * nz + z``), and JSON-friendly
    metadata. The first occupied voxel may already contain the near-plane
    point; its reported entry depth is then exactly ``min_depth``.

    CUDA uses float32 geometry, CPU float64 geometry; both return float32 depth.
    Work is vectorized over a tile of pixels and active rays, with no Python
    loop over individual rays or voxels. Chunk size controls temporary memory;
    262144 is a useful larger tile for full-resolution camera images.
    """
    if (not isinstance(semantics, np.ndarray) or semantics.ndim != 3
            or any(size <= 0 for size in semantics.shape)
            or semantics.dtype != np.uint8 or np.any(semantics > 17)):
        raise ValueError("semantics must be a nonempty uint8 XYZ array with class IDs 0..17")
    if isinstance(free_id, bool) or not isinstance(free_id, Integral) or not 0 <= free_id <= 17:
        raise ValueError("free_id must be an integer in 0..17")
    pose = _finite_array(camera_to_grid, (4, 4), "camera_to_grid")
    if not np.allclose(pose[3], [0, 0, 0, 1], rtol=0, atol=1e-8):
        raise ValueError("camera_to_grid must have homogeneous last row [0,0,0,1]")
    if np.linalg.matrix_rank(pose[:3, :3]) != 3:
        raise ValueError("camera_to_grid must have an invertible linear transform")
    calibration = _finite_array(intrinsic, (3, 3), "intrinsic")
    try:
        inverse_intrinsic = np.linalg.inv(calibration)
    except np.linalg.LinAlgError as error:
        raise ValueError("intrinsic must be invertible") from error
    if not np.isfinite(inverse_intrinsic).all():
        raise ValueError("intrinsic must have a finite inverse")
    if not np.allclose(calibration[2], [0, 0, 1], rtol=0, atol=1e-8):
        raise ValueError("intrinsic must have last row [0,0,1] for camera-Z depth")
    bounds = _finite_array(point_cloud_range, (6,), "point_cloud_range")
    spacing = _finite_array(voxel_size, (3,), "voxel_size")
    if np.any(spacing <= 0) or np.any(bounds[3:] <= bounds[:3]):
        raise ValueError("voxel_size and point_cloud_range extents must be positive")
    shape = np.asarray(semantics.shape, dtype=np.int64)
    if not np.allclose(bounds[3:] - bounds[:3], shape * spacing, rtol=1e-6, atol=1e-7):
        raise ValueError("point_cloud_range must match semantics XYZ shape * voxel_size")
    if (not isinstance(image_size, (tuple, list)) or len(image_size) != 2
            or any(isinstance(n, bool) or not isinstance(n, Integral) or n <= 0 for n in image_size)):
        raise ValueError("image_size must be positive integer (width, height)")
    if (isinstance(min_depth, (bool, np.bool_)) or not np.isscalar(min_depth)
            or not np.isfinite(min_depth) or min_depth <= 0):
        raise ValueError("min_depth must be finite and positive")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    import torch

    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    width, height = map(int, image_size)
    count = width * height
    dtype = torch.float32 if target.type == "cuda" else torch.float64
    # Compose source geometry in double before transferring GPU float32 values.
    # Elementwise ray construction below also avoids TF32 matrix multiplication.
    projection = (pose[:3, :3] @ inverse_intrinsic) / spacing[:, None]
    origin_array = (pose[:3, 3] - bounds[:3]) / spacing
    with torch.inference_mode(), torch.autocast(device_type=target.type, enabled=False):
        transform = torch.as_tensor(projection, dtype=dtype, device=target)
        origin = torch.as_tensor(origin_array, dtype=dtype, device=target)
        extent = torch.as_tensor(shape, dtype=dtype, device=target)
        shape_tensor = torch.as_tensor(shape, dtype=torch.int64, device=target)
        # Copy once: no caller array is modified, including noncontiguous grids.
        grid = torch.tensor(np.ascontiguousarray(semantics).reshape(-1), device=target)
        labels = torch.full((count,), 255, dtype=torch.uint8, device=target)
        depths = torch.zeros(count, dtype=torch.float32, device=target)
        indices = torch.full((count,), -1, dtype=torch.int64, device=target)
        epsilon = torch.finfo(dtype).eps
        max_steps = int(shape.sum()) + 3
        for start in range(0, count, int(chunk_size)):
            pixels = torch.arange(start, min(start + chunk_size, count), device=target)
            u = (pixels % width).to(dtype) + .5
            v = torch.div(pixels, width, rounding_mode="floor").to(dtype) + .5
            direction = (u[:, None] * transform[:, 0]
                         + v[:, None] * transform[:, 1] + transform[:, 2])
            parallel = direction == 0
            safe_direction = torch.where(parallel, torch.ones_like(direction), direction)
            t0 = -origin / safe_direction
            t1 = (extent - origin) / safe_direction
            slab_near = torch.where(parallel, -torch.inf, torch.minimum(t0, t1))
            slab_far = torch.where(parallel, torch.inf, torch.maximum(t0, t1))
            near = slab_near.amax(dim=1).clamp_min(float(min_depth))
            far = slab_far.amin(dim=1)
            # A ray lying on the outer maximum face owns no grid cell.
            parallel_outside = (parallel & ((origin < 0) | (origin >= extent))).any(dim=1)
            valid = (~parallel_outside) & (far > near) & torch.isfinite(near)
            active_ids = torch.nonzero(valid, as_tuple=True)[0]
            pixels = pixels.index_select(0, active_ids)
            direction = direction.index_select(0, active_ids)
            near, far = near.index_select(0, active_ids), far.index_select(0, active_ids)
            if pixels.numel() == 0:
                continue
            position = origin + near[:, None] * direction
            position = torch.minimum(torch.maximum(position, torch.zeros_like(position)), extent)
            # Undo roundoff at entry faces, then choose the forward cell. This
            # changes ownership only; near retains the unbiased entry depth.
            rounded = position.round()
            tolerance = 2 * epsilon * torch.maximum(torch.ones_like(position), position.abs())
            position = torch.where((position - rounded).abs() <= tolerance, rounded, position)
            cells = torch.floor(position).to(torch.int64)
            negative_boundary = (direction < 0) & (position == torch.floor(position))
            cells -= negative_boundary.to(torch.int64)
            step = torch.sign(direction).to(torch.int64)
            parallel = direction == 0
            safe_direction = torch.where(parallel, torch.ones_like(direction), direction)
            boundary = cells + (step > 0).to(torch.int64)
            next_boundary = torch.where(parallel, torch.inf, (boundary - origin) / safe_direction)
            delta = torch.where(parallel, torch.inf, 1 / direction.abs())
            for _ in range(max_steps):
                # All active rays own an in-bounds, positive-length interval.
                inside = ((cells >= 0) & (cells < shape_tensor)).all(dim=1)
                interval_end = torch.minimum(next_boundary.amin(dim=1), far)
                valid = inside & (interval_end > near)
                flat = ((cells[:, 0].clamp(0, int(shape[0]) - 1) * int(shape[1])
                         + cells[:, 1].clamp(0, int(shape[1]) - 1)) * int(shape[2])
                        + cells[:, 2].clamp(0, int(shape[2]) - 1))
                classes = grid[flat]
                hit = valid & (classes != free_id)
                hit_ids = torch.nonzero(hit, as_tuple=True)[0]
                hit_pixels = pixels.index_select(0, hit_ids)
                labels[hit_pixels] = classes.index_select(0, hit_ids)
                depths[hit_pixels] = near.index_select(0, hit_ids).to(torch.float32)
                indices[hit_pixels] = flat.index_select(0, hit_ids)
                crossing = next_boundary.amin(dim=1)
                live = inside & ~hit & (crossing < far)
                # Compute each boolean selection once. Repeating boolean
                # indexing for every state tensor causes repeated CUDA nonzero
                # synchronizations, which dominate long transparent traversals.
                live_ids = torch.nonzero(live, as_tuple=True)[0]
                pixels = pixels.index_select(0, live_ids)
                if pixels.numel() == 0:
                    break
                cells, step = cells.index_select(0, live_ids), step.index_select(0, live_ids)
                next_boundary = next_boundary.index_select(0, live_ids)
                delta = delta.index_select(0, live_ids)
                near, far = crossing.index_select(0, live_ids), far.index_select(0, live_ids)
                # Simultaneous crossings skip cells touched only at a point.
                tie_tolerance = 2 * epsilon * torch.maximum(torch.ones_like(near), near.abs())
                tied = (next_boundary - near[:, None]).abs() <= tie_tolerance[:, None]
                cells += tied.to(torch.int64) * step
                # Multiplying infinity by zero would create NaNs for parallel axes.
                next_boundary = torch.where(tied, next_boundary + delta, next_boundary)
            else:
                raise RuntimeError("DDA exceeded shape.sum()+3 steps; geometry is inconsistent")
        result = {
            "labels": labels.cpu().numpy().reshape(height, width),
            "depth": depths.cpu().numpy().reshape(height, width),
            "voxel_index": indices.cpu().numpy().reshape(height, width),
        }
    result["metadata"] = {
        "method": "pixel_center_exact_voxel_dda", "axis_order": "XYZ",
        "pixel_centers": "K^-1 [u+0.5,v+0.5,1]", "depth_units": "camera_z_metres",
        "image_size_wh": [width, height], "grid_shape_xyz": shape.tolist(),
        "point_cloud_range": bounds.tolist(), "voxel_size": spacing.tolist(),
        "min_depth": float(min_depth), "free_id": int(free_id),
        "no_hit_label": 255, "no_hit_depth": 0, "no_hit_voxel_index": -1,
        "voxel_index_order": "(x * ny + y) * nz + z",
        "device": str(target), "geometry_dtype": str(dtype).removeprefix("torch."),
        "chunk_size": int(chunk_size), "max_steps_per_ray": max_steps,
    }
    return result

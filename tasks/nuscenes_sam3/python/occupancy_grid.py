"""Sparse 3D occupancy grids built from complete aligned point clouds."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


_VALID_CLASS_IDS = range(1, 255)


def _validate_resolution(resolution: float) -> float:
    if isinstance(resolution, (bool, np.bool_)):
        raise ValueError("resolution must be finite and positive")
    try:
        value = float(resolution)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("resolution must be finite and positive") from error
    if not np.isfinite(value) or value <= 0:
        raise ValueError("resolution must be finite and positive")
    return value


def voxelize_points(xyz: np.ndarray, resolution: float = .5) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted unique integer voxel keys and per-voxel point counts.

    Coordinates are assigned with ``floor(xyz / resolution)``. No labels,
    sampling, or semantic filtering are applied.
    """
    resolution = _validate_resolution(resolution)
    points = np.asarray(xyz)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")
    try:
        points = points.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("xyz must contain finite numeric coordinates") from error
    if not np.isfinite(points).all():
        raise ValueError("xyz must contain finite numeric coordinates")
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.int64), np.empty((0,), dtype=np.uint64)

    scaled = np.floor(points / resolution)
    if np.any(scaled < -(2.0 ** 63)) or np.any(scaled >= 2.0 ** 63):
        raise ValueError("voxel indices exceed int64 range")
    keys = scaled.astype(np.int64)
    unique, counts = np.unique(keys, axis=0, return_counts=True)
    return unique, counts.astype(np.uint64, copy=False)


def _validate_labels(labels, point_count: int) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 1 or len(values) != point_count:
        raise ValueError("semantic labels must be a one-dimensional array aligned with xyz")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("semantic labels must contain integer class ids")
    if np.any(values < 0) or np.any(values > 254):
        raise ValueError("semantic class ids must be in 0..254")
    return values.astype(np.uint8, copy=False)


def _semantic_frame(xyz, labels, resolution):
    points = np.asarray(xyz)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")
    labels = _validate_labels(labels, len(points))
    assigned = (labels >= 1) & (labels <= 254)
    indices, counts = voxelize_points(points[assigned], resolution)
    voxel_keys = np.floor(points[assigned].astype(np.float64) / resolution).astype(np.int64)
    assigned_labels = labels[assigned]
    votes: dict[tuple[int, int, int], dict[int, int]] = {}
    for key, class_id in zip(voxel_keys, assigned_labels):
        voxel = tuple(int(value) for value in key)
        by_class = votes.setdefault(voxel, {})
        class_key = int(class_id)
        by_class[class_key] = by_class.get(class_key, 0) + 1
    class_ids = []
    sparse_votes = []
    for key_index, key in enumerate(indices):
        vote = votes[tuple(int(value) for value in key)]
        winner = min(vote, key=lambda class_id: (-vote[class_id], class_id))
        class_ids.append(winner)
        sparse_votes.extend((key_index, class_id, count) for class_id, count in sorted(vote.items()))
    return _frame_record(indices, counts, np.asarray(class_ids, dtype=np.uint8), sparse_votes)


def _frame_record(indices: np.ndarray, counts: np.ndarray, class_ids=None, sparse_votes=None) -> dict:
    point_count = int(counts.sum(dtype=np.uint64))
    record = {
        "indices": indices.tolist(),
        "point_counts": [int(count) for count in counts],
        "voxel_count": int(len(indices)),
        "point_count": point_count,
    }
    if class_ids is not None:
        record["class_ids"] = np.asarray(class_ids, dtype=np.uint8).tolist()
        record["vote_counts"] = [list(row) for row in (sparse_votes or [])]
    return record


def build_occupancy_grid(frame_xyz: Iterable[np.ndarray], resolution: float = .5, *, semantic: bool = False) -> dict:
    """Build sparse occupancy, optionally retaining votes for assigned classes.

    In semantic mode each input item is ``(xyz, labels)``. Only labels 1..254
    occupy voxels; labels are reduced by point-count majority, with smaller IDs
    winning ties. Fused classes are voted from all supporting points.
    """
    resolution = _validate_resolution(resolution)
    frames = []
    fused_counts: dict[tuple[int, int, int], int] = {}
    fused_votes: dict[tuple[int, int, int], dict[int, int]] = {}
    total_points = 0
    for frame_data in frame_xyz:
        if semantic:
            if not isinstance(frame_data, (tuple, list)) or len(frame_data) != 2:
                raise ValueError("semantic frames must be (xyz, labels) pairs")
            xyz, labels = frame_data
            record = _semantic_frame(xyz, labels, resolution)
        else:
            record = _frame_record(*voxelize_points(frame_data, resolution))
        frames.append(record)
        total_points += record["point_count"]
        for key, count in zip(record["indices"], record["point_counts"]):
            voxel = tuple(key)
            fused_counts[voxel] = fused_counts.get(voxel, 0) + count
        if semantic:
            for voxel_index, class_id, class_count in record["vote_counts"]:
                voxel = tuple(record["indices"][voxel_index])
                vote = fused_votes.setdefault(voxel, {})
                vote[class_id] = vote.get(class_id, 0) + class_count

    fused_indices = np.asarray(sorted(fused_counts), dtype=np.int64).reshape((-1, 3))
    fused_point_counts = np.asarray([fused_counts[tuple(key)] for key in fused_indices], dtype=np.uint64)
    if semantic:
        fused_class_ids = []
        fused_sparse_votes = []
        for index, key in enumerate(fused_indices):
            vote = fused_votes[tuple(int(value) for value in key)]
            fused_class_ids.append(min(vote, key=lambda class_id: (-vote[class_id], class_id)))
            fused_sparse_votes.extend((index, class_id, count) for class_id, count in sorted(vote.items()))
        fused = _frame_record(fused_indices, fused_point_counts, np.asarray(fused_class_ids, dtype=np.uint8), fused_sparse_votes)
    else:
        fused = _frame_record(fused_indices, fused_point_counts)
    if sum(fused["point_counts"]) != total_points or any(
        sum(frame["point_counts"]) != frame["point_count"] for frame in frames
    ):
        raise RuntimeError("occupancy point accounting invariant failed")
    metadata = {
        "representation": "sparse_voxel_keys",
        "coordinate_frame": "input_xyz",
        "occupied_only": True,
        "free_space_raytracing": False,
        "semantic_filtering": bool(semantic),
        "semantic_source": "aligned_frame_labels_class_ids_1_254" if semantic else None,
        "class_reduction": "point_count_majority_tie_smallest_id" if semantic else None,
        "fusion_vote_source": "raw_semantic_point_counts" if semantic else None,
        "frame_count": len(frames),
        "point_count": total_points,
        "voxel_count": fused["voxel_count"],
    }
    return {"resolution": resolution, "origin": [0, 0, 0], "frames": frames, "fused": fused, "metadata": metadata}


def save_occupancy_grid(scene_dir, frame_xyz: Iterable[np.ndarray], resolution: float = .5, *, semantic: bool = False,
                        basename: str = "occupancy_grid", metadata: dict | None = None) -> dict:
    """Save sparse arrays and compact JSON; 中文：保存稀疏数组及精简元数据。

    ``basename`` names an independent layer; metadata can describe point selection
    performed by the caller without adding semantic class arrays to that layer.
    ``basename`` 用于独立图层命名；metadata 可说明调用方的点筛选方式。
    """
    if not isinstance(basename, str) or not basename or basename in (".", "..") or any(char in basename for char in '/\\'):
        raise ValueError("basename must be a nonempty file basename")
    output_dir = Path(scene_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_occupancy_grid(frame_xyz, resolution, semantic=semantic)
    if metadata is not None:
        payload["metadata"].update(metadata)
    arrays = {
        "fused_indices": np.asarray(payload["fused"]["indices"], dtype=np.int64).reshape((-1, 3)),
        "fused_point_counts": np.asarray(payload["fused"]["point_counts"], dtype=np.uint64),
    }
    if semantic:
        arrays["fused_class_ids"] = np.asarray(payload["fused"]["class_ids"], dtype=np.uint8)
        arrays["fused_vote_counts"] = np.asarray(payload["fused"]["vote_counts"], dtype=np.uint64).reshape((-1, 3))
    for frame_index, frame in enumerate(payload["frames"]):
        arrays[f"frame_{frame_index}_indices"] = np.asarray(frame["indices"], dtype=np.int64).reshape((-1, 3))
        arrays[f"frame_{frame_index}_counts"] = np.asarray(frame["point_counts"], dtype=np.uint64)
        if semantic:
            arrays[f"frame_{frame_index}_class_ids"] = np.asarray(frame["class_ids"], dtype=np.uint8)
            arrays[f"frame_{frame_index}_vote_counts"] = np.asarray(frame["vote_counts"], dtype=np.uint64).reshape((-1, 3))
    np.savez_compressed(output_dir / f"{basename}.npz", **arrays)
    metadata = {
        "resolution": payload["resolution"],
        "origin": payload["origin"],
        "metadata": payload["metadata"],
        "frames": [{key: frame[key] for key in (("voxel_count", "point_count", "class_ids", "vote_counts") if semantic else ("voxel_count", "point_count"))} for frame in payload["frames"]],
        "fused": {key: payload["fused"][key] for key in (("voxel_count", "point_count", "class_ids", "vote_counts") if semantic else ("voxel_count", "point_count"))},
    }
    (output_dir / f"{basename}.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _validate_plane_ids(plane_ids) -> tuple[int, ...]:
    if isinstance(plane_ids, (str, bytes)):
        raise ValueError("plane_ids must be a nonempty sequence of unique class IDs")
    try:
        values = list(plane_ids)
    except TypeError as error:
        raise ValueError("plane_ids must be a nonempty sequence of unique class IDs") from error
    if not values:
        raise ValueError("plane_ids must not be empty")
    result = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("plane_ids must contain integer class IDs")
        if value < 1 or value > 254:
            raise ValueError("plane_ids class IDs must be in 1..254")
        result.append(int(value))
    if len(set(result)) != len(result):
        raise ValueError("plane_ids must be unique")
    return tuple(sorted(result))


def _validate_flat_input(occupancy, plane_z_scene):
    if not isinstance(occupancy, dict):
        raise ValueError("occupancy must be a mapping payload")
    resolution = _validate_resolution(occupancy.get("resolution"))
    origin = np.asarray(occupancy.get("origin"))
    if origin.ndim != 1 or len(origin) != 3:
        raise ValueError("occupancy origin must contain three finite coordinates")
    if isinstance(plane_z_scene, (bool, np.bool_)) or origin.dtype.kind not in "iuf":
        raise ValueError("occupancy origin and plane_z_scene must be finite numeric values")
    try:
        origin = origin.astype(np.float64, copy=False)
        z_scene = float(plane_z_scene)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("occupancy origin and plane_z_scene must be finite numeric values") from error
    if not np.isfinite(origin).all() or not np.isfinite(z_scene):
        raise ValueError("occupancy origin and plane_z_scene must be finite numeric values")
    metadata = occupancy.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("occupancy metadata must be a mapping")
    coordinate_frame = metadata.get("coordinate_frame")
    if coordinate_frame is not None and (not isinstance(coordinate_frame, str) or not coordinate_frame.strip()):
        raise ValueError("occupancy coordinate_frame metadata must be a nonempty string")
    return resolution, origin, z_scene, metadata


def _flat_record(record, name):
    if not isinstance(record, dict):
        raise ValueError(f"{name} must be a mapping record")
    indices = np.asarray(record.get("indices"))
    if indices.shape == (0,):
        indices = np.empty((0, 3), dtype=np.int64)
    if indices.ndim != 2 or indices.shape[1] != 3 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"{name} indices must have integer shape (N, 3)")
    if np.issubdtype(indices.dtype, np.unsignedinteger) and np.any(indices > np.iinfo(np.int64).max):
        raise ValueError(f"{name} indices exceed int64 range")
    indices = indices.astype(np.int64, copy=False)
    n = len(indices)
    if len({tuple(key) for key in indices.tolist()}) != n:
        raise ValueError(f"{name} voxel indices must be unique")
    classes = np.asarray(record.get("class_ids"))
    counts = np.asarray(record.get("point_counts"))
    if n == 0 and classes.shape == (0,):
        classes = np.empty(0, dtype=np.uint8)
    if n == 0 and counts.shape == (0,):
        counts = np.empty(0, dtype=np.uint64)
    if classes.ndim != 1 or len(classes) != n or not np.issubdtype(classes.dtype, np.integer):
        raise ValueError(f"{name} class_ids must be integer values aligned with indices")
    if np.any(classes < 0) or np.any(classes > 254):
        raise ValueError(f"{name} class_ids must be in 0..254")
    if counts.ndim != 1 or len(counts) != n or not np.issubdtype(counts.dtype, np.integer) or np.any(counts <= 0):
        raise ValueError(f"{name} point_counts must be positive integers aligned with indices")
    support = counts.astype(np.uint64, copy=False)
    if "vote_counts" in record:
        votes = np.asarray(record["vote_counts"])
        if votes.shape == (0,):
            votes = np.empty((0, 3), dtype=np.uint64)
        if votes.ndim != 2 or votes.shape[1] != 3 or not np.issubdtype(votes.dtype, np.integer):
            raise ValueError(f"{name} vote_counts must have integer shape (M, 3)")
        if np.any(votes < 0):
            raise ValueError(f"{name} vote_counts cannot contain negative values")
        votes = votes.astype(np.uint64, copy=False)
        if len(votes) and (np.any(votes[:, 0] >= n) or np.any(votes[:, 1] < 1) or
                           np.any(votes[:, 1] > 254) or np.any(votes[:, 2] == 0)):
            raise ValueError(f"{name} vote_counts contain invalid voxel IDs, class IDs, or counts")
        winner_support = np.zeros(n, dtype=np.uint64)
        vote_totals = [0] * n
        seen = set()
        for voxel_index, class_id, vote_count in votes:
            index, class_id, vote_count = int(voxel_index), int(class_id), int(vote_count)
            if (index, class_id) in seen:
                raise ValueError(f"{name} vote_counts must have unique voxel/class pairs")
            seen.add((index, class_id))
            vote_totals[index] += vote_count
            if class_id == int(classes[index]):
                winner_support[index] = vote_count
        if vote_totals != [int(count) for count in counts] or np.any(winner_support == 0):
            raise ValueError(f"{name} vote_counts must account for point_counts and winning classes")
        support = winner_support
    return indices, classes.astype(np.int64, copy=False), support


def _collapse_flat_record(record, plane_ids, name):
    indices, classes, support = _flat_record(record, name)
    allowed = np.isin(classes, plane_ids)
    by_xy = {}
    for key, class_id, count in zip(indices[allowed], classes[allowed], support[allowed]):
        xy = (int(key[0]), int(key[1]))
        cell = by_xy.setdefault(xy, {"total": 0, "votes": {}})
        cell["total"] += int(count)
        votes = cell["votes"]
        votes[int(class_id)] = votes.get(int(class_id), 0) + int(count)
    xy_keys = sorted(by_xy)
    class_ids, point_counts = [], []
    for xy in xy_keys:
        cell = by_xy[xy]
        class_ids.append(min(cell["votes"], key=lambda class_id: (-cell["votes"][class_id], class_id)))
        point_counts.append(cell["total"])
    return {"indices": [list(key) for key in xy_keys], "class_ids": class_ids,
            "point_counts": point_counts, "cell_count": len(xy_keys),
            "point_count": sum(point_counts)}


def flatten_surface_occupancy(occupancy, plane_ids, plane_z_scene=0.):
    """Collapse winning surface-class 3D voxels into sparse, support-weighted XY."""
    plane_ids = _validate_plane_ids(plane_ids)
    resolution, origin, z_scene, source_metadata = _validate_flat_input(occupancy, plane_z_scene)
    frames = occupancy.get("frames")
    if not isinstance(frames, (list, tuple)):
        raise ValueError("occupancy frames must be a sequence")
    fused = occupancy.get("fused")
    # Fused is deliberately collapsed from the fused 3D record, never from the
    # frame results: its authoritative 3D winners can differ after fusion.
    flat_frames = [_collapse_flat_record(frame, plane_ids, f"frames[{i}]") for i, frame in enumerate(frames)]
    flat_fused = _collapse_flat_record(fused, plane_ids, "fused")
    metadata = {
        "source": "3D occupancy winning class IDs",
        "surface_class_ids": list(plane_ids),
        "support_count_source": "winning-class vote_counts; legacy point_counts when vote_counts absent",
        "xy_reduction": "sum winning-class supports by XY and choose largest support, tie smallest class ID",
        "source_coordinate_frame": source_metadata.get("coordinate_frame"),
    }
    return {"resolution": resolution, "origin": origin[:2].tolist(), "plane_z_scene": z_scene,
            "frames": flat_frames, "fused": flat_fused, "metadata": metadata}


def save_flat_occupancy(scene_dir, payload) -> dict:
    """Save flat sparse numeric arrays and deterministic JSON metadata."""
    if not isinstance(payload, dict):
        raise ValueError("payload must be a mapping")
    resolution = _validate_resolution(payload.get("resolution"))
    origin = np.asarray(payload.get("origin"))
    if origin.shape != (2,):
        raise ValueError("flat occupancy origin must have two coordinates")
    if origin.dtype.kind not in "iuf" or isinstance(payload.get("plane_z_scene"), (bool, np.bool_)):
        raise ValueError("flat occupancy calibration must be finite numeric values")
    if not isinstance(payload.get("metadata"), dict):
        raise ValueError("flat occupancy metadata must be a mapping")
    try:
        origin = origin.astype(np.float64, copy=False)
        plane_z = float(payload.get("plane_z_scene"))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("flat occupancy calibration must be finite numeric values") from error
    if not np.isfinite(origin).all() or not np.isfinite(plane_z):
        raise ValueError("flat occupancy calibration must be finite numeric values")
    frames = payload.get("frames")
    if not isinstance(frames, (list, tuple)):
        raise ValueError("flat occupancy frames must be a sequence")

    def arrays_for(record, prefix):
        if not isinstance(record, dict):
            raise ValueError(f"{prefix} must be a mapping record")
        indices = np.asarray(record.get("indices"))
        classes = np.asarray(record.get("class_ids"))
        counts = np.asarray(record.get("point_counts"))
        if indices.shape == (0,):
            indices = np.empty((0, 2), dtype=np.int64)
        if indices.ndim != 2 or indices.shape[1] != 2 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"{prefix} indices must have integer shape (N, 2)")
        if np.issubdtype(indices.dtype, np.unsignedinteger) and np.any(indices > np.iinfo(np.int64).max):
            raise ValueError(f"{prefix} indices exceed int64 range")
        n = len(indices)
        if n == 0 and classes.shape == (0,):
            classes = np.empty(0, dtype=np.uint8)
        if n == 0 and counts.shape == (0,):
            counts = np.empty(0, dtype=np.uint64)
        if classes.shape != (n,) or not np.issubdtype(classes.dtype, np.integer) or np.any(classes < 1) or np.any(classes > 254):
            raise ValueError(f"{prefix} class_ids must contain IDs in 1..254 aligned with indices")
        if counts.shape != (n,) or not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
            raise ValueError(f"{prefix} point_counts must be nonnegative integers aligned with indices")
        order = np.lexsort((indices[:, 1], indices[:, 0]))
        indices = indices[order].astype(np.int64, copy=False)
        classes = classes[order].astype(np.uint8, copy=False)
        counts = counts[order].astype(np.uint64, copy=False)
        if n > 1 and np.any(np.all(indices[1:] == indices[:-1], axis=1)):
            raise ValueError(f"{prefix} indices must be unique")
        return {f"{prefix}_indices": indices, f"{prefix}_class_ids": classes,
                f"{prefix}_point_counts": counts}

    arrays = {}
    summaries = []
    for i, record in enumerate(frames):
        record_arrays = arrays_for(record, f"frame_{i}")
        arrays.update(record_arrays)
        summaries.append({"cell_count": len(record_arrays[f"frame_{i}_indices"]),
                          "point_count": int(record_arrays[f"frame_{i}_point_counts"].sum(dtype=np.uint64))})
    fused_arrays = arrays_for(payload.get("fused"), "fused")
    arrays.update(fused_arrays)
    fused_summary = {"cell_count": len(fused_arrays["fused_indices"]),
                     "point_count": int(fused_arrays["fused_point_counts"].sum(dtype=np.uint64))}
    output_dir = Path(scene_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "occupancy_flat.npz", **arrays)
    metadata = {"resolution": resolution, "origin": origin.tolist(), "plane_z_scene": plane_z,
                "frames": summaries, "fused": fused_summary, "metadata": payload.get("metadata", {})}
    (output_dir / "occupancy_flat.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return payload

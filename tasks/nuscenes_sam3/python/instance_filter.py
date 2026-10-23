"""Exact per-instance 3D point filtering using NumPy and the standard library.

Indices always refer to the original scan. SOR excludes only the query point
itself (other points at identical coordinates are valid zero-distance neighbors).
DBSCAN includes the query in min_samples, uses exact Euclidean radius tests, and
visits points in original-index order. No ground truth or semantic pooling is used.
"""

from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from itertools import product
import math
from numbers import Integral, Real

import numpy as np


_DEFAULTS = {
    "algorithm": "none", "sor_k": 16, "sor_std": 2.0,
    "dbscan_eps": 1.0, "dbscan_min_samples": 5,
    "cluster_policy": "significant", "min_cluster_size": 10,
    "cluster_ratio": 0.1,
}
_CELL_OFFSETS = tuple(product((-1, 0, 1), repeat=3))


def _integer(value, name, minimum=1, maximum=None):
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside the allowed range")
    return value


def _positive(value, name):
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def validate_filter(config=None):
    """Return a fresh normalized configuration, validating even unused options."""
    if config is not None and not isinstance(config, Mapping):
        raise ValueError("filter config must be a mapping or None")
    config = {} if config is None else dict(config)
    unknown = set(config) - set(_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown filter options: {sorted(unknown, key=str)}")
    result = dict(_DEFAULTS, **config)
    if result["algorithm"] not in ("none", "sor", "dbscan", "sor_dbscan"):
        raise ValueError("algorithm must be none, sor, dbscan or sor_dbscan")
    if result["cluster_policy"] not in ("largest", "significant"):
        raise ValueError("cluster_policy must be largest or significant")
    for key in ("sor_k", "dbscan_min_samples", "min_cluster_size"):
        result[key] = _integer(result[key], key)
    for key in ("sor_std", "dbscan_eps", "cluster_ratio"):
        result[key] = _positive(result[key], key)
    if result["cluster_ratio"] > 1:
        raise ValueError("cluster_ratio must satisfy 0 < cluster_ratio <= 1")
    return result


def add_filter_args(parser):
    """Add reusable filtering arguments to an argparse parser; return the parser."""
    parser.add_argument("--point-filter", choices=("none", "sor", "dbscan", "sor_dbscan"),
                        default=_DEFAULTS["algorithm"], help="per-instance 3D point filter")
    parser.add_argument("--sor-k", type=int, default=_DEFAULTS["sor_k"])
    parser.add_argument("--sor-std", type=float, default=_DEFAULTS["sor_std"])
    parser.add_argument("--dbscan-eps", type=float, default=_DEFAULTS["dbscan_eps"])
    parser.add_argument("--dbscan-min-samples", type=int, default=_DEFAULTS["dbscan_min_samples"])
    parser.add_argument("--cluster-policy", choices=("largest", "significant"),
                        default=_DEFAULTS["cluster_policy"])
    parser.add_argument("--min-cluster-size", type=int, default=_DEFAULTS["min_cluster_size"])
    parser.add_argument("--cluster-ratio", type=float, default=_DEFAULTS["cluster_ratio"])
    return parser


def config_from_args(args):
    """Normalize filtering arguments, with defaults for absent Namespace fields."""
    return validate_filter({key: getattr(args, "point_filter" if key == "algorithm" else key, value)
                            for key, value in _DEFAULTS.items()})


def _xyz_array(xyz):
    array = np.asarray(xyz)
    if array.ndim != 2 or array.shape[1] != 3 or array.dtype.kind not in "iuf":
        raise ValueError("xyz must be a real numeric Nx3 array")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("xyz must contain finite coordinates")
    return array


def _indices_array(point_indices, size):
    if isinstance(point_indices, (str, bytes, Mapping)):
        raise ValueError("point_indices must be a one-dimensional integer sequence")
    try:
        array = np.asarray(point_indices)
    except (TypeError, ValueError) as error:
        raise ValueError("point_indices must be a one-dimensional integer sequence") from error
    if array.ndim != 1 or (array.size and array.dtype.kind not in "iu"):
        raise ValueError("point_indices must be a one-dimensional integer sequence")
    if array.size and (np.any(array < 0) or np.any(array >= size)):
        raise ValueError("point_indices are outside xyz bounds")
    return np.unique(array.astype(np.int64))


def _sor(points, config):
    count = len(points)
    k = min(config["sor_k"], max(0, count - 1))
    stage = {"stage": "sor", "input": count, "k": k,
             "std_ratio": config["sor_std"], "std_ddof": 0}
    if count <= 2:
        stage.update(kept=count, removed=0, status="too_few")
        return np.ones(count, dtype=bool), stage
    means = np.empty(count, dtype=np.float64)
    # At most 256 x N distances; distance and scratch matrices share a 32 MiB
    # budget. Summing coordinate differences avoids cancellation for duplicates.
    chunk_size = min(256, max(1, (32 * 1024 * 1024) // (16 * count)))
    for start in range(0, count, chunk_size):
        stop = min(count, start + chunk_size)
        squared = np.zeros((stop - start, count), dtype=np.float64)
        for axis in range(3):
            delta = points[start:stop, axis, None] - points[None, :, axis]
            np.square(delta, out=delta)
            squared += delta
        squared[np.arange(stop - start), np.arange(start, stop)] = np.inf
        squared.partition(k - 1, axis=1)
        nearest = squared[:, :k]
        np.sqrt(nearest, out=nearest)
        means[start:stop] = nearest.mean(axis=1)
    mean, std = float(means.mean()), float(means.std(ddof=0))
    threshold = mean + config["sor_std"] * std
    keep = means <= threshold
    stage.update(mean_distance=mean, std_distance=std, threshold=threshold,
                 kept=int(keep.sum()), removed=int((~keep).sum()), status="filtered")
    return keep, stage


def _dbscan(points, config):
    count, eps = len(points), config["dbscan_eps"]
    cells = {}
    keys = []
    for index, point in enumerate(points):
        key = tuple(math.floor(float(coordinate) / eps) for coordinate in point)
        keys.append(key)
        cells.setdefault(key, []).append(index)
    cells = {key: np.asarray(indices, dtype=np.int64) for key, indices in cells.items()}
    eps_squared = eps * eps

    def neighbors(index):
        key = keys[index]
        nearby = [cells[other] for dx, dy, dz in _CELL_OFFSETS
                  if (other := (key[0] + dx, key[1] + dy, key[2] + dz)) in cells]
        candidates = np.concatenate(nearby)
        delta = points[candidates] - points[index]
        squared = np.einsum("ij,ij->i", delta, delta)
        return np.sort(candidates[squared <= eps_squared])

    # -2 means unvisited internally; public -2 is reserved for SOR removals.
    labels = np.full(count, -2, dtype=np.int32)
    queued = np.zeros(count, dtype=bool)
    cluster_count = 0
    for seed in range(count):
        if labels[seed] != -2:
            continue
        adjacent = neighbors(seed)
        if len(adjacent) < config["dbscan_min_samples"]:
            labels[seed] = -1
            continue
        labels[seed] = cluster_count
        queue = deque(int(index) for index in adjacent if index != seed)
        queued[adjacent] = True
        while queue:
            index = queue.popleft()
            if labels[index] == -1:
                labels[index] = cluster_count  # Previously visited noise is a border point.
            if labels[index] != -2:
                continue
            labels[index] = cluster_count
            adjacent = neighbors(index)
            if len(adjacent) >= config["dbscan_min_samples"]:
                pending = adjacent[~queued[adjacent]]
                queued[pending] = True
                queue.extend(int(candidate) for candidate in pending)
        cluster_count += 1
    sizes = np.bincount(labels[labels >= 0], minlength=cluster_count)
    largest_size = int(sizes.max()) if cluster_count else 0
    if config["cluster_policy"] == "largest":
        accepted = [int(np.argmax(sizes))] if cluster_count else []
        threshold = None
    else:
        threshold = max(config["min_cluster_size"], largest_size * config["cluster_ratio"])
        accepted = np.flatnonzero(sizes >= threshold).tolist()
    keep = np.isin(labels, accepted)
    discarded = [int(index) for index in range(cluster_count) if index not in accepted]
    stage = {"stage": "dbscan", "input": count, "kept": int(keep.sum()),
             "removed": int((~keep).sum()), "status": "filtered" if count else "too_few",
             "eps": eps, "min_samples": config["dbscan_min_samples"],
             "cluster_policy": config["cluster_policy"], "cluster_count": cluster_count,
             "cluster_sizes": sizes.tolist(), "largest_size": largest_size,
             "size_threshold": threshold, "kept_cluster_labels": accepted,
             "discarded_cluster_labels": discarded,
             "discarded_cluster_points": int(sum(sizes[index] for index in discarded)),
             "noise_points": int(np.count_nonzero(labels == -1)),
             "warning": ("cluster policy discarded spatial components; separated valid surfaces "
                         "may also be removed") if discarded else None}
    return keep, labels, stage


def _filter_points(xyz, indices, config):
    count = len(indices)
    keep = np.ones(count, dtype=bool)
    labels = np.zeros(count, dtype=np.int32)
    stages = []
    algorithm = config["algorithm"]
    if algorithm in ("sor", "sor_dbscan"):
        keep, stage = _sor(xyz[indices], config)
        stages.append(stage)
        labels[~keep] = -2
    if algorithm in ("dbscan", "sor_dbscan"):
        positions = np.flatnonzero(keep)
        accepted, cluster_labels, stage = _dbscan(xyz[indices[positions]], config)
        stages.append(stage)
        labels[positions] = cluster_labels
        keep[positions] = accepted
    status = ("disabled" if algorithm == "none" else
              "too_few" if count == 0 or (algorithm == "sor" and count <= 2) else "filtered")
    metadata = {"config": dict(config), "input": count, "kept": int(keep.sum()),
                "removed": int((~keep).sum()), "status": status, "stages": stages,
                "uses_gt": False, "index_order": "sorted unique original scan indices"}
    return {"kept_indices": indices[keep], "removed_indices": indices[~keep],
            "cluster_labels": labels, "metadata": metadata}


def filter_instance_points(xyz, point_indices, config=None):
    """Filter one instance; labels align with sorted unique input point indices.

    Cluster IDs describe DBSCAN before cluster-policy selection, so a discarded
    component retains its nonnegative ID. Noise is -1 and SOR-removed points -2.
    With only SOR or disabled filtering, retained points have label 0. N <= 2 is
    preserved by SOR; DBSCAN still applies min_samples and the cluster policy.
    """
    config = validate_filter(config)
    xyz = _xyz_array(xyz)
    return _filter_points(xyz, _indices_array(point_indices, len(xyz)), config)


def _json_copy(value):
    if isinstance(value, np.ndarray):
        return _json_copy(value.tolist())
    if isinstance(value, np.generic):
        return _json_copy(value.item())
    if isinstance(value, Mapping):
        return {key: _json_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_copy(item) for item in value]
    return deepcopy(value)


def _surface_ids(surface_class_ids):
    if isinstance(surface_class_ids, (str, bytes, Mapping)):
        raise ValueError("surface_class_ids must be an iterable of class IDs in 1..254")
    try:
        return tuple(sorted({_integer(value, "surface_class_ids", 1, 254)
                             for value in surface_class_ids}))
    except TypeError as error:
        raise ValueError("surface_class_ids must be an iterable of class IDs in 1..254") from error


def _clean_instances(xyz, instances, config, surface_class_ids=()):
    surface_class_ids = _surface_ids(surface_class_ids)
    if not isinstance(instances, list):
        raise ValueError("instances must be a list of dictionaries")
    cleaned, class_stats = [], {}
    for record in instances:
        if not isinstance(record, dict) or "point_indices" not in record or "class_id" not in record:
            raise ValueError("each instance must have point_indices and class_id")
        class_id = _integer(record["class_id"], "class_id", 1, 255)
        indices = _indices_array(record["point_indices"], len(xyz))
        if class_id in surface_class_ids:
            count = len(indices)
            metadata = {"config": dict(config), "effective_algorithm": "none",
                        "reason": "surface class exempt from outlier filtering",
                        "input": count, "kept": count, "removed": 0,
                        "status": "surface_exempt", "stages": [], "uses_gt": False,
                        "index_order": "sorted unique original scan indices"}
            result = {"kept_indices": indices, "removed_indices": indices[:0],
                      "metadata": metadata}
        else:
            result = _filter_points(xyz, indices, config)
        item = _json_copy(record)
        item.update(class_id=class_id, point_indices=result["kept_indices"].tolist(),
                    original_point_indices=indices.tolist(),
                    removed_point_indices=result["removed_indices"].tolist(),
                    filter=result["metadata"])
        cleaned.append(item)
        counts = class_stats.setdefault(class_id, {"class_id": class_id, "instances": 0,
                                                   "input": 0, "kept": 0, "removed": 0})
        counts["instances"] += 1
        for key in ("input", "kept", "removed"):
            counts[key] += result["metadata"][key]
    stats = {"config": dict(config), "input_instances": len(instances),
             "kept_instances": sum(bool(item["point_indices"]) for item in cleaned),
             "empty_instances": sum(not item["point_indices"] for item in cleaned),
             "input": sum(item["filter"]["input"] for item in cleaned),
             "kept": sum(item["filter"]["kept"] for item in cleaned),
             "removed": sum(item["filter"]["removed"] for item in cleaned),
             "per_class": [class_stats[key] for key in sorted(class_stats)],
             "surface_class_ids": list(surface_class_ids),
             "exempt_surface_points": sum(item["filter"]["input"] for item in cleaned
                                          if item["class_id"] in surface_class_ids),
             "exempt_surface_instances": sum(item["class_id"] in surface_class_ids
                                             for item in cleaned),
             "count_rule": "per-instance observations; shared scan points may count repeatedly",
             "uses_gt": False}
    return {"instances": cleaned, "stats": stats}


def clean_instances(xyz, instances, config=None, surface_class_ids=()):
    """Copy records, retaining empty instances and exempting requested surface classes."""
    return _clean_instances(_xyz_array(xyz), instances, validate_filter(config), surface_class_ids)


def clean_point_support(xyz, prediction, instances, config=None, surface_class_ids=()):
    """Clear unsupported predictions using the union of kept same-class instances.

    Enabled filtering requires original same-class instance coverage of every
    nonzero, non-surface prediction. Surface predictions and background stay
    unchanged. Overlapping cameras may save a point rejected by another instance.
    Disabled mode is an identity operation and accepts empty instance records.
    Raw XYZ, predictions and nested input records are never mutated.
    """
    config = validate_filter(config)
    xyz = _xyz_array(xyz)
    prediction = np.asarray(prediction)
    if prediction.ndim != 1 or len(prediction) != len(xyz) or prediction.dtype != np.uint8:
        raise ValueError("prediction must be a uint8 vector aligned with xyz")
    result = _clean_instances(xyz, instances, config, surface_class_ids)
    surface_class_ids = result["stats"]["surface_class_ids"]
    surface_mask = np.isin(prediction, surface_class_ids)
    keep_mask = np.ones(len(xyz), dtype=bool)
    enabled = config["algorithm"] != "none"
    if enabled:
        covered = np.zeros(len(xyz), dtype=bool)
        supported = np.zeros(len(xyz), dtype=bool)
        for record in result["instances"]:
            original = np.asarray(record["original_point_indices"], dtype=np.int64)
            kept = np.asarray(record["point_indices"], dtype=np.int64)
            covered[original] |= prediction[original] == record["class_id"]
            supported[kept] |= prediction[kept] == record["class_id"]
        missing = (prediction != 0) & ~surface_mask & ~covered
        if missing.any():
            raise ValueError(f"{int(missing.sum())} nonzero prediction points lack same-class "
                             "instance membership; supply complete original instance records")
        keep_mask = (prediction == 0) | surface_mask | supported
    output = prediction.copy()
    output[~keep_mask] = 0
    removed = np.flatnonzero(~keep_mask).astype(np.int64)
    per_class = []
    for class_id in np.unique(prediction[prediction != 0]):
        selected = prediction == class_id
        per_class.append({"class_id": int(class_id), "input": int(selected.sum()),
                          "kept": int(np.count_nonzero(selected & keep_mask)),
                          "removed": int(np.count_nonzero(selected & ~keep_mask))})
    metadata = {"config": dict(config), "status": "filtered" if enabled else "disabled",
                "input": int(np.count_nonzero(prediction)),
                "kept": int(np.count_nonzero(output)), "removed": int(len(removed)),
                "per_class": per_class, "instance_stats": result["stats"],
                "per_instance": [dict(index=index, class_id=record["class_id"],
                                      **deepcopy(record["filter"]))
                                 for index, record in enumerate(result["instances"])],
                "surface_class_ids": list(surface_class_ids),
                "exempt_surface_points": int(np.count_nonzero(surface_mask)),
                "exempt_surface_instances": result["stats"]["exempt_surface_instances"],
                "ownership": ("all predictions retained; filtering disabled" if not enabled else
                              "surface predictions retained without instance coverage; other "
                              "predictions use union of kept instances matching original predicted class"
                              if surface_class_ids else
                              "union of kept instances matching original predicted class"),
                "uses_gt": False}
    return {"prediction": output, "keep_mask": keep_mask, "removed_indices": removed,
            "instances": result["instances"], "metadata": metadata}

"""Conservative, frame-local instance merging using shared original LiDAR indices.

This module uses only the Python standard library. Geometry, ROI indices and GT
fields are deliberately irrelevant: callers provide records for one scan.
"""

from copy import deepcopy
from collections import Counter
from collections.abc import Iterable, Mapping
import math
from numbers import Integral, Real


_REQUIRED = ("global_instance_id", "camera", "class_id", "class_name", "score", "point_indices")


def _integer(value, name, minimum, maximum=None):
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside the allowed range")
    return value


def _finite_real(value, name):
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be finite and numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _class_ids(values, name):
    if (not isinstance(values, Iterable)
            or isinstance(values, (str, bytes, Mapping))):
        raise TypeError(f"{name} must be an iterable of class IDs")
    return frozenset(_integer(value, f"{name} entry", 1, 254) for value in values)


def _normalize(instances, maximum_class_id=16):
    if not isinstance(instances, list):
        raise TypeError("instances must be a list of dictionaries")
    records = {}
    for item in instances:
        if not isinstance(item, dict):
            raise TypeError("each instance must be a dictionary")
        missing = [key for key in _REQUIRED if key not in item]
        if missing:
            raise ValueError(f"instance is missing required fields: {', '.join(missing)}")
        identifier = _integer(item["global_instance_id"], "global_instance_id", 1)
        if identifier in records:
            raise ValueError("global_instance_id must be unique")
        camera = item["camera"]
        if (not isinstance(camera, str) or not camera or camera in (".", "..")
                or any(character.isspace() or ord(character) < 32
                       or character in '/\\<>:"|?*' for character in camera)):
            raise ValueError("camera must be a nonempty safe string token")
        class_id = _integer(item["class_id"], "class_id", 1, maximum_class_id)
        if not isinstance(item["class_name"], str):
            raise TypeError("class_name must be a string")
        score = _finite_real(item["score"], "score")
        indices = item["point_indices"]
        if (not isinstance(indices, Iterable)
                or isinstance(indices, (str, bytes, Mapping))):
            raise TypeError("point_indices must be a one-dimensional integer iterable")
        if getattr(indices, "ndim", 1) != 1:
            raise ValueError("point_indices must be one-dimensional")
        points = frozenset(_integer(index, "point_indices entry", 0) for index in indices)
        records[identifier] = {"camera": camera, "class_id": class_id,
                               "class_name": item["class_name"], "score": score,
                               "points": points}
    return records


def merge_instances_by_points(
    instances: list[dict], min_shared_points: int = 3, min_overlap: float = .3,
    eligible_class_ids=None,
) -> dict:
    """Merge same-class foreground records from different cameras in one scan.

    A pair qualifies when shared >= min_shared_points AND
    shared / min(unique_size_a, unique_size_b) >= min_overlap. Candidate order is
    descending overlap, IoU, shared count, minimum member score, then ascending
    original IDs. Union-find additionally enforces one member per camera and
    complete-link eligibility between every member of the two prospective groups.
    Thus a chain A-B-C cannot merge without qualifying evidence for A-C.

    All inputs survive, including empty records and stuff labels 11..16. Returned
    group IDs are minimum original IDs; mapping uses integer keys (JSON writers
    may stringify these). Duplicate indices within one record are normalized
    before all counts. No nested input fields are mutated or returned by alias.

    By default, input class IDs must be 1..16 and only classes 1..10 merge.
    Supplying eligible_class_ids accepts input IDs 1..254 and compares only
    classes explicitly selected by that iterable (an empty iterable merges none).
    """
    min_shared_points = _integer(min_shared_points, "min_shared_points", 1)
    min_overlap = _finite_real(min_overlap, "min_overlap")
    if not 0 < min_overlap <= 1:
        raise ValueError("min_overlap must satisfy 0 < min_overlap <= 1")
    legacy = eligible_class_ids is None
    eligible_ids = (frozenset(range(1, 11)) if legacy else
                    _class_ids(eligible_class_ids, "eligible_class_ids"))
    records = _normalize(instances, maximum_class_id=16 if legacy else 254)
    identifiers = sorted(records)
    foreground = [identifier for identifier in identifiers
                  if records[identifier]["class_id"] in eligible_ids]
    nonempty = [identifier for identifier in foreground if records[identifier]["points"]]
    pairs, eligible = [], {}
    # Generate all comparable pairs, including disjoint pairs, in stable ID order.
    for offset, first in enumerate(nonempty):
        a = records[first]
        for second in nonempty[offset + 1:]:
            b = records[second]
            if a["camera"] == b["camera"] or a["class_id"] != b["class_id"]:
                continue
            size_a, size_b = len(a["points"]), len(b["points"])
            shared = len(a["points"] & b["points"])
            union = size_a + size_b - shared
            overlap = shared / min(size_a, size_b)
            qualifies = shared >= min_shared_points and overlap >= min_overlap
            pair = {"instance_a": first, "instance_b": second,
                    "camera_a": a["camera"], "camera_b": b["camera"],
                    "class_id": a["class_id"], "size_a": size_a, "size_b": size_b,
                    "shared_points": shared, "union_points": union,
                    "point_iou": shared / union, "overlap_smaller": overlap,
                    "eligible": qualifies, "decision": "rejected_threshold"}
            pairs.append(pair)
            if qualifies:
                eligible[(first, second)] = pair

    parent = {identifier: identifier for identifier in identifiers}
    members = {identifier: {identifier} for identifier in identifiers}
    cameras = {identifier: {records[identifier]["camera"]} for identifier in identifiers}

    def find(identifier):
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier

    candidates = sorted(eligible.values(), key=lambda pair: (
        -pair["overlap_smaller"], -pair["point_iou"], -pair["shared_points"],
        -min(records[pair["instance_a"]]["score"], records[pair["instance_b"]]["score"]),
        pair["instance_a"], pair["instance_b"],
    ))
    for pair in candidates:
        root_a, root_b = find(pair["instance_a"]), find(pair["instance_b"])
        if root_a == root_b:
            pair["decision"] = "cycle_redundant"
        elif cameras[root_a] & cameras[root_b]:
            pair["decision"] = "rejected_camera_conflict"
        elif any((min(first, second), max(first, second)) not in eligible
                 for first in members[root_a] for second in members[root_b]):
            pair["decision"] = "rejected_group_consistency"
        else:
            keep, remove = sorted((root_a, root_b))
            parent[remove] = keep
            members[keep].update(members.pop(remove))
            cameras[keep].update(cameras.pop(remove))
            pair["decision"] = "accepted"

    groups, mapping = [], {}
    for root in sorted(members):
        member_ids = sorted(members[root])
        points = set().union(*(records[identifier]["points"] for identifier in member_ids))
        observations = sum(len(records[identifier]["points"]) for identifier in member_ids)
        representative = records[root]
        groups.append({
            "merged_instance_id": root, "member_instance_ids": member_ids,
            "source_cameras": sorted(cameras[root]),
            "class_id": representative["class_id"], "class_name": representative["class_name"],
            "score": max(records[identifier]["score"] for identifier in member_ids),
            "point_indices": sorted(points), "source_point_observations": observations,
            "deduplicated_shared_observations": observations - len(points),
            "member_count": len(member_ids), "merged": len(member_ids) > 1,
        })
        mapping.update((identifier, root) for identifier in member_ids)
    decisions = Counter(pair["decision"] for pair in pairs)
    metadata = {
        "input_instances": len(identifiers), "input_foreground_instances": len(foreground),
        "nonempty_foreground_instances": len(nonempty), "output_groups": len(groups),
        "merged_groups": sum(group["merged"] for group in groups),
        "reduction": len(identifiers) - len(groups), "pair_count": len(pairs),
        "eligible_pairs": len(eligible), "accepted_edges": decisions["accepted"],
        "rejected_threshold": decisions["rejected_threshold"],
        "cycle_redundant": decisions["cycle_redundant"],
        "rejected_camera_conflict": decisions["rejected_camera_conflict"],
        "rejected_group_consistency": decisions["rejected_group_consistency"],
        "min_shared_points": min_shared_points, "min_overlap": min_overlap,
        "foreground_class_ids": sorted(eligible_ids),
        "stuff_class_ids": (list(range(11, 17)) if legacy else
                            sorted({record["class_id"] for record in records.values()} - eligible_ids)),
        "rule": (("same-class foreground (1..10), " if legacy else
                  "same-class explicitly eligible class, ") +
                 "different camera, nonempty full-scan original "
                 "point_indices; shared_points >= min_shared_points AND overlap_smaller >= min_overlap; "
                 "one member per camera; complete-link: every cross-group member pair must be eligible"),
        "candidate_order": "descending overlap_smaller, point_iou, shared_points, minscore; ascending instance_a, instance_b",
        "group_consistency": "complete-link; missing or below-threshold pair evidence rejects transitive chain merges",
        "group_id_rule": "minimum member global_instance_id; no renumbering",
        "score_rule": "maximum member score; no probability combination",
        "point_count_rule": "source observations sum per-member unique points; deduplicated shared observations = source observations minus union size",
        "point_ownership": "union within each group; overlap between separate groups and different classes is allowed",
        "evidence": "original LiDAR point-index overlap only; no GT, geometry, hull, bbox or ROI filtering",
        "identity_scope": "one scan/frame only; no cross-frame identity or tracking",
        "uses_gt": False, "uses_geometry": False, "cross_frame_identity": False,
    }
    return {"groups": groups, "pairs": pairs,
            "mapping": dict(sorted(mapping.items())), "metadata": metadata}


def merge_instance_support(instances, surface_class_ids=(), min_shared_points=3,
                           min_overlap=.3):
    """Merge cleaned, per-camera support for one frame and retain provenance.

    Call AFTER clean_point_support: only point_indices are evidence and enter
    the sorted, deduplicated union; original_point_indices never add support.
    Every observed class in 1..254 is eligible except explicit surface classes,
    which survive as singleton records. Pair thresholds and conservative camera
    and complete-link rules are identical to merge_instances_by_points.

    Returns that function's groups/pairs/mapping/metadata plus instances suitable
    for build_instance_boxes. Each output uses the minimum member global ID,
    its camera as representative, maximum score, source_cameras and sorted
    member_instance_ids. source_instances contains independent deep copies of
    the normalized per-camera records, including all filter/original support
    details. Aggregate records do not inherit a single member's filter details.

    Legacy records may omit class_name (defaults to str(class_id)), score (0.)
    or global_instance_id (smallest unused positive ID, in input order). Present
    IDs and all support indices remain strictly validated; duplicate explicit
    IDs fail. Inputs and nested provenance are never mutated or aliased.
    """
    surface_ids = _class_ids(surface_class_ids, "surface_class_ids")
    if not isinstance(instances, list):
        raise TypeError("instances must be a list of dictionaries")
    used_ids = set()
    for item in instances:
        if not isinstance(item, dict):
            raise TypeError("each instance must be a dictionary")
        if "global_instance_id" in item:
            identifier = _integer(item["global_instance_id"], "global_instance_id", 1)
            if identifier in used_ids:
                raise ValueError("global_instance_id must be unique")
            used_ids.add(identifier)
    normalized = []
    next_id = 1
    for item in instances:
        record = deepcopy(item)
        if "global_instance_id" not in record:
            while next_id in used_ids:
                next_id += 1
            record["global_instance_id"] = next_id
            used_ids.add(next_id)
        if "class_id" in record:
            record.setdefault("class_name", str(record["class_id"]))
        record.setdefault("score", 0.)
        normalized.append(record)
    # Validate before deriving eligible classes, including excluded surfaces.
    records = _normalize(normalized, maximum_class_id=254)
    eligible_ids = sorted({record["class_id"] for record in records.values()} - surface_ids)
    report = merge_instances_by_points(normalized, min_shared_points, min_overlap,
                                       eligible_class_ids=eligible_ids)
    sources = {record["global_instance_id"]: record for record in normalized}
    output = []
    for group in report["groups"]:
        identifier = group["merged_instance_id"]
        representative = sources[identifier]
        record = deepcopy(group)
        record.update(global_instance_id=identifier, camera=representative["camera"],
                      source_instances=[deepcopy(sources[member])
                                        for member in group["member_instance_ids"]])
        for key in ("camera_instance_id", "instance_id"):
            if key in representative:
                record[key] = deepcopy(representative[key])
        output.append(record)
    report["instances"] = output
    report["metadata"]["surface_class_ids"] = sorted(surface_ids)
    report["metadata"]["support_rule"] = "union of cleaned point_indices only; original support retained in source_instances"
    return report

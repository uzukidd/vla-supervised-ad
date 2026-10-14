"""Backproject camera labels to unique LiDAR points and export full point clouds."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from segmentation_metrics import ConfusionMatrix

UNCOVERED_COLOR = (95, 105, 120)
UNASSIGNED_COLOR = (190, 190, 190)


def fuse_point_labels(point_count: int, observations: list[dict], num_classes: int) -> dict:
    """Nonzero majority vote; ties prefer the earliest camera observation.

    Zero labels abstain, but their points still count as geometrically covered.
    Camera observations must contain unique indices within each camera.
    """
    if point_count < 0 or not 1 <= num_classes <= 255 or len(observations) > 65535:
        raise ValueError("Invalid point count, class count or observation count")
    votes = np.zeros((point_count, num_classes), dtype=np.uint16)
    observation_count = np.zeros(point_count, dtype=np.uint16)
    validated = []
    for observation in observations:
        indices = np.asarray(observation["point_indices"])
        labels = np.asarray(observation["prediction"])
        if indices.ndim != 1 or labels.shape != indices.shape or indices.dtype.kind not in "ui" or labels.dtype.kind not in "ui":
            raise ValueError("Observation indices and predictions must be matching 1D integer arrays")
        if np.any(indices < 0) or np.any(indices >= point_count) or np.any(labels < 0) or np.any(labels >= num_classes):
            raise ValueError("Observation point index or prediction class is out of range")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("A camera observation must not repeat point indices")
        indices = indices.astype(np.int64)
        labels = labels.astype(np.uint8)
        observation_count[indices] += 1
        active = labels != 0
        votes[indices[active], labels[active]] += 1
        validated.append((indices, labels))
    max_votes = votes.max(axis=1)
    prediction = np.zeros(point_count, dtype=np.uint8)
    source_camera = np.full(point_count, -1, dtype=np.int32)
    for camera_index, (indices, labels) in enumerate(validated):
        chosen = (labels != 0) & (source_camera[indices] == -1) & (votes[indices, labels] == max_votes[indices])
        prediction[indices[chosen]] = labels[chosen]
        source_camera[indices[chosen]] = camera_index
    return {"prediction": prediction, "covered": observation_count > 0,
            "observation_count": observation_count, "winning_votes": max_votes,
            "source_camera": source_camera}


def prediction_colors(prediction: np.ndarray, covered: np.ndarray, palette: np.ndarray) -> np.ndarray:
    colors = palette[prediction].copy()
    colors[(prediction == 0) & covered] = UNASSIGNED_COLOR
    colors[~covered] = UNCOVERED_COLOR
    return colors


def write_ply(path: Path, xyz: np.ndarray, colors: np.ndarray, labels: np.ndarray,
              covered: np.ndarray, source_camera: np.ndarray) -> None:
    """Binary little-endian PLY, preserving point order and useful scalar fields."""
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.all(np.isfinite(xyz)):
        raise ValueError("PLY coordinates must be a finite Nx3 array")
    n = len(xyz)
    if colors.shape != (n, 3) or any(array.shape != (n,) for array in (labels, covered, source_camera)):
        raise ValueError("PLY attributes must align with every original point")
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                      ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                      ("class_id", "u1"), ("covered", "u1"), ("source_camera", "<i4")])
    vertices = np.empty(n, dtype=dtype)
    for axis, name in enumerate(("x", "y", "z")):
        vertices[name] = xyz[:, axis]
    for axis, name in enumerate(("red", "green", "blue")):
        vertices[name] = colors[:, axis]
    vertices["class_id"] = labels
    vertices["covered"] = covered
    vertices["source_camera"] = source_camera
    header = ("ply\nformat binary_little_endian 1.0\ncomment Coordinates: original LiDAR sensor frame, meters\n"
              f"element vertex {n}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar class_id\n"
              "property uchar covered\nproperty int source_camera\nend_header\n")
    with Path(path).open("wb") as stream:
        stream.write(header.encode("ascii"))
        vertices.tofile(stream)


def _point_report(matrix: ConfusionMatrix) -> dict:
    report = matrix.report()
    report["point_accuracy"] = report.pop("pixel_accuracy")
    report["valid_points"] = report.pop("valid_pixels")
    for row in report["per_class"]:
        row["gt_points"] = row.pop("gt_pixels")
        row["predicted_points"] = row.pop("predicted_pixels")
    report["metric_definitions"].pop("pixel_accuracy")
    report["metric_definitions"].update({"point_accuracy": "sum(diagonal(matrix)) / valid_points",
                                          "count_unit": "Unique LiDAR points per keyframe; each point counts once across cameras",
                                          "matrix": "GT rows, prediction columns; ignored GT class 0 excluded"})
    return report


class UniquePointMetrics:
    def __init__(self, class_names: dict[int, str]):
        self.class_names = dict(class_names)
        self.covered = ConfusionMatrix(class_names, ignore_ids={0})
        self.full_scan = ConfusionMatrix(class_names, ignore_ids={0})
        self.counts = {name: 0 for name in ("frames", "total_points", "covered_points", "assigned_points",
                                           "valid_gt_points", "covered_valid_gt_points", "assigned_valid_gt_points")}
        self.class_total = np.zeros(len(class_names), dtype=np.int64)
        self.class_covered = np.zeros(len(class_names), dtype=np.int64)

    def update(self, target: np.ndarray, prediction: np.ndarray, covered: np.ndarray) -> None:
        if target.ndim != 1 or prediction.shape != target.shape or covered.shape != target.shape or covered.dtype != bool:
            raise ValueError("Unique point metrics require aligned 1D labels and a boolean coverage mask")
        if np.any((prediction != 0) & ~covered):
            raise ValueError("Uncovered points must have prediction ID 0")
        self.full_scan.update(target, prediction)
        self.covered.update(target[covered], prediction[covered])
        valid = target != 0
        assigned = prediction != 0
        values = (1, len(target), int(covered.sum()), int(assigned.sum()), int(valid.sum()),
                  int((valid & covered).sum()), int((valid & assigned).sum()))
        for name, value in zip(self.counts, values):
            self.counts[name] += value
        self.class_total += np.bincount(target, minlength=len(self.class_names))
        self.class_covered += np.bincount(target[covered], minlength=len(self.class_names))

    def report(self) -> dict:
        counts = dict(self.counts)
        ratio = lambda a, b: float(a / b) if b else None
        coverage = {**counts, "geometric_coverage": ratio(counts["covered_points"], counts["total_points"]),
                    "valid_gt_coverage": ratio(counts["covered_valid_gt_points"], counts["valid_gt_points"]),
                    "assigned_fraction_of_covered": ratio(counts["assigned_points"], counts["covered_points"]),
                    "assigned_fraction_of_valid_gt": ratio(counts["assigned_valid_gt_points"], counts["valid_gt_points"]),
                    "per_class": [{"id": i, "name": name, "ignored": i == 0,
                                   "total_gt_points": int(self.class_total[i]), "covered_gt_points": int(self.class_covered[i]),
                                   "coverage": ratio(int(self.class_covered[i]), int(self.class_total[i]))}
                                  for i, name in self.class_names.items()]}
        return {"covered_points": _point_report(self.covered), "full_scan": _point_report(self.full_scan),
                "coverage": coverage,
                "evaluation_scope": {"unit": "unique LiDAR points per keyframe",
                                     "covered_points": "Only geometry-covered points; GT0 ignored; covered prediction0 remains an error",
                                     "full_scan": "Every valid GT point in selected scans; uncovered prediction0 counts as an error",
                                     "fusion": "Nonzero majority vote across cameras, ties prefer earlier camera observation; zero abstains",
                                     "limitations": "Camera-derived labels with sparse projection; occlusion beyond same-pixel depth filtering "
                                                    "and dynamic-object time offsets remain. Not an official nuScenes submission score."}}

    def save(self, output: Path) -> dict:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        report = self.report()
        (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        fields = ("scope", "metric", "value", "id", "name", "ignored", "iou", "dice", "precision", "recall",
                  "gt_points", "predicted_points", "tp", "fp", "fn")
        with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for scope in ("covered_points", "full_scan"):
                for name in ("point_accuracy", "mean_iou", "mean_accuracy", "frequency_weighted_iou", "macro_dice"):
                    writer.writerow({"scope": scope, "metric": name, "value": report[scope][name]})
                for row in report[scope]["per_class"]:
                    writer.writerow({"scope": scope, **row})
            for name, value in report["coverage"].items():
                if name != "per_class":
                    writer.writerow({"scope": "coverage", "metric": name, "value": value})
        with (output / "coverage.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("id", "name", "ignored", "total_gt_points", "covered_gt_points", "coverage"))
            writer.writeheader()
            writer.writerows(report["coverage"]["per_class"])
        for name, matrix in (("covered", self.covered), ("full_scan", self.full_scan)):
            with (output / f"confusion_matrix_{name}.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["gt_id", "gt_name", *(f"prediction_{i}:{label}" for i, label in self.class_names.items())])
                for i, label in self.class_names.items():
                    writer.writerow([i, label, *matrix.matrix[i].tolist()])
        return report

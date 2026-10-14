"""NumPy semantic-segmentation metrics with explicit uncovered-pixel handling.

Rows of ``ConfusionMatrix.matrix`` are ground-truth IDs and columns are
prediction IDs. By default GT 0 (unlabeled) and GT 255 are completely excluded.
Prediction 0 is still a valid prediction: on a retained GT pixel it contributes
an FN to that GT class. An explicit ``ignore_ids`` replaces the default set;
class IDs in that set are excluded from every aggregate class metric.

All scores are fractions in [0, 1]. A zero denominator gives None, serialized
as JSON null and as an empty CSV cell. mIoU and macro Dice include classes with
FP but no GT; mean accuracy averages recall only over classes with GT.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


_AGGREGATE_METRICS = (
    "pixel_accuracy",
    "mean_iou",
    "mean_accuracy",
    "frequency_weighted_iou",
    "macro_dice",
)
_PER_CLASS_FIELDS = (
    "id", "name", "ignored", "iou", "dice", "precision", "recall",
    "gt_pixels", "predicted_pixels", "tp", "fp", "fn",
)


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _is_integer(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


class ConfusionMatrix:
    """Accumulate validated label arrays and export strict JSON/CSV metrics.

    ``class_names`` must contain exactly the consecutive IDs 0 .. N-1.
    Inputs may have any matching non-scalar shape and must have integer dtypes.
    Prediction IDs are validated on the entire array, even at ignored GT
    pixels. GT IDs outside the class map are allowed only if explicitly ignored.
    Validation happens before accumulation, so failed updates change no counts.
    """

    def __init__(
        self,
        class_names: dict[int, str],
        ignore_ids: set[int] | None = None,
    ) -> None:
        if not isinstance(class_names, dict):
            raise TypeError("class_names must be a dict mapping integer IDs to names")
        if not class_names:
            raise ValueError("class_names must contain at least one class")
        if any(not _is_integer(class_id) for class_id in class_names):
            raise TypeError("class IDs must be integers, excluding booleans")
        if set(class_names) != set(range(len(class_names))):
            raise ValueError("class IDs must be consecutive from 0 to N-1")
        if any(not isinstance(name, str) for name in class_names.values()):
            raise TypeError("class names must be strings")
        if ignore_ids is not None:
            if not isinstance(ignore_ids, (set, frozenset)):
                raise TypeError("ignore_ids must be a set of integer IDs or None")
            if any(not _is_integer(class_id) for class_id in ignore_ids):
                raise TypeError("ignored IDs must be integers, excluding booleans")

        self.class_names = {int(i): class_names[i] for i in range(len(class_names))}
        self.ignore_ids = frozenset({0, 255} if ignore_ids is None else ignore_ids)
        self.num_classes = len(self.class_names)
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, target: np.ndarray, prediction: np.ndarray) -> None:
        """Add one batch, excluding ignored GT pixels from all matrix counts."""
        for name, labels in (("target", target), ("prediction", prediction)):
            if not isinstance(labels, np.ndarray):
                raise TypeError(f"{name} must be a numpy.ndarray")
            if not np.issubdtype(labels.dtype, np.integer):
                raise TypeError(f"{name} labels must have an integer dtype")
            if labels.ndim == 0:
                raise ValueError(f"{name} must have at least one dimension")
        if target.shape != prediction.shape:
            raise ValueError(
                f"target and prediction shapes must match: {target.shape} != {prediction.shape}"
            )
        invalid_prediction = (prediction < 0) | (prediction >= self.num_classes)
        if np.any(invalid_prediction):
            bad_ids = np.unique(prediction[invalid_prediction]).tolist()
            raise ValueError(
                f"prediction IDs must be in [0, {self.num_classes - 1}]; got {bad_ids}"
            )

        retained = ~np.isin(target, list(self.ignore_ids))
        retained_target = target[retained]
        invalid_target = (retained_target < 0) | (retained_target >= self.num_classes)
        if np.any(invalid_target):
            bad_ids = np.unique(retained_target[invalid_target]).tolist()
            raise ValueError(f"unknown non-ignored target IDs: {bad_ids}")
        if retained_target.size == 0:
            return
        encoded = (
            retained_target.astype(np.int64) * self.num_classes
            + prediction[retained].astype(np.int64)
        )
        counts = np.bincount(encoded, minlength=self.num_classes ** 2)
        self.matrix += counts.reshape(self.matrix.shape)

    def report(self) -> dict:
        """Return JSON-compatible counts, fractional scores, and definitions."""
        gt_counts = self.matrix.sum(axis=1)
        prediction_counts = self.matrix.sum(axis=0)
        total = int(gt_counts.sum())
        per_class = []
        ious = []
        recalls = []
        dices = []
        weighted_iou = 0.0
        for class_id, name in self.class_names.items():
            tp = int(self.matrix[class_id, class_id])
            gt_pixels = int(gt_counts[class_id])
            predicted_pixels = int(prediction_counts[class_id])
            fp = predicted_pixels - tp
            fn = gt_pixels - tp
            iou = _ratio(tp, tp + fp + fn)
            dice = _ratio(2 * tp, 2 * tp + fp + fn)
            precision = _ratio(tp, tp + fp)
            recall = _ratio(tp, tp + fn)
            ignored = class_id in self.ignore_ids
            per_class.append({
                "id": class_id,
                "name": name,
                "ignored": ignored,
                "iou": iou,
                "dice": dice,
                "precision": precision,
                "recall": recall,
                "gt_pixels": gt_pixels,
                "predicted_pixels": predicted_pixels,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            })
            if not ignored:
                if iou is not None:
                    ious.append(iou)
                if dice is not None:
                    dices.append(dice)
                if recall is not None:
                    recalls.append(recall)
                if gt_pixels:
                    weighted_iou += gt_pixels / total * iou

        return {
            "pixel_accuracy": _ratio(int(np.trace(self.matrix)), total),
            "mean_iou": _mean(ious),
            "mean_accuracy": _mean(recalls),
            "frequency_weighted_iou": float(weighted_iou) if total else None,
            "macro_dice": _mean(dices),
            "valid_pixels": total,
            "per_class": per_class,
            "confusion_matrix": self.matrix.tolist(),
            "metric_definitions": {
                "matrix": "Rows are GT IDs; columns are prediction IDs; counts use retained GT pixels only.",
                "tp": "matrix[c,c]",
                "fp": "sum(matrix[:,c]) - tp",
                "fn": "sum(matrix[c,:]) - tp",
                "iou": "tp / (tp + fp + fn)",
                "dice": "2*tp / (2*tp + fp + fn)",
                "precision": "tp / (tp + fp)",
                "recall": "tp / (tp + fn)",
                "pixel_accuracy": "sum(diagonal(matrix)) / valid_pixels",
                "mean_iou": "Mean IoU over non-ignored classes with nonzero union, including FP-only classes.",
                "mean_accuracy": "Mean recall over non-ignored classes with GT pixels.",
                "frequency_weighted_iou": "Sum over non-ignored classes with GT: (gt_pixels / valid_pixels) * IoU.",
                "macro_dice": "Mean Dice over non-ignored classes with nonzero (2*tp + fp + fn), including FP-only classes.",
                "undefined": "Zero denominators and empty aggregate populations yield null (empty cells in CSV).",
                "scale": "All scores are fractions from 0 to 1, not percentages.",
            },
            "ignore_policy": {
                "ignore_ids": sorted(int(i) for i in self.ignore_ids),
                "ignored_class_ids": [i for i in self.class_names if i in self.ignore_ids],
                "ground_truth": "Ignored GT pixels are completely excluded, regardless of their predictions.",
                "prediction": "Prediction 0 remains counted on retained GT and contributes FN to the true class when different; all prediction IDs must be in range even on ignored GT.",
                "aggregates": "Class IDs in ignore_ids are excluded from all macro metrics and frequency-weighted IoU; per_class still lists their counts.",
                "configuration": "None defaults to {0,255}; an explicit ignore_ids set replaces those defaults.",
            },
        }

    def save(self, output_dir: Path) -> None:
        """Write metrics.json, metrics.csv, and confusion_matrix.csv.

        metrics.csv contains aggregate rows (metric/value) followed by per-class
        rows. confusion_matrix.csv labels both GT rows and prediction columns;
        its integer body equals the matrix included in metrics.json.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report = self.report()
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("scope", "metric", "value", *_PER_CLASS_FIELDS))
            writer.writeheader()
            for metric in _AGGREGATE_METRICS:
                writer.writerow({"scope": "aggregate", "metric": metric, "value": report[metric]})
            for row in report["per_class"]:
                writer.writerow({"scope": "per_class", **row})
        with (output_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["gt_id", "gt_name", *(f"prediction_{i}:{name}" for i, name in self.class_names.items())])
            for class_id, name in self.class_names.items():
                writer.writerow([class_id, name, *self.matrix[class_id].tolist()])

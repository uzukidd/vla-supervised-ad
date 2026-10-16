"""Run with: python -m unittest discover -s tests -p test_segmentation_metrics.py."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from shared.python.segmentation_metrics import ConfusionMatrix


class ConfusionMatrixTests(unittest.TestCase):
    def setUp(self):
        self.names = {0: "unlabeled", 1: "sidewalk", 2: "road"}
        self.metrics = ConfusionMatrix(self.names)

    def test_prediction_zero_is_false_negative(self):
        self.metrics.update(np.array([1, 1, 2, 2]), np.array([0, 1, 0, 2]))
        result = self.metrics.report()
        np.testing.assert_array_equal(self.metrics.matrix, [[0, 0, 0], [1, 1, 0], [1, 0, 1]])
        for metric in ("pixel_accuracy", "mean_iou", "mean_accuracy", "frequency_weighted_iou"):
            self.assertEqual(result[metric], 0.5)
        self.assertAlmostEqual(result["macro_dice"], 2 / 3)
        for row in result["per_class"][1:]:
            self.assertEqual(row["fn"], 1)
            self.assertEqual(row["precision"], 1.0)
            self.assertEqual(row["recall"], 0.5)
        self.assertTrue(result["per_class"][0]["ignored"])
        self.assertEqual(result["per_class"][0]["predicted_pixels"], 2)

    def test_all_uncovered_pixels_have_zero_scores(self):
        self.metrics.update(np.array([1, 2]), np.array([0, 0]))
        result = self.metrics.report()
        for metric in ("pixel_accuracy", "mean_iou", "mean_accuracy", "frequency_weighted_iou", "macro_dice"):
            self.assertEqual(result[metric], 0.0)
        self.assertIsNone(result["per_class"][1]["precision"])
        self.assertEqual(result["per_class"][1]["fn"], 1)

    def test_ignored_gt_pixels_excluded_from_every_count(self):
        self.metrics.update(np.array([[0, 255, 1], [255, 0, 2]]), np.array([[1, 2, 1], [0, 2, 2]]))
        result = self.metrics.report()
        np.testing.assert_array_equal(self.metrics.matrix, [[0, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(result["valid_pixels"], 2)
        for metric in ("pixel_accuracy", "mean_iou", "mean_accuracy", "frequency_weighted_iou", "macro_dice"):
            self.assertEqual(result[metric], 1.0)

    def test_fp_only_class_included_in_iou_and_dice_not_mean_accuracy(self):
        self.metrics.update(np.array([1, 1]), np.array([1, 2]))
        result = self.metrics.report()
        absent = result["per_class"][2]
        self.assertEqual(absent["gt_pixels"], 0)
        self.assertEqual(absent["fp"], 1)
        self.assertEqual(absent["iou"], 0.0)
        self.assertEqual(absent["dice"], 0.0)
        self.assertEqual(absent["precision"], 0.0)
        self.assertIsNone(absent["recall"])
        self.assertEqual(result["mean_iou"], 0.25)
        self.assertEqual(result["mean_accuracy"], 0.5)
        self.assertEqual(result["frequency_weighted_iou"], 0.5)
        self.assertAlmostEqual(result["macro_dice"], 1 / 3)

    def test_class_missing_from_gt_and_predictions_has_undefined_scores(self):
        self.metrics.update(np.array([1]), np.array([1]))
        result = self.metrics.report()
        for metric in ("iou", "dice", "precision", "recall"):
            self.assertIsNone(result["per_class"][2][metric])
        self.assertEqual(result["mean_iou"], 1.0)
        self.assertEqual(result["mean_accuracy"], 1.0)
        self.assertEqual(result["macro_dice"], 1.0)

    def test_known_matrix_hand_calculation(self):
        self.metrics.update(
            np.array([1, 1, 1, 1, 2, 2, 2, 2, 2]),
            np.array([0, 1, 1, 2, 0, 1, 2, 2, 2]),
        )
        result = self.metrics.report()
        expected = [[0, 0, 0], [1, 2, 1], [1, 1, 3]]
        self.assertEqual(result["confusion_matrix"], expected)
        self.assertAlmostEqual(result["pixel_accuracy"], 5 / 9)
        self.assertAlmostEqual(result["mean_iou"], 9 / 20)
        self.assertAlmostEqual(result["mean_accuracy"], 11 / 20)
        self.assertAlmostEqual(result["frequency_weighted_iou"], 41 / 90)
        self.assertAlmostEqual(result["macro_dice"], 13 / 21)
        for row, expected_counts, expected_scores in (
            (result["per_class"][1], (4, 3, 2, 1, 2), (2 / 5, 4 / 7, 2 / 3, 1 / 2)),
            (result["per_class"][2], (5, 4, 3, 1, 2), (1 / 2, 2 / 3, 3 / 4, 3 / 5)),
        ):
            self.assertEqual(tuple(row[key] for key in ("gt_pixels", "predicted_pixels", "tp", "fp", "fn")), expected_counts)
            for key, expected_score in zip(("iou", "dice", "precision", "recall"), expected_scores):
                self.assertAlmostEqual(row[key], expected_score)
        self.assertEqual(result["per_class"][1]["id"], 1)
        self.assertEqual(result["per_class"][1]["name"], "sidewalk")

    def test_updates_accumulate_and_do_not_mutate_inputs(self):
        target = np.array([[1, 2]], dtype=np.uint8)
        prediction = np.array([[0, 2]], dtype=np.int16)
        self.metrics.update(target, prediction)
        self.metrics.update(target[:, ::-1], prediction[:, ::-1])
        np.testing.assert_array_equal(self.metrics.matrix, [[0, 0, 0], [2, 0, 0], [0, 0, 2]])
        np.testing.assert_array_equal(target, [[1, 2]])
        np.testing.assert_array_equal(prediction, [[0, 2]])

    def test_no_valid_pixels(self):
        for target, prediction in (
            (np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
            (np.array([0, 255]), np.array([1, 2])),
        ):
            with self.subTest(target=target):
                metrics = ConfusionMatrix(self.names)
                metrics.update(target, prediction)
                result = metrics.report()
                self.assertEqual(result["valid_pixels"], 0)
                for metric in ("pixel_accuracy", "mean_iou", "mean_accuracy", "frequency_weighted_iou", "macro_dice"):
                    self.assertIsNone(result[metric])
                for row in result["per_class"]:
                    for key in ("iou", "dice", "precision", "recall"):
                        self.assertIsNone(row[key])
                    self.assertEqual(row["gt_pixels"], 0)
                    self.assertEqual(row["predicted_pixels"], 0)
                json.dumps(result, allow_nan=False)

    def test_new_matrix_report_has_no_defined_aggregate(self):
        result = self.metrics.report()
        self.assertIsNone(result["pixel_accuracy"])
        self.assertIsNone(result["mean_iou"])
        self.assertIsNone(result["frequency_weighted_iou"])

    def test_explicit_ignore_set_replaces_defaults(self):
        metrics = ConfusionMatrix(self.names, ignore_ids={255})
        metrics.update(np.array([0, 1]), np.array([0, 0]))
        result = metrics.report()
        self.assertFalse(result["per_class"][0]["ignored"])
        self.assertEqual(result["mean_iou"], 0.25)
        self.assertEqual(result["mean_accuracy"], 0.5)
        self.assertEqual(result["pixel_accuracy"], 0.5)
        with self.assertRaises(ValueError):
            ConfusionMatrix(self.names, ignore_ids=set()).update(np.array([255]), np.array([0]))

    def test_custom_ignored_class_still_has_prediction_column(self):
        metrics = ConfusionMatrix(self.names, ignore_ids={0, 2, 255, -1})
        metrics.update(np.array([2, -1, 1, 1]), np.array([1, 1, 2, 1]))
        result = metrics.report()
        self.assertEqual(result["valid_pixels"], 2)
        self.assertEqual(result["per_class"][2]["fp"], 1)
        self.assertEqual(result["per_class"][1]["fn"], 1)
        self.assertEqual(result["mean_iou"], 0.5)
        self.assertEqual(result["mean_accuracy"], 0.5)
        self.assertAlmostEqual(result["macro_dice"], 2 / 3)

    def test_invalid_constructor(self):
        cases = (
            ([], None, TypeError),
            ({}, None, ValueError),
            ({1: "sidewalk"}, None, ValueError),
            ({0: "unlabeled", 2: "road"}, None, ValueError),
            ({-1: "negative", 0: "unlabeled"}, None, ValueError),
            ({0.0: "float"}, None, TypeError),
            ({False: "boolean"}, None, TypeError),
            ({0: 17}, None, TypeError),
            (self.names, [0, 255], TypeError),
            (self.names, {0, 1.5}, TypeError),
            (self.names, {False}, TypeError),
        )
        for names, ignore_ids, exception in cases:
            with self.subTest(names=names, ignore_ids=ignore_ids):
                with self.assertRaises(exception):
                    ConfusionMatrix(names, ignore_ids)

    def test_shape_mismatch_even_with_same_size(self):
        with self.assertRaisesRegex(ValueError, "shapes must match"):
            self.metrics.update(np.array([[1, 2]]), np.array([[1], [2]]))
        np.testing.assert_array_equal(self.metrics.matrix, np.zeros((3, 3), dtype=np.int64))

    def test_invalid_arrays_and_dtypes(self):
        integer = np.array([1])
        invalids = ([1], np.array([1.0]), np.array([True]), np.array(["1"]), np.array([1], dtype=object), np.array([1j]))
        for invalid in invalids:
            for target, prediction in ((invalid, integer), (integer, invalid)):
                with self.subTest(target=target, prediction=prediction):
                    with self.assertRaises(TypeError):
                        self.metrics.update(target, prediction)
        with self.assertRaises(ValueError):
            self.metrics.update(np.array(1), np.array(1))

    def test_prediction_range_validated_even_on_ignored_gt(self):
        for target in (0, 255, 1):
            for prediction in (-1, 3, 255):
                with self.subTest(target=target, prediction=prediction):
                    with self.assertRaisesRegex(ValueError, "prediction IDs"):
                        self.metrics.update(np.array([target]), np.array([prediction]))

    def test_unknown_gt_rejected_without_partial_update(self):
        self.metrics.update(np.array([1]), np.array([1]))
        before = self.metrics.matrix.copy()
        for unknown in (-1, 3, 254, 256):
            with self.subTest(unknown=unknown):
                with self.assertRaisesRegex(ValueError, "unknown non-ignored target"):
                    self.metrics.update(np.array([1, unknown]), np.array([2, 1]))
                np.testing.assert_array_equal(self.metrics.matrix, before)
        with self.assertRaises(ValueError):
            self.metrics.update(np.array([1, 1]), np.array([2, 3]))
        np.testing.assert_array_equal(self.metrics.matrix, before)

    def test_large_unsigned_ids_do_not_wrap_into_valid_range(self):
        valid = np.array([1], dtype=np.uint64)
        invalid = np.array([2 ** 64 - 1], dtype=np.uint64)
        with self.assertRaises(ValueError):
            self.metrics.update(invalid, valid)
        with self.assertRaises(ValueError):
            self.metrics.update(valid, invalid)

    def test_configuration_is_copied(self):
        names = self.names.copy()
        ignored = {0, 255}
        metrics = ConfusionMatrix(names, ignored)
        names[1] = "changed"
        ignored.add(1)
        metrics.update(np.array([1]), np.array([1]))
        self.assertEqual(metrics.report()["per_class"][1]["name"], "sidewalk")
        self.assertEqual(metrics.report()["valid_pixels"], 1)

    def test_save_strict_json_csv_and_matrix_orientation(self):
        metrics = ConfusionMatrix({0: "unlabeled", 1: "人行道,sidewalk", 2: "road"})
        metrics.update(np.array([0, 255, 1, 1, 2]), np.array([2, 1, 0, 2, 2]))
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "nested" / "metrics"
            metrics.save(output_dir)
            self.assertEqual({path.name for path in output_dir.iterdir()}, {"metrics.json", "metrics.csv", "confusion_matrix.csv"})
            def reject_constant(token):
                self.fail(f"nonstandard JSON constant: {token}")
            result = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"), parse_constant=reject_constant)
            self.assertEqual(result, metrics.report())
            self.assertIn("metric_definitions", result)
            self.assertEqual(result["ignore_policy"]["ignore_ids"], [0, 255])
            self.assertIn("FN", result["ignore_policy"]["prediction"])
            with (output_dir / "metrics.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            aggregates = {row["metric"]: float(row["value"]) for row in rows if row["scope"] == "aggregate"}
            self.assertEqual(aggregates["pixel_accuracy"], 1 / 3)
            class_rows = [row for row in rows if row["scope"] == "per_class"]
            self.assertEqual(class_rows[1]["name"], "人行道,sidewalk")
            self.assertEqual(class_rows[0]["recall"], "")
            self.assertEqual(class_rows[1]["fn"], "2")
            with (output_dir / "confusion_matrix.csv").open(encoding="utf-8", newline="") as stream:
                matrix_rows = list(csv.reader(stream))
            self.assertEqual(matrix_rows[0][:2], ["gt_id", "gt_name"])
            self.assertEqual(matrix_rows[0][3], "prediction_1:人行道,sidewalk")
            self.assertEqual([[int(value) for value in row[2:]] for row in matrix_rows[1:]], result["confusion_matrix"])
            self.assertEqual(matrix_rows[2][0:2], ["1", "人行道,sidewalk"])

    def test_save_no_valid_pixels_as_json_null_and_csv_empty(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            self.metrics.save(output_dir)
            raw = (output_dir / "metrics.json").read_text(encoding="utf-8")
            self.assertNotIn("NaN", raw)
            self.assertNotIn("Infinity", raw)
            self.assertIn('"pixel_accuracy": null', raw)
            with (output_dir / "metrics.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(all(row["value"] == "" for row in rows if row["scope"] == "aggregate"))


if __name__ == "__main__":
    unittest.main()

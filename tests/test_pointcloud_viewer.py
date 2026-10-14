"""Offline point-cloud viewer tests; run in the vla-supervised WSL environment."""

from html.parser import HTMLParser
import json
from pathlib import Path
import re
import struct
import tempfile
import unittest
from unittest.mock import patch
import zlib

import numpy as np

from python import pointcloud_viewer as viewer


class ScriptParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts = []
        self.attributes = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        self.attributes.extend(attrs)
        if tag == "script":
            self.current = {"attrs": dict(attrs), "text": ""}
            self.scripts.append(self.current)

    def handle_data(self, data):
        if self.current is not None:
            self.current["text"] += data

    def handle_endtag(self, tag):
        if tag == "script":
            self.current = None


class PointcloudViewerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.xyz = np.array([[0, 0, 0], [1, 0, 1], [2, 1, 0], [3, 2, 1],
                             [4, 2, 0], [5, 3, 2], [6, 4, 0]], dtype=np.float32)
        self.pred = np.array([1, 0, 1, 2, 2, 0, 1], dtype=np.uint8)
        self.gt = np.array([1, 1, 1, 1, 0, 0, 2], dtype=np.uint8)
        self.covered = np.array([False, True, True, True, True, True, False])
        self.names = {0: "unlabeled", 1: "road", 2: "sidewalk"}
        self.palette = np.zeros((256, 3), dtype=np.uint8)
        self.palette[0] = (255, 0, 255)  # Must never leak into ignore/unlabeled colors.
        self.palette[1] = (10, 20, 30)
        self.palette[2] = (80, 90, 100)
        self.summary = {"total_points": 7, "coverage": 5 / 7,
                        "covered_points": {"mean_iou": np.float64(0.5), "point_accuracy": 0.6}}

    def arguments(self, **changes):
        values = dict(xyz=self.xyz, prediction=self.pred, target=self.gt,
                      covered=self.covered, class_names=self.names,
                      palette=self.palette, summary=self.summary)
        values.update(changes)
        return values

    def html(self, **changes):
        path = self.root / "nested" / "view.html"
        viewer.save_pointcloud_html(path, **self.arguments(**changes))
        text = path.read_text(encoding="utf-8")
        parser = ScriptParser()
        parser.feed(text)
        payload = next(script["text"] for script in parser.scripts
                       if script["attrs"].get("id") == "pointcloud-data")
        return path, text, parser, json.loads(payload)

    def test_prediction_colors_coverage_and_unlabeled(self):
        colors = viewer._display_colors(self.pred, self.gt, self.covered, self.palette)
        np.testing.assert_array_equal(colors["prediction"], [
            viewer.UNCOVERED_COLOR, viewer.UNLABELED_COLOR, self.palette[1], self.palette[2],
            self.palette[2], viewer.UNLABELED_COLOR, viewer.UNCOVERED_COLOR])
        np.testing.assert_array_equal(colors["coverage"], [
            viewer.UNCOVERED_COLOR, viewer.COVERED_COLOR, viewer.COVERED_COLOR,
            viewer.COVERED_COLOR, viewer.COVERED_COLOR, viewer.COVERED_COLOR,
            viewer.UNCOVERED_COLOR])
        for colors_for_mode in colors.values():
            self.assertEqual(colors_for_mode.dtype, np.uint8)

    def test_gt_is_full_scan_and_error_only_valid_covered(self):
        colors = viewer._display_colors(self.pred, self.gt, self.covered, self.palette)
        np.testing.assert_array_equal(colors["gt"], [self.palette[1], self.palette[1],
            self.palette[1], self.palette[1], viewer.IGNORED_COLOR,
            viewer.IGNORED_COLOR, self.palette[2]])
        np.testing.assert_array_equal(colors["error"], [viewer.IGNORED_COLOR,
            viewer.ERROR_COLOR, viewer.CORRECT_COLOR, viewer.ERROR_COLOR,
            viewer.IGNORED_COLOR, viewer.IGNORED_COLOR, viewer.IGNORED_COLOR])

    def test_html_is_self_contained_and_valid_json(self):
        path, text, parser, data = self.html()
        self.assertTrue(path.is_file())
        self.assertEqual(len(parser.scripts), 2)
        self.assertTrue(all("src" not in script["attrs"] for script in parser.scripts))
        self.assertFalse(any(name in {"src", "href"} for name, _ in parser.attributes))
        self.assertNotRegex(text, r"https?://|\bfetch\s*\(|XMLHttpRequest|import\s+.*from")
        self.assertIn("getContext('webgl'", text)
        self.assertIn("pointerdown", text)
        self.assertIn("wheel", text)
        self.assertIn("coveredOnly", text)
        self.assertIn("No automatic range crop", text)
        self.assertIn("PLY/NPZ preserve all points", text)
        self.assertIn("metrics use complete data", text)
        self.assertEqual(data["total_points"], 7)
        self.assertEqual(data["rendered_points"], 7)
        self.assertEqual(data["covered_points"], 5)
        self.assertEqual(data["summary"]["covered_points"]["mean_iou"], 0.5)
        np.testing.assert_array_equal(data["xyz"], self.xyz)
        np.testing.assert_array_equal(data["target"], self.gt)
        np.testing.assert_array_equal(data["colors"]["prediction"],
                                      viewer._display_colors(self.pred, self.gt, self.covered, self.palette)["prediction"])
        self.assertNotIn("JSON.stringify(data.summary", text)
        self.assertIn("summaryRows(data.summary)", text)

    def test_html_escapes_names_and_summary_without_injection(self):
        attack = '</script><script>alert("x")</script><img src=x onerror=alert(1)>&\u2028\u2029'
        _, text, parser, data = self.html(class_names={0: "unlabeled", 1: attack},
                                          summary={attack: attack, "mean_iou": 0.5})
        self.assertNotIn(attack, text)
        self.assertEqual(data["class_names"]["1"], attack)
        self.assertEqual(data["summary"][attack], attack)
        self.assertEqual(len(parser.scripts), 2)
        payload = parser.scripts[0]["text"]
        self.assertNotIn("<", payload)
        self.assertIn(r"\u003c/script\u003e", payload)
        self.assertIn(r"\u2028", payload)
        self.assertIn("text.textContent=label", text)
        self.assertIn("number.textContent=text", text)
        self.assertNotIn("innerHTML", text)

    def test_uniform_sampling_is_deterministic_and_keeps_full_bounds(self):
        xyz = np.arange(303, dtype=np.float64).reshape(101, 3)
        xyz[50] = (10000, -10000, 9000)  # Not sampled, but must determine framing.
        pred = np.ones(101, dtype=np.uint8)
        covered = np.ones(101, dtype=bool)
        source = xyz.copy()
        path = self.root / "sample.html"
        args = self.arguments(xyz=xyz, prediction=pred, target=None, covered=covered,
                              summary={"metric_points": 101})
        viewer.save_pointcloud_html(path, **args, max_points=4)
        first = path.read_text(encoding="utf-8")
        viewer.save_pointcloud_html(path, **args, max_points=4)
        self.assertEqual(first, path.read_text(encoding="utf-8"))
        data = json.loads(re.search(r'type="application/json">(.*?)</script>', first, re.S).group(1))
        self.assertEqual(data["sampled_indices"], [0, 33, 66, 100])
        self.assertEqual(data["total_points"], 101)
        self.assertEqual(data["rendered_points"], 4)
        self.assertEqual(data["summary"]["metric_points"], 101)
        np.testing.assert_array_equal(data["bounds"], [xyz.min(axis=0), xyz.max(axis=0)])
        np.testing.assert_array_equal(data["xyz"], xyz[[0, 33, 66, 100]])
        np.testing.assert_array_equal(xyz, source)
        np.testing.assert_array_equal(viewer._sample_indices(7, 1), [0])
        np.testing.assert_array_equal(viewer._sample_indices(7, 10), np.arange(7))
        np.testing.assert_array_equal(viewer._sample_indices(0, 10), [])

    def test_without_gt_and_empty_cloud(self):
        _, text, _, data = self.html(target=None)
        self.assertIsNone(data["target"])
        self.assertEqual(set(data["colors"]), {"prediction", "coverage"})
        self.assertIn("option.disabled=true", text)
        _, _, _, empty = self.html(xyz=np.empty((0, 3)), prediction=np.empty(0, np.uint8),
                                  target=None, covered=np.empty(0, bool))
        self.assertEqual(empty["xyz"], [])
        self.assertEqual(empty["rendered_points"], 0)
        self.assertEqual(empty["bounds"], [[0, 0, 0], [0, 0, 0]])

    def test_shape_length_and_dtype_validation_for_both_apis(self):
        bad_inputs = [
            {"xyz": np.zeros((7, 4))}, {"xyz": np.zeros(21)}, {"xyz": [[1, 2, 3]]},
            {"xyz": self.xyz.astype(complex)}, {"xyz": np.full((7, 3), np.nan)},
            {"prediction": np.zeros((7, 1), np.uint8)}, {"prediction": self.pred[:-1]},
            {"prediction": self.pred.astype(np.int64)}, {"prediction": self.pred.tolist()},
            {"covered": self.covered[:-1]}, {"covered": self.covered.astype(np.uint8)},
            {"target": self.gt[:-1]}, {"target": self.gt.reshape(7, 1)},
            {"target": self.gt.astype(float)}, {"target": np.full(7, -1)},
            {"target": np.full(7, 256)}, {"target": self.gt.tolist()},
            {"palette": self.palette[:3]}, {"palette": self.palette.astype(float)},
            {"class_names": {256: "bad"}}, {"class_names": {True: "bad"}},
            {"class_names": {1: 3}}, {"class_names": []},
            {"summary": []}, {"summary": {"bad": float("inf")}},
        ]
        for function in (viewer.save_pointcloud_html, viewer.save_pointcloud_preview):
            for changes in bad_inputs:
                with self.subTest(function=function.__name__, changed=list(changes)):
                    path = self.root / "should_not_exist"
                    with self.assertRaises((TypeError, ValueError)):
                        function(path, **self.arguments(**changes))
                    self.assertFalse(path.exists())

    def test_max_points_validation(self):
        for bad in (0, -1, 1.5, True, "60000"):
            with self.subTest(max_points=bad):
                with self.assertRaises((TypeError, ValueError)):
                    viewer.save_pointcloud_html(self.root / "bad.html", **self.arguments(), max_points=bad)

    def test_inputs_are_not_modified(self):
        snapshots = [item.copy() for item in (self.xyz, self.pred, self.gt, self.covered, self.palette)]
        self.html()
        for source, snapshot in zip((self.xyz, self.pred, self.gt, self.covered, self.palette), snapshots):
            np.testing.assert_array_equal(source, snapshot)

    def assert_png(self, path):
        raw = path.read_bytes()
        self.assertEqual(raw[:8], b"\x89PNG\r\n\x1a\n")
        offset, image_bytes, chunks = 8, bytearray(), []
        width = height = None
        while offset < len(raw):
            length = struct.unpack(">I", raw[offset:offset + 4])[0]
            kind = raw[offset + 4:offset + 8]
            contents = raw[offset + 8:offset + 8 + length]
            crc = struct.unpack(">I", raw[offset + 8 + length:offset + 12 + length])[0]
            self.assertEqual(zlib.crc32(kind + contents) & 0xffffffff, crc)
            chunks.append(kind)
            if kind == b"IHDR":
                width, height = struct.unpack(">II", contents[:8])
            elif kind == b"IDAT":
                image_bytes.extend(contents)
            offset += length + 12
        self.assertEqual(offset, len(raw))
        self.assertEqual(chunks[-1], b"IEND")
        self.assertGreater(width, 500)
        self.assertGreater(height, 500)
        self.assertGreater(len(zlib.decompress(image_bytes)), width * height)
        return raw

    def test_png_valid_and_retains_all_points_and_full_summary(self):
        from matplotlib.axes import Axes
        from mpl_toolkits.mplot3d.axes3d import Axes3D
        path = self.root / "nested" / "preview.png"
        calls = []
        scatter2d, scatter3d = Axes.scatter, Axes3D.scatter
        def track2d(axes, x, y, *args, **kwargs):
            calls.append(("top", np.asarray(x).copy(), kwargs["c"]))
            return scatter2d(axes, x, y, *args, **kwargs)
        def track3d(axes, x, y, z, *args, **kwargs):
            calls.append(("oblique", np.asarray(x).copy(), kwargs["c"]))
            return scatter3d(axes, x, y, z, *args, **kwargs)
        # Axes3D.scatter internally delegates to Axes.scatter; only the two
        # direct top/oblique callbacks are used to count each display mode.
        with patch.object(Axes, "scatter", track2d), patch.object(Axes3D, "scatter", track3d):
            viewer.save_pointcloud_preview(path, **self.arguments())
        raw = self.assert_png(path)
        direct_top = [call for call in calls if call[0] == "top" and len(call[1]) == len(self.xyz)]
        oblique = [call for call in calls if call[0] == "oblique"]
        self.assertGreaterEqual(len(direct_top), 4)
        self.assertEqual(len(oblique), 4)
        for _, x, _ in direct_top + oblique:
            self.assertEqual(len(x), len(self.xyz))
        np.testing.assert_array_equal(oblique[1][2],
                                      viewer._display_colors(self.pred, self.gt, self.covered, self.palette)["gt"] / 255)
        self.assertIn(b"Summary\x00", raw)
        self.assertIn(b'"mean_iou":0.5', raw)

    def test_empty_png_and_literal_class_text(self):
        path = self.root / "empty.png"
        viewer.save_pointcloud_preview(path, **self.arguments(
            xyz=np.empty((0, 3)), prediction=np.empty(0, np.uint8),
            target=None, covered=np.empty(0, bool), class_names={1: "$bad_math_{"}))
        self.assert_png(path)

    def test_compact_numeric_summary(self):
        summary = viewer._summary_text({"matrix": [[1, 2], [3, 4]], "note": "long" * 100,
                                       "full_scan": {"mean_iou": 0.5, "point_accuracy": 0.75},
                                       "coverage": 0.6, "total_points": 100})
        self.assertIn("full_scan.mIoU: 50.00%", summary)
        self.assertIn("point_accuracy: 75.00%", summary)
        self.assertIn("coverage: 60.00%", summary)
        self.assertIn("total_points: 100", summary)
        self.assertNotIn("matrix", summary)
        self.assertNotIn("long", summary)


if __name__ == "__main__":
    unittest.main()

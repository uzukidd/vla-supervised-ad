"""Regression tests for semantic fusion, taxonomy, and local Parquet decoding."""
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from sam3_sidewalk import (dataset_info, iter_samples, load_categories, make_palette,
                           merge_instance, parse_class_file, save_visualization)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write_config(self, text):
        path = self.root / "classes.txt"
        path.write_text(text, encoding="utf-8")
        return path

    def test_custom_names_prompts_and_bom(self):
        config = self.write_config("\ufeff# comment\nroad\nplants = tree | bush # prompt aliases\n")
        categories, labels = load_categories("custom", self.root, config)
        self.assertEqual(labels, {0: "unassigned", 1: "road", 2: "plants"})
        self.assertEqual(categories[1].prompts, ("tree", "bush"))

    def test_invalid_custom_entries(self):
        for text in ("road\nROAD", "person = ", "person = person | ", "unlabeled", "# empty"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_class_file(self.write_config(text))

    def test_taxonomy_ids_preserved_when_prompt_file_reordered(self):
        (self.root / "id2label.json").write_text(json.dumps({0: "unlabeled", 1: "flat-road", 2: "sky"}))
        categories, labels = load_categories("sidewalk", self.root, self.write_config("sky = sky\nflat-road = road"))
        self.assertEqual([(c.id, c.name) for c in categories], [(1, "flat-road"), (2, "sky")])
        self.assertEqual(labels[0], "unlabeled")
        with self.assertRaises(ValueError):
            load_categories("sidewalk", self.root, self.write_config("road\nsky"))

    def test_overlap_confidence_ties_and_uncovered_pixels(self):
        prediction = np.zeros((2, 3), dtype=np.uint8)
        score = np.full((2, 3), -np.inf, dtype=np.float32)
        merge_instance(prediction, score, np.array([[1, 1, 0], [0, 0, 0]]), .75, 1)
        merge_instance(prediction, score, np.array([[0, 1, 1], [0, 0, 0]]), .875, 2)
        merge_instance(prediction, score, np.array([[0, 0, 1], [1, 0, 0]]), .875, 3)
        np.testing.assert_array_equal(prediction, [[1, 2, 2], [3, 0, 0]])

    @staticmethod
    def encode(array):
        output = BytesIO()
        Image.fromarray(array).save(output, format="PNG")
        return {"bytes": output.getvalue(), "path": None}

    def make_shards(self):
        files = []
        for shard in range(2):
            rows = [{"pixel_values": self.encode(np.full((3, 4, 3), i, dtype=np.uint8)),
                     "label": self.encode(np.full((3, 4), i, dtype=np.uint8))}
                    for i in range(shard * 3, shard * 3 + 3)]
            path = self.root / f"train-{shard:05d}-of-00002.parquet"
            pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)
            files.append(path)
        return files

    def test_parquet_selected_indices_across_shards_and_palette_masks(self):
        files = self.make_shards()
        self.assertEqual(dataset_info(files, True)["rows"], 6)
        samples = list(iter_samples(files, self.root, 2, 3, True))
        self.assertEqual([row[0] for row in samples], [2, 3, 4])
        for index, image, target in samples:
            self.assertEqual(image.size, (4, 3))
            self.assertTrue((target == index).all())
        mask = Image.fromarray(np.array([[0, 1], [2, 255]], dtype=np.uint8)).convert("P")
        output = BytesIO()
        mask.save(output, format="PNG")
        rows = [{"pixel_values": self.encode(np.zeros((2, 2, 3), dtype=np.uint8)),
                 "label": {"bytes": output.getvalue(), "path": None}}]
        path = self.root / "palette.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        _, _, target = next(iter_samples([path], self.root, 0, 1, True))
        np.testing.assert_array_equal(target, [[0, 1], [2, 255]])

    def test_custom_reader_does_not_require_labels(self):
        path = self.root / "image_only.parquet"
        pq.write_table(pa.Table.from_pylist([{"pixel_values": self.encode(np.zeros((2, 2, 3), dtype=np.uint8))}]), path)
        dataset_info([path], False)
        _, _, target = next(iter_samples([path], self.root, 0, 1, False))
        self.assertIsNone(target)
        with self.assertRaises(ValueError):
            dataset_info([path], True)

    def test_visualization_saves_comparison_overlay_and_semantic_colors(self):
        for name in ("visualizations", "overlays", "color_masks"):
            (self.root / name).mkdir()
        image = Image.new("RGB", (64, 32), (50, 100, 150))
        prediction = np.zeros((32, 64), dtype=np.uint8)
        prediction[:, 32:] = 1
        target = prediction.copy()
        target[0] = 255
        save_visualization(self.root, "sample", image, prediction, target,
                           {0: "unlabeled", 1: "road"}, make_palette(), .5)
        for name, extension in (("visualizations", "jpg"), ("overlays", "jpg"), ("color_masks", "png")):
            self.assertTrue((self.root / name / f"sample.{extension}").is_file())
        with Image.open(self.root / "overlays/sample.jpg") as overlay:
            self.assertEqual(overlay.size, image.size)
        with Image.open(self.root / "color_masks/sample.png") as colored:
            np.testing.assert_array_equal(np.asarray(colored)[5, 40], make_palette()[1])


if __name__ == "__main__":
    unittest.main()

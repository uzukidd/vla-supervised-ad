"""Export integration tests using small camera masks and a full synthetic scan."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tasks.nuscenes_sam3.python.nuscenes_pointcloud import export_run
from tasks.sidewalk.python.sam3_sidewalk import make_palette


class ExportTests(unittest.TestCase):
    def test_source_masks_fuse_into_one_full_scan_and_unique_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "predictions").mkdir()
            config = {"status": "completed", "processed_images": 2, "mode": "lidarseg", "taxonomy": "lidarseg32",
                      "dataset_dir": "unused", "version": "v1.0-mini", "min_depth": 1.0}
            (source / "run_config.json").write_text(json.dumps(config))
            palette = make_palette()
            labels = {0: "ignore", 1: "animal", 2: "person"}
            (source / "classes.json").write_text(json.dumps([
                {"id": i, "name": name, "color_rgb": palette[i].tolist()} for i, name in labels.items()]))
            views = [{"sample_token": "sample", "scene_name": "scene-test", "lidar_token": "lidar",
                      "camera_token": f"camera_{i}", "camera": f"CAM_{i}", "image_path": f"image_{i}.jpg", "stem": f"mask_{i}"}
                     for i in range(2)]
            (source / "samples.jsonl").write_text("\n".join(json.dumps(view) for view in views))
            Image.fromarray(np.array([[1, 1], [0, 0]], dtype=np.uint8)).save(source / "predictions/mask_0.png")
            Image.fromarray(np.array([[2, 2], [0, 0]], dtype=np.uint8)).save(source / "predictions/mask_1.png")
            xyz = np.arange(15, dtype=np.float32).reshape(5, 3)
            class Dataset:
                def __init__(self, *args):
                    self.root = root
                def load_lidar(self, token):
                    return {"xyz": xyz, "raw_labels": np.array([1, 1, 2, 2, 0], dtype=np.uint8),
                            "intensity": np.zeros(5), "ring": np.zeros(5)}
                def projection(self, view, depth):
                    return {"point_indices": np.array([0, 1]) if view["camera"] == "CAM_0" else np.array([0, 2]),
                            "pixels": np.array([[0, 0], [1, 0]]), "image_size_wh": [2, 2]}
            output = root / "output"
            with patch("tasks.nuscenes_sam3.python.nuscenes_pointcloud.NuScenesMini", Dataset):
                state = export_run(source, output, make_preview=False)
            self.assertEqual(state["status"], "completed")
            self.assertEqual(state["processed_frames"], 1)
            frame = next((output / "pointclouds").iterdir())
            with np.load(frame / "points.npz") as data:
                np.testing.assert_array_equal(data["xyz"], xyz)
                np.testing.assert_array_equal(data["prediction"], [1, 1, 2, 0, 0])
                np.testing.assert_array_equal(data["covered"], [1, 1, 1, 0, 0])
                np.testing.assert_array_equal(data["observation_count"], [2, 1, 1, 0, 0])
                self.assertEqual(data["camera_names"].tolist(), ["CAM_0", "CAM_1"])
            report = json.loads((output / "evaluation/metrics.json").read_text())
            self.assertEqual(report["covered_points"]["valid_points"], 3)
            self.assertEqual(report["covered_points"]["point_accuracy"], 1.0)
            self.assertEqual(report["full_scan"]["valid_points"], 4)
            self.assertEqual(report["full_scan"]["point_accuracy"], .75)
            self.assertEqual(report["coverage"]["valid_gt_coverage"], .75)
            self.assertEqual(report["coverage"]["total_points"], 5)
            self.assertTrue((frame / "bev_metadata.json").is_file())
            with np.load(frame / "bev_raster.npz") as raster:
                occupied = raster["occupied"]
                self.assertEqual(occupied.shape, (500, 500))
                self.assertFalse(any("drivable" in key or "obstacle" in key for key in raster.files))
                self.assertNotIn("point_index", raster.files)
                self.assertEqual(int(raster["point_count"].sum()), 5)
                self.assertEqual(int(occupied.sum()), 5)
                np.testing.assert_array_equal(np.sort(raster["prediction"][occupied]), [0, 0, 1, 1, 2])
                np.testing.assert_array_equal(np.sort(raster["target"][occupied]), [0, 1, 1, 2, 2])
                self.assertTrue((raster["prediction_count"][occupied] == 1).all())
                self.assertTrue((raster["point_count"][~occupied] == 0).all())
                self.assertTrue((raster["prediction"][~occupied] == 255).all())
            self.assertFalse(list(frame.glob("*.png")))
            self.assertTrue((frame / 'ground/ground_grid.npz').is_file())
            self.assertFalse(list((frame / 'ground').glob('*.png')))
            self.assertFalse(list(frame.glob("*.ply")))
            self.assertFalse(list(output.rglob("*.html")))
            preview_output = root / "preview"
            with patch("tasks.nuscenes_sam3.python.nuscenes_pointcloud.NuScenesMini", Dataset):
                export_run(source, preview_output, bev_resolution=1.0)
            preview_frame = next((preview_output / "pointclouds").iterdir())
            for name in ("bev_prediction.png", "bev_ground_truth.png", "bev_errors.png", "bev_coverage.png", "bev_density.png", "bev_comparison.png"):
                self.assertTrue((preview_frame / name).is_file())
            for name in ('ground_plane.png', 'ground_sources.png', 'ground_comparison.png'):
                self.assertTrue((preview_frame / 'ground' / name).is_file())
            self.assertFalse(list(preview_frame.glob("*drivable*")))
            with Image.open(preview_frame / "bev_prediction.png") as image:
                self.assertEqual(image.size, (100, 100))
            auto_output = root / "automatic"
            with patch("tasks.nuscenes_sam3.python.nuscenes_pointcloud.NuScenesMini", Dataset):
                export_run(source, auto_output, make_preview=False, bev_roi_size=0)
            auto_frame = next((auto_output / "pointclouds").iterdir())
            auto_meta = json.loads((auto_frame / "bev_metadata.json").read_text())
            self.assertEqual(auto_meta["grid_mode"], "automatic_extent")
            self.assertEqual(json.loads((auto_output / "evaluation/metrics.json").read_text()), report)
            self.assertFalse(list(preview_frame.glob("*.ply")))
            self.assertFalse(list(preview_output.rglob("*.html")))
            self.assertEqual(json.loads((preview_output / "evaluation/metrics.json").read_text()), report)
            with patch("tasks.nuscenes_sam3.python.nuscenes_pointcloud.NuScenesMini", Dataset), self.assertRaises(FileExistsError):
                export_run(source, output, make_preview=False)


if __name__ == "__main__":
    unittest.main()

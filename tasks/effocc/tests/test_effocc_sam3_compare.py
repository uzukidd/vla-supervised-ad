"""Same-frame matching and independent labels for SAM3 / EFFOcc comparisons."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tasks.effocc.python.effocc_sam3_compare import CAMERAS, compare_frame, comparison_sheet, find_projections, parser, sam3_overlay
from tasks.sidewalk.python.sam3_sidewalk import make_palette


class SAM3ComparisonTests(unittest.TestCase):
    def test_sam3_unassigned_preserves_original_and_color_matches_shared_id(self):
        image = Image.new('RGB', (3, 2), (20, 40, 60))
        prediction = np.array([[0, 4, 11], [0, 0, 0]], dtype=np.uint8)
        palette = make_palette()
        result = np.asarray(sam3_overlay(image, prediction, palette, .5))
        original = np.asarray(image)
        np.testing.assert_array_equal(result[prediction == 0], original[prediction == 0])
        np.testing.assert_array_equal(result[prediction != 0], np.rint(original[prediction != 0] * .5 + palette[prediction[prediction != 0]] * .5))
        with self.assertRaises(ValueError):
            sam3_overlay(image, np.full((2, 3), 17, dtype=np.uint8), palette, .5)
        with self.assertRaises(ValueError):
            sam3_overlay(image, prediction, palette, -1)

    def test_three_columns_and_camera_rows_remain_distinct(self):
        panels = [(name, Image.new('RGB', (16, 9), (10, 20, 30)),
                         Image.new('RGB', (16, 9), (40, 50, 60)),
                         Image.new('RGB', (16, 9), (70, 80, 90))) for name in CAMERAS]
        result = comparison_sheet(panels, make_palette(), 'Comparison')
        self.assertEqual(result.size, (1920, 45 + 6 * 388 + 180))
        for row in range(6):
            y = 45 + row * 388 + 28 + 100
            self.assertEqual(result.getpixel((100, y)), (10, 20, 30))
            self.assertEqual(result.getpixel((740, y)), (40, 50, 60))
            self.assertEqual(result.getpixel((1380, y)), (70, 80, 90))

    def test_defaults_match_nuscenes_16_prompts(self):
        args = parser().parse_args(['--input-dir', 'example'])
        self.assertEqual(args.max_samples, 1)
        self.assertEqual(args.classes_file.name, 'nuscenes_classes.txt')
        self.assertEqual(args.alpha, .45)
        self.assertEqual(args.precision, 'bf16')

    def test_six_original_frames_masks_and_projections_keep_separate_label_meanings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame, projection, output = root / 'frame', root / 'projection', root / 'out'
            frame.mkdir()
            projection.mkdir()
            (frame / 'sample.json').write_text(json.dumps({'input': {'sample_token': 'sample'}}))
            cameras = []
            projection_records = []
            occ = np.array([[0, 4, 255], [11, 255, 4]], dtype=np.uint8)
            sam = np.array([[0, 11, 11], [4, 0, 0]], dtype=np.uint8)
            for name in CAMERAS:
                image_path = root / f'{name}.png'
                Image.new('RGB', (3, 2), (20, 40, 60)).save(image_path)
                Image.fromarray(occ).save(projection / f'{name}_class_ids.png')
                cameras.append({'camera': name, 'camera_token': name + '_token', 'image_path': image_path})
                projection_records.append({'camera': name, 'camera_token': name + '_token'})
            projection_config = {'sample_token': 'sample', 'status': 'completed', 'reference_camera': 'CAM_FRONT_LEFT', 'cameras': projection_records}
            (projection / 'projection_config.json').write_text(json.dumps(projection_config))
            class Segmenter:
                calls = 0
                def predict(self, image, categories):
                    self.calls += 1
                    return sam.copy(), [{'class_id': 11, 'score': .9}]
            segmenter = Segmenter()
            with patch('tasks.effocc.python.effocc_sam3_compare.camera_geometry', return_value=cameras):
                report = compare_frame(object(), frame, projection, output, segmenter, [], .45)
            self.assertEqual(segmenter.calls, 6)
            self.assertEqual(report['status'], 'completed')
            self.assertEqual(report['evaluation'], None)
            self.assertEqual(len(report['cameras']), 6)
            for name in CAMERAS:
                with Image.open(output / f'{name}_sam3_class_ids.png') as image:
                    np.testing.assert_array_equal(np.asarray(image), sam)
                with Image.open(output / f'{name}_occupancy_class_ids.png') as image:
                    np.testing.assert_array_equal(np.asarray(image), occ)
                with Image.open(projection / f'{name}_class_ids.png') as image:
                    np.testing.assert_array_equal(np.asarray(image), occ)
            self.assertTrue((output / 'six_cameras_comparison.jpg').is_file())
            self.assertFalse((output / 'metrics.json').exists())
            projection_config['cameras'][0]['camera_token'] = 'different_frame'
            (projection / 'projection_config.json').write_text(json.dumps(projection_config))
            with patch('tasks.effocc.python.effocc_sam3_compare.camera_geometry', return_value=cameras), self.assertRaises(ValueError):
                compare_frame(object(), frame, projection, root / 'bad', segmenter, [])

    def test_projection_reuse_requires_matching_sample_and_complete_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = root / 'frame'
            projection = frame / 'camera_projections'
            projection.mkdir(parents=True)
            (frame / 'sample.json').write_text(json.dumps({'input': {'sample_token': 'sample'}}))
            (projection / 'projection_config.json').write_text(json.dumps({'sample_token': 'sample', 'status': 'completed'}))
            with patch('tasks.effocc.python.effocc_sam3_compare.render_frame') as renderer:
                self.assertEqual(find_projections(None, frame, None, 'cpu', .45, root / 'fallback'), projection)
                renderer.assert_not_called()
            (projection / 'projection_config.json').write_text(json.dumps({'sample_token': 'different', 'status': 'completed'}))
            with self.assertRaises(ValueError):
                find_projections(None, frame, projection, 'cpu', .45, root / 'fallback')


if __name__ == '__main__':
    unittest.main()

"""Complete-task launcher: stage translation, failure handling and result collection."""
from pathlib import Path
import json
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from shared.python.one_click import build_stages, collect_results, main, refresh_results


class OneClickTests(unittest.TestCase):
    def test_six_camera_sample_counts_and_cpu_mode(self):
        stages = build_stages('nuscenes_sam3', 3, 2, Path('out'), 'cpu')
        args = stages[0]['args']
        self.assertEqual(args[args.index('--max-images') + 1], '18')
        self.assertEqual(args[args.index('--start-index') + 1], '12')
        self.assertEqual(args[args.index('--precision') + 1], 'fp32')
        self.assertIn('--export-pointcloud', args)
        self.assertEqual(args[args.index('--bev-roi-size') + 1], '100')
        for task in ('sidewalk', 'nuimages'):
            args = build_stages(task, 3, 2, Path('out'), 'cuda')[0]['args']
            self.assertEqual(args[args.index('--max-images') + 1], '3')
            self.assertEqual(args[args.index('--start-index') + 1], '2')

    def test_effocc_two_stages_preserve_sample_selection_and_release_process(self):
        stages = build_stages('effocc', 2, 4, Path('out'), 'cuda')
        self.assertEqual(len(stages), 2)
        first, second = stages
        self.assertTrue(first['module'].endswith('effocc_mini'))
        self.assertTrue(second['module'].endswith('effocc_sam3_compare'))
        self.assertNotIn('--no-camera-projection', first['args'])
        self.assertEqual(first['args'][first['args'].index('--start-index') + 1], '4')
        self.assertEqual(second['args'][second['args'].index('--input-dir') + 1], str(Path('out/raw')))
        self.assertEqual(second['args'][second['args'].index('--max-samples') + 1], '2')

    def test_invalid_selection_rejected(self):
        for task, samples, start, device in (('bad', 1, 0, 'cpu'), ('effocc', 0, 0, 'cuda'),
                                             ('sidewalk', 1, -1, 'cuda'), ('nuimages', 1, 0, 'bad')):
            with self.subTest(task=task, samples=samples), self.assertRaises(ValueError):
                build_stages(task, samples, start, Path('out'), device)

    @staticmethod
    def image(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (8, 4), (30, 50, 70)).save(path)

    def test_collect_sam3_gt_images_and_metrics_result_page(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.image(root / 'raw/visualizations/sample.jpg')
            (root / 'raw/metrics.json').write_text('{"mean_iou":0.5}')
            images = collect_results('sidewalk', root)
            self.assertEqual(len(images), 1)
            self.assertTrue((root / images[0]['path']).is_file())
            self.assertEqual(json.loads((root / 'results/image_or_camera_metrics_metrics.json').read_text())['mean_iou'], .5)
            page = (root / 'result.html').read_text()
            self.assertIn(images[0]['path'], page)
            self.assertNotIn('https://', page)
            self.assertIn('Dataset ground truth', page)

    def test_collect_effocc_final_and_per_camera_comparisons(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.image(root / 'comparison/sample/six_cameras_comparison.jpg')
            self.image(root / 'comparison/sample/CAM_FRONT_comparison.jpg')
            images = collect_results('effocc', root)
            self.assertEqual(len(images), 2)
            self.assertIn('independent predictions', (root / 'result.html').read_text())
            self.assertFalse(list((root / 'results').glob('*metrics*')))

    def test_nuscenes_results_require_bev_and_show_bev_before_cameras(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.image(root / 'raw/visualizations/camera.jpg')
            with self.assertRaisesRegex(RuntimeError, 'BEV output is missing'):
                collect_results('nuscenes_sam3', root)
            frame = root / 'raw/pointcloud_results/pointclouds/frame'
            frame.mkdir(parents=True)
            np.savez_compressed(frame / 'points.npz', xyz=np.zeros((0, 3)))
            for name in ('bev_prediction.png', 'bev_ground_truth.png', 'bev_comparison.png'):
                self.image(frame / name)
            with self.assertRaisesRegex(RuntimeError, 'ground_plane'):
                collect_results('nuscenes_sam3', root)
            for name in ('ground_plane.png', 'ground_sources.png', 'ground_comparison.png'):
                self.image(frame / 'ground' / name)
            images = collect_results('nuscenes_sam3', root)
            self.assertEqual(len(images), 7)
            self.assertTrue(all(row['path'].startswith('results/BEV/frame/') for row in images[:6]))
            page = (root / 'result.html').read_text()
            self.assertLess(page.index('id="bev"'), page.index('id="cameras"'))
            self.assertIn('results/BEV/frame/ground_plane.png', page)
            self.assertTrue((root / 'results/BEV/frame/bev_prediction.png').is_file())

    def test_refresh_completed_nuscenes_run_uses_saved_points_without_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'run_config.json').write_text(json.dumps({'task': 'nuscenes_sam3', 'status': 'completed', 'samples': 1}))
            raw = root / 'raw'
            frame = raw / 'pointcloud_results/pointclouds/frame'
            frame.mkdir(parents=True)
            (raw / 'run_config.json').write_text(json.dumps({'taxonomy': 'lidarseg16'}))
            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
            np.savez_compressed(frame / 'points.npz', xyz=points, prediction=np.full(3, 11, dtype=np.uint8))
            (frame / 'bev_metadata.json').write_text(json.dumps({'grid_mode': 'fixed_roi', 'resolution_m_per_pixel': .2,
                'x_min_m': -5., 'y_min_m': -5., 'x_max_m': 5., 'y_max_m': 5., 'width': 50, 'height': 50}))
            for name in ('bev_prediction.png', 'bev_ground_truth.png', 'bev_comparison.png'):
                self.image(frame / name)
            saved = (frame / 'points.npz').read_bytes()
            with patch('shared.python.one_click.subprocess.run') as runner:
                config = refresh_results('nuscenes_sam3', root)
                runner.assert_not_called()
            self.assertEqual(config['status'], 'completed')
            self.assertEqual(config['comparison_images'], 6)
            self.assertEqual((frame / 'points.npz').read_bytes(), saved)
            self.assertTrue((root / 'results/BEV/frame/ground_plane.png').is_file())
            # Refresh is idempotent when the additional presentation products exist.
            with patch('shared.python.one_click.subprocess.run') as runner:
                refresh_results('nuscenes_sam3', root)
                runner.assert_not_called()
            (root / 'run_config.json').write_text(json.dumps({'task': 'nuscenes_sam3', 'status': 'running'}))
            with self.assertRaises(ValueError):
                refresh_results('nuscenes_sam3', root)

    def test_empty_comparison_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(RuntimeError):
            collect_results('effocc', Path(directory))

    def test_dry_run_and_failed_stage_do_not_claim_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'out'
            with patch('shared.python.one_click.ensure_environment'), patch('shared.python.one_click.subprocess.run') as runner:
                main(['--task', 'effocc', '--output-dir', str(output), '--dry-run'])
                runner.assert_not_called()
            self.assertFalse(output.exists())
            with patch('shared.python.one_click.ensure_environment'), patch('shared.python.one_click.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'test')):
                with self.assertRaises(subprocess.CalledProcessError):
                    main(['--task', 'effocc', '--output-dir', str(output)])
            config = json.loads((output / 'run_config.json').read_text())
            self.assertEqual(config['status'], 'failed')
            self.assertEqual(config['completed_stages'], 0)
            self.assertFalse((output / 'result.html').exists())


if __name__ == '__main__':
    unittest.main()

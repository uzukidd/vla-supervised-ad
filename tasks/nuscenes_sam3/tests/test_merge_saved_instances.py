"""Saved instance merging keeps source files and deduplicates group hull support."""
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
from PIL import Image

from tasks.nuscenes_sam3.python.merge_saved_instances import merge_frame, write_merge_page
from tasks.nuscenes_sam3.python.instance_hulls import points_in_convex_polygon


class MergeSavedTests(unittest.TestCase):
    def test_saved_frame_to_merged_hulls_and_offline_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'source', root / 'merged/frame'
            source.mkdir()
            xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0], [10, 10, 0]], dtype=np.float32)
            instances = [
                {'global_instance_id': 1, 'camera': 'CAM_A', 'class_id': 4, 'class_name': 'car', 'score': .8,
                 'point_indices': [0, 1, 2, 3]},
                {'global_instance_id': 2, 'camera': 'CAM_B', 'class_id': 4, 'class_name': 'car', 'score': .9,
                 'point_indices': [0, 1, 2, 4]},
                {'global_instance_id': 3, 'camera': 'CAM_A', 'class_id': 11, 'class_name': 'driveable_surface', 'score': .7,
                 'point_indices': [5]},
            ]
            data = {'metadata': {'roi_size_m': 100, 'width': 500, 'height': 500, 'resolution_m_per_pixel': .2}, 'instances': instances}
            path = source / 'instance_hulls.json'
            path.write_text(json.dumps(data))
            original_text = path.read_bytes()
            ids = np.array([[1, 1, 1, 1, 0, 3], [2, 2, 2, 0, 2, 0]], dtype=np.int32)
            np.savez_compressed(source / 'points.npz', xyz=xyz, camera_instance_ids=ids, camera_names=np.array(['CAM_A', 'CAM_B']))
            summary = merge_frame(source, output)
            self.assertEqual(summary['merged_group_count'], 1)
            self.assertEqual(summary['foreground_with_roi_points_before'], 2)
            self.assertEqual(summary['foreground_with_roi_points_after'], 1)
            result = json.loads((output / 'merged_hulls.json').read_text())
            merged = next(row for row in result['groups'] if row['merged'])
            self.assertEqual(merged['member_instance_ids'], [1, 2])
            self.assertEqual(merged['point_indices'], [0, 1, 2, 3, 4])
            self.assertTrue(points_in_convex_polygon(xyz[:5, :2], np.asarray(merged['hull_xy'])).all())
            with np.load(output / 'merged_points.npz', allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays['camera_instance_ids'], ids)
                np.testing.assert_array_equal(arrays['camera_merged_ids'], np.where(ids == 2, 1, ids))
                np.testing.assert_array_equal(arrays['xyz'], xyz)
            self.assertEqual(path.read_bytes(), original_text)
            with Image.open(output / 'bev_after_merge.png') as image:
                self.assertEqual(image.size, (500, 500))
            self.assertTrue((output / 'match_pairs.csv').exists())
            write_merge_page(root / 'merged', [output])
            self.assertIn('merge_comparison.png', (root / 'merged/result.html').read_text())
            with self.assertRaises(FileExistsError):
                merge_frame(source, output)


if __name__ == '__main__':
    unittest.main()

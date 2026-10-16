"""Camera association and original-image overlay checks for EFFOcc projections."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tasks.effocc.python.effocc_cameras import CAMERAS, camera_geometry, contact_sheet, overlay_projection, pose_matrix, projection_colors
from tasks.sidewalk.python.sam3_sidewalk import make_palette


class CameraProjectionTests(unittest.TestCase):
    def test_full_pose_chain_includes_camera_time_and_reference_time(self):
        q = [1, 0, 0, 0]
        class Dataset:
            def __init__(self):
                self._keyframes = {'sample': {name: {'token': name, 'timestamp': i, 'ego_pose_token': f'pose{i}',
                                                     'calibrated_sensor_token': f'cal{i}', 'filename': name + '.jpg'}
                                                for i, name in enumerate(CAMERAS)}}
                self.tables = {'ego_pose': {}, 'calibrated_sensor': {}}
                for i in range(6):
                    self.tables['ego_pose'][f'pose{i}'] = {'rotation': q, 'translation': [100 + i, 200, 0]}
                    self.tables['calibrated_sensor'][f'cal{i}'] = {'rotation': q, 'translation': [2, 3, 1],
                                                                    'camera_intrinsic': np.eye(3).tolist()}
            def _get(self, table, token):
                return self.tables[table][token]
            def _media_path(self, filename, name):
                return Path(filename)
        cameras = camera_geometry(Dataset(), 'sample', 'CAM_FRONT_LEFT')
        self.assertEqual([camera['camera'] for camera in cameras], list(CAMERAS))
        np.testing.assert_allclose(cameras[0]['camera_to_grid'][:3, 3], [2, 3, 1])
        np.testing.assert_allclose(cameras[1]['camera_to_grid'][:3, 3], [3, 3, 1])
        np.testing.assert_allclose(cameras[5]['camera_to_grid'][:3, 3], [7, 3, 1])
        with self.assertRaises(ValueError):
            camera_geometry(Dataset(), 'sample', 'bad')

    def test_projection_overlay_only_changes_ray_hits(self):
        image = Image.new('RGB', (3, 2), (20, 40, 60))
        labels = np.array([[255, 4, 255], [0, 255, 11]], dtype=np.uint8)
        palette = make_palette()
        palette[0] = (100, 100, 100)
        before = np.asarray(image).copy()
        result = np.asarray(overlay_projection(image, labels, palette, .5))
        np.testing.assert_array_equal(result[labels == 255], before[labels == 255])
        np.testing.assert_array_equal(result[labels != 255], np.rint(before[labels != 255] * .5 + palette[labels[labels != 255]] * .5).astype(np.uint8))
        np.testing.assert_array_equal(np.asarray(projection_colors(labels, palette))[0, 0], [255, 255, 255])
        np.testing.assert_array_equal(np.asarray(image), before)
        with self.assertRaises(ValueError):
            overlay_projection(image, labels, palette, 2)

    def test_contact_sheet_contains_all_six_panels_without_shape_change_in_source(self):
        palette = make_palette()
        images = [(name, Image.new('RGB', (160, 90), (20 * i, 10, 15))) for i, name in enumerate(CAMERAS)]
        sheet = contact_sheet(images, ('others',), palette, 'Six camera test', legend=False)
        self.assertEqual(sheet.width, 1920)
        for i, (_, image) in enumerate(images):
            self.assertEqual(image.size, (160, 90))
            pixel = sheet.getpixel(((i % 3) * 640 + 300, 45 + (i // 3) * 388 + 28 + 180))
            self.assertEqual(pixel, (20 * i, 10, 15))


if __name__ == '__main__':
    unittest.main()

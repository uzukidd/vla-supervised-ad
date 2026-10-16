"""SAM3 original mask streaming, point ownership and packed-mask preservation."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

from shared.python.sam3 import Category, Sam3Segmenter
from tasks.nuscenes_sam3.python.instance_bev import InstancePointCollector, instance_overlay


class InstancePointTests(unittest.TestCase):
    def test_original_masks_preserved_but_points_exclusive_by_score(self):
        collector = InstancePointCollector((4, 3), np.array([10, 20, 30]), np.array([[0, 0], [1, 0], [2, 0]]))
        first = np.zeros((3, 4), dtype=bool)
        first[0, :2] = True
        second = np.zeros((3, 4), dtype=bool)
        second[0, 1:3] = True
        third = second.copy()
        for index, (mask, score, label) in enumerate(((first, .6, 4), (second, .8, 7), (third, .8, 4)), 1):
            collector(mask, {'instance_id': index, 'class_id': label, 'class_name': str(label), 'score': score})
        np.testing.assert_array_equal(collector.point_ids, [1, 2, 2])
        instances = collector.instance_points('CAM_FRONT', 100)
        self.assertEqual(instances[0]['global_instance_id'], 101)
        self.assertEqual(instances[0]['point_indices'], [10])
        self.assertEqual(instances[1]['point_indices'], [20, 30])
        self.assertEqual(instances[2]['point_indices'], [])
        self.assertEqual(instances[0]['raw_projected_point_count'], 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'masks.npz'
            collector.save_masks(path)
            with np.load(path, allow_pickle=False) as data:
                for i, expected in enumerate((first, second, third)):
                    actual = np.unpackbits(data['packed_masks'][i], count=12, bitorder='little').reshape(3, 4).astype(bool)
                    np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(data['projected_instance_ids'], collector.point_ids)
                np.testing.assert_array_equal(data['instance_map'], collector.image_ids)
        image = Image.new('RGB', (4, 3), (20, 30, 40))
        result = np.asarray(instance_overlay(image, collector.image_ids, 100))
        self.assertTrue((result[1] == (20, 30, 40)).all())

    def test_semantic_predict_stream_callback_keeps_original_return_contract(self):
        import torch
        from contextlib import nullcontext
        image = Image.new('RGB', (4, 3))
        masks = torch.zeros((2, 1, 3, 4))
        masks[0, 0, 0, :2] = 1
        masks[1, 0, 0, 1:3] = 1
        segmenter = object.__new__(Sam3Segmenter)
        segmenter.torch = torch
        segmenter.precision = 'fp32'
        segmenter.mask_threshold = .5
        segmenter.processor = Mock()
        segmenter.processor.set_image.return_value = {}
        segmenter.processor.set_text_prompt.return_value = {'scores': torch.tensor([.6, .8]),
                                                           'boxes': torch.zeros((2, 4)), 'masks_logits': masks}
        collector = InstancePointCollector((4, 3), np.arange(3), np.array([[0, 0], [1, 0], [2, 0]]))
        semantic, records = segmenter.predict(image, [Category(4, 'car', ('car',))], instance_callback=collector)
        np.testing.assert_array_equal(semantic[0], [4, 4, 4, 0])
        np.testing.assert_array_equal(collector.point_ids, [1, 2, 2])
        self.assertEqual(len(records), 2)
        self.assertNotIn('instance_id', records[0])
        without_callback, old_records = segmenter.predict(image, [Category(4, 'car', ('car',))])
        np.testing.assert_array_equal(without_callback, semantic)
        self.assertEqual(records, old_records)

    def test_callback_rejects_wrong_shape_id_and_projection(self):
        with self.assertRaises(ValueError):
            InstancePointCollector((4, 3), np.array([1]), np.array([[4, 0]]))
        with self.assertRaises(ValueError):
            InstancePointCollector((4, 3), np.array([1, 1]), np.array([[0, 0], [1, 0]]))
        collector = InstancePointCollector((4, 3), np.array([], dtype=int), np.empty((0, 2), dtype=int))
        with self.assertRaises(ValueError):
            collector(np.zeros((3, 4), dtype=bool), {'instance_id': 2, 'score': .8})


if __name__ == '__main__':
    unittest.main()

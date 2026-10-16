"""Operator and shape checks for the inference-only EFFOcc adapter."""
from pathlib import Path
import sys
import unittest
import tempfile

import numpy as np
from PIL import Image
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tasks.effocc.python.effocc_model import ImageBackbone, ImageFPN, ViewTransformer, OccHead, weighted_bev_sum
from tasks.effocc.python.effocc_mini import top_bev, save_outputs


class EFFOccModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_weighted_bev_pool_matches_reference_and_chunking(self):
        coords = torch.tensor([[[[[.1, .1, .1], [.2, .2, .2]]],
                                [[[1.1, .1, .1], [8., 8., 8.]]]]])
        depth = torch.tensor([[[[.25, .5]], [[.75, .5]]]])
        features = torch.tensor([[[[2., 4.]], [[3., 6.]]]])
        expected = torch.tensor([[[[2.5, 1.5]], [[3.75, 2.25]]]])
        output = weighted_bev_sum(coords, depth, features, (0, 0, 0), (1, 1, 1), (2, 1, 1), chunk_size=1)
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(output, weighted_bev_sum(coords, depth, features, (0, 0, 0), (1, 1, 1), (2, 1, 1), chunk_size=100))
        # No valid points yields a correctly shaped all-zero map.
        empty = weighted_bev_sum(coords + 100, depth, features, (0, 0, 0), (1, 1, 1), (2, 1, 1))
        self.assertEqual(tuple(empty.shape), (1, 2, 1, 2))
        self.assertFalse(empty.any())

    def test_camera_frustum_and_post_transform_inversion(self):
        vt = ViewTransformer().eval()
        transform = torch.eye(4)[None]
        transform[:, :3, 3] = torch.tensor([1., 2., 3.])
        K = torch.eye(3)[None]
        post_rot = torch.diag(torch.tensor([2., 2., 1.]))[None]
        post_tran = torch.tensor([[10., 20., 0.]])
        result = vt.coordinates(transform, K, post_rot, post_tran)
        self.assertEqual(tuple(result.shape), (1, 88, 32, 88, 3))
        torch.testing.assert_close(result[0, 0, 0, 0], torch.tensor([-4., -8., 4.]))
        torch.testing.assert_close(result[0, -1, -1, -1], torch.tensor([(703 - 10) / 2 * 44.5 + 1,
                                                                                      (255 - 20) / 2 * 44.5 + 2, 47.5]))

    def test_resnet_neck_and_head_shapes_and_relu(self):
        with torch.inference_mode():
            backbone = ImageBackbone().eval()
            neck = ImageFPN().eval()
            inputs = backbone(torch.zeros(1, 3, 256, 704))
            self.assertEqual([tuple(x.shape) for x in inputs], [(1, 128, 32, 88), (1, 256, 16, 44), (1, 512, 8, 22)])
            self.assertEqual(tuple(neck(inputs).shape), (1, 128, 32, 88))
            head = OccHead().eval()
            x = torch.randn(1, 512, 3, 4)
            expected = head.predicter(torch.relu(head.final_conv.conv(x)).permute(0, 3, 2, 1)).reshape(1, 4, 3, 16, 18)
            torch.testing.assert_close(head(x), expected)

    def test_occupancy_top_bev_free_and_axis_orientation(self):
        occupancy = np.full((200, 200, 16), 17, dtype=np.uint8)
        occupancy[5, 10, 2] = 11
        occupancy[5, 10, 8] = 4
        occupancy[100, 20, 3] = 0
        bev = top_bev(occupancy)
        self.assertEqual(bev.shape, (200, 200))
        self.assertEqual(bev[189, 5], 4)
        self.assertEqual(bev[179, 100], 0)
        self.assertEqual(bev[0, 0], 17)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_outputs(root, occupancy, {}, {})
            with np.load(root / 'occupancy.npz') as data:
                np.testing.assert_array_equal(data['semantics'], occupancy)
                self.assertEqual(len(data['class_names']), 18)
            with Image.open(root / 'bev_semantics.png') as image:
                self.assertEqual(image.size, (200, 200))
                self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))


if __name__ == '__main__':
    unittest.main()

"""Small-grid checks for the extension-free EFFOcc sparse encoder.

Run with the designated environment:
    /home/administrator/miniconda3/envs/vla-supervised/bin/python \
        -m unittest discover -s tests -p test_effocc_sparse.py -v
"""

import itertools
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tasks.effocc.python.effocc_sparse import (  # noqa: E402
    SparseBasicBlockTorch, SparseConvTorch, SparseEncoderTorch,
    SparseTensorTorch, hard_voxelize, sparse_output_shape,
)


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
_PRECISION_BEFORE_TESTS = None


def setUpModule():
    global _PRECISION_BEFORE_TESTS
    # PyTorch 2.14 enables cuDNN TF32 by default. The independent dense
    # reference must use the same full float32 precision as the sparse GEMMs.
    _PRECISION_BEFORE_TESTS = (torch.backends.cuda.matmul.fp32_precision,
                               torch.backends.cudnn.conv.fp32_precision)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"


def tearDownModule():
    matmul, convolution = _PRECISION_BEFORE_TESTS
    torch.backends.cuda.matmul.fp32_precision = matmul
    torch.backends.cudnn.conv.fp32_precision = convolution


def gather_dense(dense, coordinates):
    return dense[coordinates[:, 0], :, coordinates[:, 1],
                 coordinates[:, 2], coordinates[:, 3]]


def expected_regular_coordinates(coordinates, shape, kernel, stride, padding):
    output = sparse_output_shape(shape, kernel, stride, padding)
    result = set()
    for b, *spatial in coordinates.cpu().tolist():
        for offset in itertools.product(*(range(k) for k in kernel)):
            numerator = [a + p - k for a, p, k in zip(spatial, padding, offset)]
            if any(a % s for a, s in zip(numerator, stride)):
                continue
            site = tuple(a // s for a, s in zip(numerator, stride))
            if all(0 <= a < n for a, n in zip(site, output)):
                result.add((b, *site))
    return result


def coordinate_set(coordinates):
    return set(map(tuple, coordinates.cpu().tolist()))


class SparseConvolutionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(312)

    def _input(self, device):
        # Unsorted, two batches, boundary sites, and a zero-valued active site.
        coordinates = torch.tensor([
            [1, 4, 5, 6], [0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 2, 3],
            [1, 0, 1, 0], [0, 3, 4, 5], [0, 0, 1, 0], [1, 2, 3, 4],
            [0, 4, 5, 6], [1, 1, 0, 6], [0, 1, 2, 4], [0, 2, 3, 3],
        ], device=device)
        features = torch.randn(len(coordinates), 2, device=device)
        features[0].zero_()
        return SparseTensorTorch(features, coordinates, (5, 6, 7), batch_size=2)

    def test_krsc_is_cross_correlation_without_kernel_flip(self):
        for device in DEVICES:
            with self.subTest(device=device), torch.inference_mode():
                x = SparseTensorTorch(torch.tensor([[2.0]], device=device),
                                      torch.tensor([[0, 2, 2, 2]], device=device),
                                      (5, 5, 5))
                conv = SparseConvTorch(1, 1, 3, padding=1).to(device)
                conv.weight.copy_(torch.arange(1, 28, device=device).reshape(1, 3, 3, 3, 1))
                out = conv(x)
                dense = F.conv3d(x.dense(), conv.weight.permute(0, 4, 1, 2, 3), padding=1)
                torch.testing.assert_close(out.dense(), dense)
                actual = dict(zip(map(tuple, out.coordinates.cpu().tolist()),
                                  out.features.flatten().cpu().tolist()))
                self.assertEqual(actual[(0, 1, 1, 1)], 54.0)
                self.assertEqual(actual[(0, 3, 3, 3)], 2.0)

    def test_submanifold_matches_dense_at_exact_input_activity(self):
        for device in DEVICES:
            with self.subTest(device=device), torch.inference_mode():
                x = self._input(device)
                conv = SparseConvTorch(2, 3, 3, padding=1, subm=True).to(device)
                out = conv(x)
                dense = F.conv3d(x.dense(), conv.weight.permute(0, 4, 1, 2, 3), padding=1)
                self.assertIs(out.coordinates, x.coordinates)
                self.assertEqual(len(out.features), len(x.features))
                torch.testing.assert_close(out.features, gather_dense(dense, x.coordinates),
                                           atol=3e-6, rtol=2e-5)
                maps = [value for key, value in x._cache.items()
                        if isinstance(key, tuple) and key[0] == "subm"]
                second = SparseConvTorch(3, 4, 3, padding=1, subm=True).to(device)
                second(out)
                self.assertEqual(len(maps), 1)
                self.assertTrue(any(value is maps[0] for value in out._cache.values()))

    def test_regular_stride_padding_and_activity_union(self):
        configurations = [
            ((3, 3, 3), (1, 1, 1), (1, 1, 1)),
            ((3, 3, 3), (2, 2, 2), (1, 1, 1)),
            ((3, 3, 3), (2, 2, 2), (0, 1, 1)),
            ((3, 1, 1), (2, 1, 1), (0, 0, 0)),
            ((2, 3, 2), (1, 2, 2), (0, 1, 0)),
        ]
        for device, (kernel, stride, padding) in itertools.product(DEVICES, configurations):
            with self.subTest(device=device, kernel=kernel, stride=stride, padding=padding):
                with torch.inference_mode():
                    x = self._input(device)
                    conv = SparseConvTorch(2, 3, kernel, stride, padding, bias=True).to(device)
                    conv.bias.fill_(0.37)
                    out = conv(x)
                    dense = F.conv3d(x.dense(), conv.weight.permute(0, 4, 1, 2, 3),
                                     conv.bias, stride=stride, padding=padding)
                    expected = expected_regular_coordinates(x.coordinates, x.spatial_shape,
                                                            kernel, stride, padding)
                    self.assertEqual(coordinate_set(out.coordinates), expected)
                    self.assertEqual(tuple(dense.shape[2:]), out.spatial_shape)
                    torch.testing.assert_close(out.features, gather_dense(dense, out.coordinates),
                                               atol=3e-6, rtol=2e-5)
                    active_mask = torch.zeros_like(dense[:, 0], dtype=torch.bool)
                    c = out.coordinates
                    active_mask[c[:, 0], c[:, 1], c[:, 2], c[:, 3]] = True
                    scattered = out.dense().permute(0, 2, 3, 4, 1)
                    self.assertTrue(torch.equal(scattered[~active_mask],
                                                torch.zeros_like(scattered[~active_mask])))

    def test_no_neighbors_bias_and_empty_input(self):
        for device in DEVICES:
            with self.subTest(device=device), torch.inference_mode():
                # A stride-2 1x1x1 convolution cannot touch an odd input site.
                x = SparseTensorTorch(torch.ones(1, 1, device=device),
                                      torch.tensor([[0, 1, 1, 1]], device=device), (5, 5, 5))
                conv = SparseConvTorch(1, 1, 1, stride=2, bias=True).to(device)
                conv.bias.fill_(4.0)
                out = conv(x)
                self.assertEqual(tuple(out.features.shape), (0, 1))
                self.assertEqual(torch.count_nonzero(out.dense()).item(), 0)
                empty = SparseTensorTorch(torch.empty(0, 2, device=device),
                                          torch.empty(0, 4, dtype=torch.int32, device=device),
                                          (5, 6, 7), batch_size=2)
                for subm in (False, True):
                    layer = SparseConvTorch(2, 3, 3, padding=1, subm=subm).to(device)
                    result = layer(empty)
                    self.assertEqual(tuple(result.features.shape), (0, 3))
                    self.assertEqual(torch.count_nonzero(result.dense()).item(), 0)

    def test_float32_inside_external_autocast(self):
        for device in DEVICES:
            with self.subTest(device=device), torch.inference_mode():
                x = self._input(device)
                conv = SparseConvTorch(2, 3, padding=1, subm=True).to(device)
                reference = conv(x).features
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    actual = conv(x).features
                self.assertEqual(actual.dtype, torch.float32)
                torch.testing.assert_close(actual, reference, atol=0, rtol=0)

    def test_batch_dense_scatter_layout(self):
        coordinates = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]])
        x = SparseTensorTorch(torch.tensor([[1., 2.], [3., 4.]]), coordinates,
                              (2, 2, 2), batch_size=2)
        dense = x.dense()
        self.assertEqual(tuple(dense.shape), (2, 2, 2, 2, 2))
        torch.testing.assert_close(gather_dense(dense, coordinates), x.features)
        self.assertEqual(dense.sum().item(), 10.0)

    def test_subm_arguments_reject_noncentered_kernel(self):
        with self.assertRaises(ValueError):
            SparseConvTorch(2, 2, 3, padding=0, subm=True)


class EncoderTests(unittest.TestCase):
    def test_eval_residual_block_bn_and_activity_mask(self):
        for device in DEVICES:
            with self.subTest(device=device), torch.inference_mode():
                torch.manual_seed(719)
                block = SparseBasicBlockTorch(2).eval().to(device)
                for bn in (block.bn1, block.bn2):
                    bn.weight.copy_(torch.tensor([1.3, 0.7], device=device))
                    bn.bias.copy_(torch.tensor([0.2, -0.4], device=device))
                    bn.running_mean.copy_(torch.tensor([0.5, -0.7], device=device))
                    bn.running_var.copy_(torch.tensor([0.6, 1.8], device=device))
                coordinates = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 2], [0, 2, 2, 2]],
                                           device=device)
                x = SparseTensorTorch(torch.randn(3, 2, device=device), coordinates, (4, 4, 4))
                actual = block(x)
                original = x.dense()
                first = F.conv3d(original, block.conv1.weight.permute(0, 4, 1, 2, 3), padding=1)
                first_features = torch.relu(block.bn1(gather_dense(first, coordinates)))
                masked = SparseTensorTorch(first_features, coordinates, x.spatial_shape).dense()
                second = F.conv3d(masked, block.conv2.weight.permute(0, 4, 1, 2, 3), padding=1)
                expected = torch.relu(block.bn2(gather_dense(second, coordinates)) + x.features)
                torch.testing.assert_close(actual.features, expected, atol=3e-6, rtol=2e-5)
                self.assertIs(actual.coordinates, x.coordinates)
                self.assertEqual(block.bn1.eps, 1e-3)
                self.assertEqual(block.bn1.momentum, 0.01)
                self.assertIn("bn1.running_var", block.state_dict())
                self.assertNotIn("norm1.running_var", block.state_dict())

    def test_checkpoint_key_hierarchy_and_krsc_shapes(self):
        model = SparseEncoderTorch().eval()
        state = model.state_dict()
        expected = {}
        def add_conv(name, in_channels, out_channels, kernel=(3, 3, 3)):
            expected[name + ".weight"] = (out_channels, *kernel, in_channels)
        def add_bn(name, channels):
            for key in ("weight", "bias", "running_mean", "running_var"):
                expected[name + "." + key] = (channels,)
            expected[name + ".num_batches_tracked"] = ()
        add_conv("conv_input.0", 5, 16)
        add_bn("conv_input.1", 16)
        for stage, channels in enumerate((16, 32, 64, 128), 1):
            for block in (0, 1):
                prefix = f"encoder_layers.encoder_layer{stage}.{block}"
                for number in (1, 2):
                    add_conv(f"{prefix}.conv{number}", channels, channels)
                    add_bn(f"{prefix}.bn{number}", channels)
            if stage < 4:
                prefix = f"encoder_layers.encoder_layer{stage}.2"
                add_conv(prefix + ".0", channels, channels * 2)
                add_bn(prefix + ".1", channels * 2)
        add_conv("conv_out.0", 128, 128, (3, 1, 1))
        add_bn("conv_out.1", 128)
        self.assertEqual({name: tuple(value.shape) for name, value in state.items()}, expected)
        # Simulate exactly the prefix removal used by the integrating caller.
        checkpoint = {"pts_middle_encoder." + name: value for name, value in state.items()}
        loaded = {name.removeprefix("pts_middle_encoder."): value for name, value in checkpoint.items()}
        result = model.load_state_dict(loaded, strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])

    def test_output_shape_formula_for_default_grid(self):
        shape = (41, 1600, 1600)
        for padding, expected in ((1, (21, 800, 800)), (1, (11, 400, 400)),
                                  ((0, 1, 1), (5, 200, 200))):
            shape = sparse_output_shape(shape, 3, 2, padding)
            self.assertEqual(shape, expected)
        self.assertEqual(sparse_output_shape(shape, (3, 1, 1), (2, 1, 1), 0), (2, 200, 200))

    def test_small_encoder_forward_and_empty_forward(self):
        for device in DEVICES:
            with self.subTest(device=device), torch.inference_mode():
                model = SparseEncoderTorch(sparse_shape=(41, 16, 16)).eval().to(device)
                coordinates = torch.tensor([[0, 1, 1, 1], [0, 20, 8, 8],
                                            [0, 39, 15, 15], [0, 19, 8, 9]],
                                           dtype=torch.int32, device=device)
                features = torch.randn(4, 5, device=device)
                actual = model(features, coordinates)
                self.assertEqual(tuple(actual.shape), (1, 256, 2, 2))
                self.assertEqual(actual.dtype, torch.float32)
                self.assertTrue(torch.isfinite(actual).all().item())
                empty = model(features[:0], coordinates[:0])
                self.assertEqual(tuple(empty.shape), (1, 256, 2, 2))
                self.assertEqual(torch.count_nonzero(empty).item(), 0)
                # Distinct forwards must not reuse old neighbor maps.
                repeated = model(features, coordinates.flip(0))
                fresh = SparseEncoderTorch((41, 16, 16)).eval().to(device)
                fresh.load_state_dict(model.state_dict())
                torch.testing.assert_close(repeated, fresh(features, coordinates.flip(0)),
                                           atol=0, rtol=0)


class VoxelizationTests(unittest.TestCase):
    def test_point_and_voxel_caps_preserve_original_order(self):
        # Voxel x=2 appears first, x=0 second, x=1 third and is discarded.
        # Later points in already retained voxels still contribute after cap.
        points = np.array([
            [2.1, .1, .1, 1, 2], [.1, .1, .1, 10, 20],
            [1.1, .1, .1, 99, 99], [2.2, .1, .1, 3, 4],
            [2.3, .1, .1, 999, 999], [.2, .1, .1, 30, 40],
            [3.0, .1, .1, 999, 999], [-.1, .1, .1, 999, 999],
        ], dtype=np.float32)
        original = points.copy()
        features, coordinates = hard_voxelize(points, (0, 0, 0, 3, 1, 1), (1, 1, 1),
                                               max_points=2, max_voxels=2)
        np.testing.assert_array_equal(coordinates, [[0, 0, 0, 2], [0, 0, 0, 0]])
        np.testing.assert_allclose(features, np.stack((points[[0, 3]].mean(0),
                                                      points[[1, 5]].mean(0))))
        self.assertEqual(features.dtype, np.float32)
        self.assertEqual(coordinates.dtype, np.int32)
        np.testing.assert_array_equal(points, original)

    def test_default_first_ten_points_and_batchzyx(self):
        points = np.tile(np.array([[-39.99, -39.94, -.65, 0, 0]], dtype=np.float32), (12, 1))
        points[:, 3] = np.arange(12)
        points[:, 4] = np.arange(12) * 2
        features, coordinates = hard_voxelize(points)
        np.testing.assert_allclose(features, points[:10].mean(0, keepdims=True), rtol=1e-6)
        np.testing.assert_array_equal(coordinates, [[0, 2, 1, 0]])

    def test_bounds_empty_and_nonfinite_xyz(self):
        points = np.array([
            [0, 0, 0, 1, 2], [1, 0, 0, 3, 4], [0, 1, 0, 3, 4],
            [0, 0, 1, 3, 4], [-.0001, 0, 0, 3, 4],
            [np.nan, 0, 0, 3, 4], [np.inf, 0, 0, 3, 4],
        ], dtype=np.float32)
        args = ((0, 0, 0, 1, 1, 1), (.5, .5, .5))
        features, coordinates = hard_voxelize(points, *args)
        np.testing.assert_array_equal(features, points[:1])
        np.testing.assert_array_equal(coordinates, [[0, 0, 0, 0]])
        for data, cap in ((points[:0], 120000), (points[1:], 120000), (points, 0)):
            features, coordinates = hard_voxelize(data, *args, max_voxels=cap)
            self.assertEqual(features.shape, (0, 5))
            self.assertEqual(coordinates.shape, (0, 4))
            self.assertEqual(features.dtype, np.float32)
            self.assertEqual(coordinates.dtype, np.int32)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            hard_voxelize(np.zeros((2, 4), dtype=np.float32))
        with self.assertRaises(ValueError):
            hard_voxelize(np.zeros((2, 5), dtype=np.float32), max_points=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

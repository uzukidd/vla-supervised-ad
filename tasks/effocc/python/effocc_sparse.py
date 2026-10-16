# Inference adaptation of EFFOcc/OpenMMLab SparseEncoder (Apache-2.0).
# Original architecture: Copyright (c) OpenMMLab. All rights reserved.
"""EFFOcc SparseEncoder evaluation using only NumPy and ordinary PyTorch.

Weights retain spconv 2 KRSC layout [out, kz, ky, kx, in]. Coordinates are
[batch, z, y, x]. No dense input grid or compiled sparse extension is needed.
Only the final, downsampled feature map is materialized as a dense tensor.

The module hierarchy follows EFFOcc's sparse_encoder.py / sparse_block.py:
remove ``pts_middle_encoder.`` from checkpoint keys before strict loading.
Call ``SparseEncoderTorch().eval()`` and use ``torch.inference_mode()`` for
inference. Input features and convolution accumulation always use float32,
including when the caller has enabled autocast.
"""

from __future__ import annotations

import itertools
import math
from typing import Sequence

import numpy as np
import torch
from torch import nn


def _triple(value: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    result = tuple(int(v) for v in value)
    if len(result) != 3:
        raise ValueError("Expected three spatial dimensions (z, y, x)")
    return result


def sparse_output_shape(spatial_shape, kernel_size, stride=1, padding=0):
    """spconv's dilation=1 regular-convolution output formula, in zyx order."""
    kernel_size, stride, padding = map(_triple, (kernel_size, stride, padding))
    result = tuple((n + 2 * p - k) // s + 1
                   for n, k, s, p in zip(spatial_shape, kernel_size, stride, padding))
    if any(n <= 0 for n in result):
        raise ValueError(f"Convolution has nonpositive output shape {result}")
    return result


def _linear_keys(coordinates: torch.Tensor, spatial_shape) -> torch.Tensor:
    z, y, x = spatial_shape
    bzy = (coordinates[:, 0] * z + coordinates[:, 1]) * y + coordinates[:, 2]
    return bzy * x + coordinates[:, 3]


def _coordinates_from_keys(keys: torch.Tensor, spatial_shape) -> torch.Tensor:
    z, y, x = spatial_shape
    xx = keys.remainder(x)
    q = torch.div(keys, x, rounding_mode="floor")
    yy = q.remainder(y)
    q = torch.div(q, y, rounding_mode="floor")
    zz = q.remainder(z)
    bb = torch.div(q, z, rounding_mode="floor")
    return torch.stack((bb, zz, yy, xx), dim=1)


class SparseTensorTorch:
    """Features plus unique activity coordinates, with a per-activity cache.

    ``replace_feature`` shares coordinates and cache. A regular convolution
    starts a new cache, so maps cannot leak across different inputs/forwards.
    Mutating coordinates in place after construction is not supported.
    """

    def __init__(self, features, coordinates, spatial_shape, batch_size=1):
        if features.ndim != 2 or coordinates.ndim != 2 or coordinates.shape[1] != 4:
            raise ValueError("Expected features [N,C] and coordinates [N,4]")
        if len(features) != len(coordinates):
            raise ValueError("Feature and coordinate counts differ")
        self.features = features.float()
        self.coordinates = coordinates.to(device=features.device, dtype=torch.int64)
        self.spatial_shape = _triple(spatial_shape)
        self.batch_size = int(batch_size)
        if self.batch_size < 1 or min(self.spatial_shape) < 1:
            raise ValueError("Batch size and spatial dimensions must be positive")
        self._cache = {}

    @property
    def indices(self):
        return self.coordinates

    def replace_feature(self, features):
        if len(features) != len(self.coordinates):
            raise ValueError("Replacement features must preserve the activity set")
        result = object.__new__(SparseTensorTorch)
        result.features = features
        result.coordinates = self.coordinates
        result.spatial_shape = self.spatial_shape
        result.batch_size = self.batch_size
        result._cache = self._cache
        return result

    def _sorted_index(self):
        if "sorted_index" not in self._cache:
            keys = _linear_keys(self.coordinates, self.spatial_shape)
            self._cache["sorted_index"] = torch.sort(keys)
        return self._cache["sorted_index"]

    def dense(self):
        """Scatter activity into [B,C,Z,Y,X]; used only after downsampling."""
        channels = self.features.shape[1]
        sites = self.batch_size * math.prod(self.spatial_shape)
        # Channels first gives the same C-major, then Z-major reshape as spconv.
        flat = self.features.new_zeros((channels, sites))
        keys = _linear_keys(self.coordinates, self.spatial_shape)
        flat[:, keys] = self.features.T
        return flat.reshape(channels, self.batch_size, *self.spatial_shape).permute(
            1, 0, 2, 3, 4).contiguous()


def _regular_coordinates(x, kernel_size, stride, padding, output_shape):
    """Union of legal output sites touched by at least one input activity."""
    parts = []
    c = x.coordinates
    # Each temporary is O(N), never [N, kernel_volume, 4].
    for offset in itertools.product(*(range(k) for k in kernel_size)):
        valid = torch.ones(len(c), dtype=torch.bool, device=c.device)
        axes = []
        for axis in range(3):
            numerator = c[:, axis + 1] + padding[axis] - offset[axis]
            out_axis = torch.div(numerator, stride[axis], rounding_mode="floor")
            valid &= numerator.remainder(stride[axis]) == 0
            valid &= (out_axis >= 0) & (out_axis < output_shape[axis])
            axes.append(out_axis)
        key = ((c[:, 0] * output_shape[0] + axes[0]) * output_shape[1]
               + axes[1]) * output_shape[2] + axes[2]
        parts.append(key[valid])
    keys = torch.unique(torch.cat(parts), sorted=True)
    return _coordinates_from_keys(keys, output_shape)


def _neighbor_map(x, output_coordinates, kernel_size, stride, padding):
    """One pair of compact input/output row lists per KRSC kernel offset."""
    sorted_keys, sorted_rows = x._sorted_index()
    c = output_coordinates
    maps = []
    for offset in itertools.product(*(range(k) for k in kernel_size)):
        if len(sorted_keys) == 0 or len(c) == 0:
            empty = c.new_empty((0,))
            maps.append((empty, empty))
            continue
        valid = torch.ones(len(c), dtype=torch.bool, device=c.device)
        axes = []
        for axis in range(3):
            in_axis = c[:, axis + 1] * stride[axis] - padding[axis] + offset[axis]
            valid &= (in_axis >= 0) & (in_axis < x.spatial_shape[axis])
            axes.append(in_axis)
        keys = ((c[:, 0] * x.spatial_shape[0] + axes[0]) * x.spatial_shape[1]
                + axes[1]) * x.spatial_shape[2] + axes[2]
        positions = torch.searchsorted(sorted_keys, keys)
        # Clamp before indexing: searchsorted returns N for keys above the end.
        safe = positions.clamp(max=len(sorted_keys) - 1)
        valid &= positions < len(sorted_keys)
        valid &= sorted_keys[safe] == keys
        output_rows = torch.nonzero(valid, as_tuple=False).flatten()
        input_rows = sorted_rows[safe[output_rows]]
        maps.append((input_rows, output_rows))
    return maps


class SparseConvTorch(nn.Module):
    """3D sparse cross-correlation with KRSC weights and no dense input grid.

    ``forward(SparseTensorTorch) -> SparseTensorTorch``. Submanifold mode
    preserves the exact input row order and activity coordinates. Regular mode
    emits the sorted union of sites with an input neighbor, even for zero-valued
    input features/weights. Bias applies only to these active output sites.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, subm=False, bias=False):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.subm = bool(subm)
        if min(self.kernel_size) < 1 or min(self.stride) < 1 or min(self.padding) < 0:
            raise ValueError("Invalid kernel, stride or padding")
        if self.subm and (self.stride != (1, 1, 1)
                          or self.padding != tuple(k // 2 for k in self.kernel_size)
                          or any(k % 2 == 0 for k in self.kernel_size)):
            raise ValueError("Submanifold mode requires odd kernels, stride 1, centered padding")
        self.weight = nn.Parameter(torch.empty(out_channels, *self.kernel_size, in_channels))
        nn.init.normal_(self.weight, std=math.sqrt(2 / (in_channels * math.prod(self.kernel_size))))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: SparseTensorTorch) -> SparseTensorTorch:
        with torch.autocast(device_type=x.features.device.type, enabled=False):
            return self._forward_float32(x)

    def _forward_float32(self, x):
        if x.features.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels")
        if self.weight.device != x.features.device:
            raise ValueError("Convolution parameters and features must use the same device")
        if self.subm:
            coordinates = x.coordinates
            shape = x.spatial_shape
            cache_key = ("subm", shape, self.kernel_size, self.stride, self.padding)
            if cache_key not in x._cache:
                x._cache[cache_key] = _neighbor_map(
                    x, coordinates, self.kernel_size, self.stride, self.padding)
            neighbors = x._cache[cache_key]
        else:
            shape = sparse_output_shape(x.spatial_shape, self.kernel_size,
                                        self.stride, self.padding)
            coordinates = _regular_coordinates(x, self.kernel_size, self.stride,
                                               self.padding, shape)
            neighbors = _neighbor_map(x, coordinates, self.kernel_size,
                                      self.stride, self.padding)
        features = x.features.float()
        result = features.new_zeros((len(coordinates), self.out_channels))
        offsets = itertools.product(*(range(k) for k in self.kernel_size))
        for offset, (input_rows, output_rows) in zip(offsets, neighbors):
            if input_rows.numel() == 0:
                continue
            kernel = self.weight[(slice(None), *offset, slice(None))].float()
            contribution = features[input_rows] @ kernel.T
            # Output rows are unique for a fixed offset; this is race free.
            result[output_rows] = result[output_rows] + contribution
        if self.bias is not None:
            result = result + self.bias.float()
        if self.subm:
            return x.replace_feature(result)
        return SparseTensorTorch(result, coordinates, shape, x.batch_size)


class SparseSequentialTorch(nn.Sequential):
    def forward(self, x):
        for layer in self:
            if isinstance(layer, (nn.BatchNorm1d, nn.ReLU)):
                x = x.replace_feature(layer(x.features))
            else:
                x = layer(x)
        return x


def _batch_norm(channels):
    return nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)


def _conv_module(in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, subm=False):
    return SparseSequentialTorch(
        SparseConvTorch(in_channels, out_channels, kernel_size, stride,
                        padding, subm=subm, bias=False),
        _batch_norm(out_channels), nn.ReLU(inplace=True))


class SparseBasicBlockTorch(nn.Module):
    """EFFOcc SparseBasicBlock with checkpoint-compatible bn1/bn2 names."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = SparseConvTorch(channels, channels, 3, padding=1, subm=True)
        self.bn1 = _batch_norm(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SparseConvTorch(channels, channels, 3, padding=1, subm=True)
        self.bn2 = _batch_norm(channels)

    def forward(self, x):
        identity = x.features
        out = self.conv1(x)
        out = out.replace_feature(self.relu(self.bn1(out.features)))
        out = self.conv2(out)
        return out.replace_feature(self.relu(self.bn2(out.features) + identity))


class SparseEncoderTorch(nn.Module):
    """Checkpoint-compatible EFFOcc lidar SparseEncoder evaluation forward.

    Default input: features [N,5], coors [N,4] batchzyx. Default output:
    [batch_size,256,200,200], float32. Only final [B,128,2,200,200] is dense.
    Smaller spatial shapes may be used for tests if all convolutions fit.
    """

    def __init__(self, sparse_shape=(41, 1600, 1600)):
        super().__init__()
        self.sparse_shape = _triple(sparse_shape)
        self.conv_input = _conv_module(5, 16, padding=1, subm=True)
        self.encoder_layers = SparseSequentialTorch()
        channels = ((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128))
        paddings = ((0, 0, 1), (0, 0, 1), (0, 0, (0, 1, 1)), (0, 0))
        in_channels = 16
        for stage, stage_channels in enumerate(channels):
            blocks = []
            for block, out_channels in enumerate(stage_channels):
                if stage < 3 and block == len(stage_channels) - 1:
                    blocks.append(_conv_module(in_channels, out_channels, stride=2,
                                               padding=paddings[stage][block]))
                else:
                    blocks.append(SparseBasicBlockTorch(out_channels))
                in_channels = out_channels
            self.encoder_layers.add_module(f"encoder_layer{stage + 1}",
                                           SparseSequentialTorch(*blocks))
        self.conv_out = _conv_module(128, 128, kernel_size=(3, 1, 1),
                                     stride=(2, 1, 1), padding=0)

    def forward(self, features, coors, batch_size=1):
        if features.ndim != 2 or features.shape[1] != 5:
            raise ValueError("SparseEncoderTorch expects features [N,5]")
        with torch.autocast(device_type=features.device.type, enabled=False):
            x = SparseTensorTorch(features.float(), coors, self.sparse_shape, batch_size)
            x = self.conv_input(x)
            x = self.encoder_layers(x)
            x = self.conv_out(x)
            dense = x.dense()
            b, c, d, h, w = dense.shape
            return dense.reshape(b, c * d, h, w)


def hard_voxelize(points, point_cloud_range=(-40, -40, -1, 40, 40, 5.4),
                  voxel_size=(0.05, 0.05, 0.16), max_points=10,
                  max_voxels=120000):
    """Deterministic hard voxelization plus HardSimpleVFE mean, on NumPy.

    Keep the first ``max_voxels`` distinct voxels in point order and the first
    ``max_points`` points per retained voxel. Bounds are [lower, upper).
    Return float32 [V,5] means and int32 [V,4] batchzyx coordinates (batch=0),
    in voxel first-appearance order. No input arrays are mutated.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 5:
        raise ValueError("Expected points [N,5]")
    if max_points < 1 or max_voxels < 0:
        raise ValueError("max_points must be positive and max_voxels nonnegative")
    bounds = np.asarray(point_cloud_range, dtype=np.float32)
    size = np.asarray(voxel_size, dtype=np.float32)
    if bounds.shape != (6,) or size.shape != (3,) or np.any(size <= 0):
        raise ValueError("Expected six range bounds and three positive voxel sizes")
    if np.any(bounds[3:] <= bounds[:3]):
        raise ValueError("Upper range bounds must exceed lower bounds")
    empty = (np.empty((0, 5), dtype=np.float32), np.empty((0, 4), dtype=np.int32))
    if len(points) == 0 or max_voxels == 0:
        return empty
    valid = np.isfinite(points[:, :3]).all(axis=1)
    valid &= ((points[:, :3] >= bounds[:3]) & (points[:, :3] < bounds[3:])).all(axis=1)
    points = points[valid]
    if len(points) == 0:
        return empty
    xyz = np.floor((points[:, :3] - bounds[:3]) / size).astype(np.int64)
    grid = np.rint((bounds[3:] - bounds[:3]) / size).astype(np.int64)
    valid = ((xyz >= 0) & (xyz < grid)).all(axis=1)
    xyz, points = xyz[valid], points[valid]
    if len(points) == 0:
        return empty
    keys = (xyz[:, 2] * grid[1] + xyz[:, 1]) * grid[0] + xyz[:, 0]
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    chosen = np.argsort(first, kind="stable")[:max_voxels]
    remap = np.full(len(first), -1, dtype=np.int64)
    remap[chosen] = np.arange(len(chosen))
    order = np.argsort(inverse, kind="stable")
    groups = inverse[order]
    starts = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1]
    ranks = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    keep = (ranks < max_points) & (remap[groups] >= 0)
    rows, group_ids = order[keep], remap[groups[keep]]
    sums = np.zeros((len(chosen), 5), dtype=np.float32)
    np.add.at(sums, group_ids, points[rows])
    counts = np.bincount(group_ids, minlength=len(chosen)).astype(np.float32)
    features = sums / counts[:, None]
    coordinates = np.zeros((len(chosen), 4), dtype=np.int32)
    coordinates[:, 1:] = xyz[first[chosen]][:, ::-1]
    return features, coordinates


__all__ = ["SparseEncoderTorch", "SparseConvTorch", "SparseTensorTorch",
           "SparseBasicBlockTorch", "sparse_output_shape", "hard_voxelize"]

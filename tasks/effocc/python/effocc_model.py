# Inference adaptation of models/EFFOcc (Apache-2.0).
# Original OpenMMLab components: Copyright (c) OpenMMLab. All rights reserved.
# Architecture and checkpoint conventions: EFFOcc / BEVDet / FlashOcc.
"""Inference-only PyTorch adaptation of the local EFFOcc Occ3D Fusion-R18.

Parameter names follow the original modules for strict checkpoint loading.
BEVPoolv2's weighted sum uses chunked index_add instead of a compiled CUDA op.
The sparse encoder is implemented separately in effocc_sparse.py.
"""
from __future__ import annotations

import time

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

from tasks.effocc.python.effocc_sparse import SparseEncoderTorch, hard_voxelize


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, activation=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=True)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        return F.relu(x) if self.activation else x


class ImageBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        source = resnet18(weights=None)
        for name in ('conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4'):
            self.add_module(name, getattr(source, name))

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        a = self.layer2(x)
        b = self.layer3(a)
        c = self.layer4(b)
        return [a, b, c]


class ImageFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lateral_convs = nn.ModuleList([ConvModule(c, 128, 1) for c in (128, 256, 512)])
        self.fpn_convs = nn.ModuleList([ConvModule(128, 128, 3, padding=1)])

    def forward(self, inputs):
        laterals = [module(x) for module, x in zip(self.lateral_convs, inputs)]
        for i in (2, 1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=laterals[i - 1].shape[2:], mode='nearest')
        return self.fpn_convs[0](laterals[0])


def weighted_bev_sum(coordinates, depth, features, grid_lower=(-40., -40., -1.),
                     grid_interval=(.4, .4, 6.4), grid_shape=(200, 200, 1), chunk_size=100000):
    """Match BEVPoolv2's B=1 weighted splat; preserve its truncation-to-long rule."""
    # Coordinates N,D,H,W,3; depth N,D,H,W; features N,C,H,W.
    device = features.device
    lower = torch.tensor(grid_lower, dtype=coordinates.dtype, device=device)
    interval = torch.tensor(grid_interval, dtype=coordinates.dtype, device=device)
    integer = ((coordinates - lower) / interval).long().reshape(-1, 3)
    nx, ny, nz = grid_shape
    valid = ((integer >= 0) & (integer < torch.tensor(grid_shape, device=device))).all(1)
    n, d, h, w = depth.shape
    feature_index = torch.arange(n * h * w, device=device).reshape(n, 1, h, w).expand(n, d, h, w).reshape(-1)
    ids = torch.nonzero(valid, as_tuple=False).flatten()
    feature_flat = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
    depth_flat = depth.reshape(-1)
    output = features.new_zeros((nz * ny * nx, features.shape[1]))
    for batch in ids.split(chunk_size):
        xyz = integer[batch]
        destination = (xyz[:, 2] * ny + xyz[:, 1]) * nx + xyz[:, 0]
        contribution = feature_flat[feature_index[batch]] * depth_flat[batch, None]
        output.index_add_(0, destination, contribution)
    return output.reshape(nz, ny, nx, features.shape[1]).permute(0, 3, 1, 2).reshape(1, nz * features.shape[1], ny, nx)


class ViewTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lidar_input_net = nn.Sequential(
            nn.Conv2d(1, 8, 1), nn.BatchNorm2d(8), nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.BatchNorm2d(64), nn.ReLU(True))
        self.depth_net = nn.Sequential(
            nn.Conv2d(192, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.Conv2d(128, 152, 1))
        depths = torch.arange(1., 45., .5).reshape(-1, 1, 1).expand(-1, 32, 88)
        xs = torch.linspace(0, 703, 88).reshape(1, 1, 88).expand(88, 32, 88)
        ys = torch.linspace(0, 255, 32).reshape(1, 32, 1).expand(88, 32, 88)
        self.register_buffer('frustum', torch.stack((xs, ys, depths), -1), persistent=False)

    def coordinates(self, camera2keyego, intrinsics, post_rots, post_trans):
        points = self.frustum[None] - post_trans[:, None, None, None, :]
        points = torch.linalg.inv(post_rots)[:, None, None, None] @ points[..., None]
        points = torch.cat((points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), -2)
        combine = camera2keyego[:, :3, :3] @ torch.linalg.inv(intrinsics)
        points = (combine[:, None, None, None] @ points).squeeze(-1)
        return points + camera2keyego[:, None, None, None, :3, 3]

    def forward(self, features, depth_maps, camera2keyego, intrinsics, post_rots, post_trans):
        depth_features = self.lidar_input_net(depth_maps[:, None])
        output = self.depth_net(torch.cat((features, depth_features), 1))
        depth = output[:, :88].softmax(1)
        features = output[:, 88:]
        coordinates = self.coordinates(camera2keyego, intrinsics, post_rots, post_trans)
        return weighted_bev_sum(coordinates, depth, features)


class ConvFuser(nn.Module):
    def __init__(self):
        super().__init__()
        self.occ_enc = nn.Sequential(nn.Conv2d(320, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True))

    def forward(self, image, lidar):
        return self.occ_enc(torch.cat((image, lidar), 1))


class Second(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        for cin, cout, stride in ((256, 128, 1), (128, 256, 2)):
            layers = [nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                      nn.BatchNorm2d(cout, eps=1e-3, momentum=.01), nn.ReLU(True)]
            for _ in range(5):
                layers.extend([nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                               nn.BatchNorm2d(cout, eps=1e-3, momentum=.01), nn.ReLU(True)])
            blocks.append(nn.Sequential(*layers))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        outputs = []
        for module in self.blocks:
            x = module(x)
            outputs.append(x)
        return outputs


class SecondFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.deblocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(128, 256, 1, bias=False), nn.BatchNorm2d(256, eps=1e-3, momentum=.01), nn.ReLU(True)),
            nn.Sequential(nn.ConvTranspose2d(256, 256, 2, stride=2, bias=False), nn.BatchNorm2d(256, eps=1e-3, momentum=.01), nn.ReLU(True))])

    def forward(self, inputs):
        return torch.cat([module(x) for module, x in zip(self.deblocks, inputs)], 1)


class OccHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.final_conv = ConvModule(512, 256, 3, padding=1, activation=True)
        self.predicter = nn.Sequential(nn.Linear(256, 512), nn.Softplus(), nn.Linear(512, 16 * 18))

    def forward(self, features):
        features = self.final_conv(features).permute(0, 3, 2, 1)
        logits = self.predicter(features)
        return logits.reshape(logits.shape[0], logits.shape[1], logits.shape[2], 16, 18)


class EFFOccFusionR18(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_backbone = ImageBackbone()
        self.img_neck = ImageFPN()
        self.img_view_transformer = ViewTransformer()
        self.pts_middle_encoder = SparseEncoderTorch()
        self.occ_fuser = ConvFuser()
        self.pts_backbone = Second()
        self.pts_neck = SecondFPN()
        self.occ_head = OccHead()

    @torch.inference_mode()
    def forward(self, sample, device='cuda'):
        timings = {}
        def mark(name, started):
            if device == 'cuda':
                torch.cuda.synchronize()
            timings[name] = time.perf_counter() - started
            print(f'{name}: {timings[name]:.2f}s', flush=True)
        started = time.perf_counter()
        tensors = {key: torch.as_tensor(sample[key], dtype=torch.float32, device=device)
                   for key in ('images', 'depth_maps', 'camera2keyego', 'intrinsics', 'post_rots', 'post_trans')}
        # One image at a time limits peak VRAM and preserves eval-mode behavior.
        feature_list = [self.img_neck(self.img_backbone(image[None])) for image in tensors['images']]
        image_features = torch.cat(feature_list)
        image_bev = self.img_view_transformer(image_features, tensors['depth_maps'], tensors['camera2keyego'],
                                              tensors['intrinsics'], tensors['post_rots'], tensors['post_trans'])
        mark('image_branch_seconds', started)
        started = time.perf_counter()
        voxel_features, coordinates = hard_voxelize(sample['points'])
        print(f'Voxels: {len(voxel_features)}', flush=True)
        lidar_bev = self.pts_middle_encoder(torch.as_tensor(voxel_features, device=device),
                                            torch.as_tensor(coordinates, device=device), 1)
        mark('lidar_branch_seconds', started)
        if image_bev.shape != (1, 64, 200, 200) or lidar_bev.shape != (1, 256, 200, 200):
            raise RuntimeError(f'Unexpected BEV shapes: image={image_bev.shape}, lidar={lidar_bev.shape}')
        started = time.perf_counter()
        logits = self.occ_head(self.pts_neck(self.pts_backbone(self.occ_fuser(image_bev, lidar_bev))))
        if not torch.isfinite(logits).all():
            raise RuntimeError('Model produced nonfinite occupancy logits')
        prediction = logits.argmax(-1).squeeze(0).to(torch.uint8).cpu().numpy()
        mark('fusion_head_seconds', started)
        return prediction, {'timings': timings, 'voxel_count': len(voxel_features),
                            'image_bev_shape': list(image_bev.shape), 'lidar_bev_shape': list(lidar_bev.shape),
                            'logit_shape': list(logits.shape), 'all_logits_finite': True}

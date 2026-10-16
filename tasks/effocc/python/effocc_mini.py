#!/usr/bin/env python
"""Run the local EFFOcc Occ3D Fusion-R18 checkpoint on nuScenes mini, without legacy compiled ops."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

from shared.python.nuscenes_data import NuScenesMini
from shared.python.runtime import ROOT, ensure_environment, save_json
from shared.python.sam3 import make_palette

CLASS_NAMES = ('others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle',
               'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface', 'other_flat',
               'sidewalk', 'terrain', 'manmade', 'vegetation', 'free')
CAMERA_ORDER = ('CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT')


def top_bev(occupancy: np.ndarray) -> np.ndarray:
    """Select the highest non-free voxel in each XY column, without resampling."""
    if occupancy.shape != (200, 200, 16) or occupancy.dtype != np.uint8 or np.any(occupancy > 17):
        raise ValueError('Expected original uint8 occupancy of shape (200,200,16), class IDs 0..17')
    occupied = occupancy != 17
    heights = np.where(occupied, np.arange(16)[None, None, :], -1).max(2)
    xy = np.indices(heights.shape)
    labels = occupancy[xy[0], xy[1], np.maximum(heights, 0)].copy()
    labels[heights < 0] = 17
    # Raw occupancy axes are X,Y,Z. PNG columns increase X; rows decrease Y.
    return labels.T[::-1].copy()


def save_outputs(output: Path, prediction: np.ndarray, sample_info: dict, inference_info: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    bev = top_bev(prediction)
    np.savez_compressed(output / 'occupancy.npz', semantics=prediction, point_cloud_range=np.array([-40., -40., -1., 40., 40., 5.4]),
                        voxel_size=np.array([.4, .4, .4]), class_names=np.array(CLASS_NAMES))
    Image.fromarray(bev).save(output / 'bev_class_ids.png')
    palette = make_palette()
    colors = palette[bev].copy()
    colors[bev == 17] = 255
    colors[bev == 0] = (100, 100, 100)
    Image.fromarray(colors).save(output / 'bev_semantics.png')
    canvas = Image.new('RGB', (900, 1050), 'white')
    canvas.paste(Image.fromarray(colors).resize((800, 800), Image.Resampling.NEAREST), (50, 45))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), 'EFFOcc Fusion-R18 | original 80m x 80m grid, 0.4m voxels | top occupied voxel', fill='black')
    draw.text((20, 855), '+X right, +Y up | 200 x 200 BEV | occupancy 200 x 200 x 16 | no occupancy GT evaluation', fill='black')
    for i, name in enumerate(CLASS_NAMES):
        x, y = 20 + (i % 3) * 290, 885 + (i // 3) * 24
        color = (255, 255, 255) if i == 17 else ((100, 100, 100) if i == 0 else tuple(int(c) for c in palette[i]))
        draw.rectangle((x, y, x + 16, y + 16), fill=color, outline=(120, 120, 120))
        draw.text((x + 23, y + 2), f'{i}: {name}', fill='black')
    canvas.save(output / 'bev_preview.png')
    unique, counts = np.unique(prediction, return_counts=True)
    save_json(output / 'sample.json', {'input': sample_info, 'inference': inference_info,
              'voxel_class_counts': {CLASS_NAMES[int(i)]: int(count) for i, count in zip(unique, counts)},
              'occupancy_reference_camera': sample_info.get('camera_keyego_reference', 'CAM_FRONT_LEFT'),
              'output_coordinate_convention': 'Original EFFOcc output X,Y,Z, referenced to its image key ego input convention',
              'bev': 'Highest Z voxel with class !=17 (free), +X right,+Y up; no interpolation',
              'evaluation': None, 'reason': 'Local lidarseg labels are sparse point classes, not Occ3D dense occupancy GT.'})


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    result.add_argument('--dataset-dir', type=Path, default=ROOT / 'datasets/nuscenes')
    result.add_argument('--checkpoint', type=Path, default=ROOT / 'assets/checkpoints/effocc_fusion_r18.pth')
    result.add_argument('--split', choices=('all', 'mini_train', 'mini_val'), default='mini_val')
    result.add_argument('--scenes', nargs='+')
    result.add_argument('--start-index', type=int, default=0, help='Index in LiDAR samples, not camera images')
    result.add_argument('--max-samples', type=int, default=1, help='0 processes all remaining selected samples')
    result.add_argument('--sweeps', type=int, default=9, help='Previous LiDAR sweeps, in addition to the current keyframe')
    result.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    result.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/effocc_mini')
    result.add_argument('--no-camera-projection', action='store_true', help='Skip occupancy rendering into the six original camera images')
    result.add_argument('--projection-alpha', type=float, default=.45, help='Opacity for occupancy overlays on original camera images')
    result.add_argument('--inspect-data', action='store_true', help='Validate six-camera/sweep inputs without building the model')
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    ensure_environment()
    if args.start_index < 0 or args.max_samples < 0 or args.sweeps < 0:
        raise ValueError('start-index, max-samples and sweeps must be nonnegative')
    if not np.isfinite(args.projection_alpha) or not 0 <= args.projection_alpha <= 1:
        raise ValueError('projection-alpha must be in [0,1]')
    if args.sweeps > 9:
        raise ValueError('This adapter supports at most 9 preceding sweeps, matching the checkpoint configuration')
    from tasks.effocc.python.effocc_data import prepare_sample
    dataset = NuScenesMini(args.dataset_dir.resolve())
    views = dataset.select_views(list(CAMERA_ORDER), args.split, args.scenes)
    grouped = {}
    for view in views:
        grouped.setdefault(view['sample_token'], []).append(view)
    samples = list(grouped.values())
    if args.start_index >= len(samples):
        raise ValueError(f'start-index must be smaller than {len(samples)} selected samples')
    selected = samples[args.start_index:args.start_index + (args.max_samples or len(samples))]
    if args.inspect_data:
        sample = prepare_sample(dataset, selected[0], sweeps_num=args.sweeps)
        report = {'python': sys.executable, 'selected_samples': len(samples), 'requested_samples': len(selected),
                  'checkpoint_exists': args.checkpoint.is_file(), 'input': sample['info'],
                  'array_shapes': {key: list(value.shape) for key, value in sample.items() if isinstance(value, np.ndarray)}}
        print(json.dumps(report, indent=2))
        return 0
    output = args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Output directory is not empty: {output}; choose a new --output-dir')
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    import torch
    from tasks.effocc.python.effocc_model import EFFOccFusionR18
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; check WSL nvidia-smi and the vla-supervised torch installation')
    # Avoid precision shortcuts while verifying an inference-only operator adaptation.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state = checkpoint['state_dict']
    model = EFFOccFusionR18()
    model.load_state_dict(state, strict=True)
    model.eval().to(args.device)
    output.mkdir(parents=True, exist_ok=True)
    run = {**{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
           'status': 'running', 'processed_samples': 0, 'selected_samples': len(selected), 'python': sys.executable,
           'torch_version': torch.__version__, 'checkpoint_tensor_count': len(state),
           'checkpoint_loading': 'strict=True; no missing/unexpected tensors',
           'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
           'adapter_source_sha256': {name: hashlib.sha256((Path(__file__).resolve().parent / name).read_bytes()).hexdigest()
                                     for name in ('effocc_mini.py', 'effocc_model.py', 'effocc_sparse.py', 'effocc_data.py',
                                                  'effocc_cameras.py', 'effocc_projection.py')},
           'modalities': ['six_camera_images', 'lidar_points_and_sweeps'],
           'lidar_depth_input': True, 'image_branch': True, 'sparse_lidar_branch': True,
           'backend': 'inference-only PyTorch adaptation; no MMCV/spconv/custom CUDA extensions',
           'validation_limit': 'Operator unit tests and strict weight loading; no full original-backend output equivalence reference available.',
           'started_at': datetime.now(timezone.utc).isoformat()}
    save_json(output / 'run_config.json', run)
    save_json(output / 'classes.json', {i: name for i, name in enumerate(CLASS_NAMES)})
    started = time.perf_counter()
    try:
        for index, sample_views in enumerate(selected):
            info = sample_views[0]
            print(f"[{index + 1}/{len(selected)}] {info['scene_name']} {info['sample_token']}", flush=True)
            sample = prepare_sample(dataset, sample_views, sweeps_num=args.sweeps)
            prediction, details = model(sample, device=args.device)
            stem = f"{info['scene_name']}_{args.start_index + index:06d}_{info['sample_token']}"
            save_outputs(output / stem, prediction, sample['info'], details)
            if not args.no_camera_projection:
                from tasks.effocc.python.effocc_cameras import render_frame
                render_frame(dataset, output / stem, output / stem / 'camera_projections',
                             alpha=args.projection_alpha, device=args.device)
            run['processed_samples'] += 1
            save_json(output / 'run_config.json', run)
            print(f'Output: {output / stem}', flush=True)
        run['status'] = 'completed'
    except BaseException as error:
        run.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        raise
    finally:
        run.update(elapsed_seconds=time.perf_counter() - started, finished_at=datetime.now(timezone.utc).isoformat())
        save_json(output / 'run_config.json', run)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, FileNotFoundError, FileExistsError) as error:
        print(f'Error: {error}', file=sys.stderr)
        raise SystemExit(1)

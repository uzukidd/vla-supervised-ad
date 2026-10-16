#!/usr/bin/env python
"""Render saved EFFOcc occupancy into the six original nuScenes cameras."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from shared.python.nuscenes_data import NuScenesMini, _quaternion_matrix
from shared.python.runtime import ROOT, ensure_environment, save_json
from shared.python.sam3 import make_palette

CAMERAS = ('CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT')


def pose_matrix(record: dict) -> np.ndarray:
    """Match float32 pose/calibration inputs in EFFOcc.prepare_inputs."""
    result = np.eye(4, dtype=np.float32)
    result[:3, :3] = _quaternion_matrix(record['rotation'])
    result[:3, 3] = record['translation']
    return result.astype(np.float64)


def camera_geometry(dataset: NuScenesMini, sample_token: str, reference_camera: str) -> list[dict]:
    if reference_camera not in (*CAMERAS, 'LIDAR_TOP'):
        raise ValueError(f'Unknown occupancy ego reference: {reference_camera}')
    channels = dataset._keyframes.get(sample_token)
    if channels is None or reference_camera not in channels:
        raise ValueError(f'Sample {sample_token} is missing its reference sensor')
    key_record = channels[reference_camera]
    key_ego_to_global = pose_matrix(dataset._get('ego_pose', key_record['ego_pose_token']))
    global_to_key_ego = np.linalg.inv(key_ego_to_global)
    cameras = []
    for name in CAMERAS:
        record = channels.get(name)
        if record is None:
            raise ValueError(f'Sample {sample_token} is missing {name}')
        calibration = dataset._get('calibrated_sensor', record['calibrated_sensor_token'])
        ego = dataset._get('ego_pose', record['ego_pose_token'])
        camera_to_grid = global_to_key_ego @ pose_matrix(ego) @ pose_matrix(calibration)
        cameras.append({'camera': name, 'camera_token': record['token'], 'timestamp': record['timestamp'],
                        'image_path': dataset._media_path(record['filename'], name),
                        'intrinsic': np.asarray(calibration['camera_intrinsic'], dtype=np.float64),
                        'camera_to_grid': camera_to_grid})
    return cameras


def overlay_projection(image: Image.Image, labels: np.ndarray, palette: np.ndarray, alpha: float) -> Image.Image:
    if labels.shape != (image.height, image.width):
        raise ValueError('Projection labels must match original image dimensions')
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('alpha must be in [0,1]')
    pixels = np.asarray(image.convert('RGB')).astype(np.float32)
    visible = labels != 255
    pixels[visible] = pixels[visible] * (1 - alpha) + palette[labels[visible]] * alpha
    return Image.fromarray(np.rint(pixels).clip(0, 255).astype(np.uint8))


def projection_colors(labels: np.ndarray, palette: np.ndarray) -> Image.Image:
    result = palette[labels].copy()
    result[labels == 255] = (255, 255, 255)
    return Image.fromarray(result)


def contact_sheet(panels: list[tuple[str, Image.Image]], class_names: tuple[str, ...], palette: np.ndarray,
                  title: str, legend: bool = True) -> Image.Image:
    panel_width, panel_height, heading_height = 640, 360, 28
    footer_height = 170 if legend else 0
    canvas = Image.new('RGB', (3 * panel_width, 45 + 2 * (panel_height + heading_height) + footer_height), 'white')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 15)
    except OSError:
        font = ImageFont.load_default()
    draw.text((12, 10), title, fill='black', font=font)
    for i, (name, image) in enumerate(panels):
        x, y = (i % 3) * panel_width, 45 + (i // 3) * (panel_height + heading_height)
        draw.text((x + 10, y + 5), name, fill='black', font=font)
        resized = image.resize((panel_width, panel_height), Image.Resampling.BILINEAR)
        canvas.paste(resized, (x, y + heading_height))
    if legend:
        top = canvas.height - footer_height + 8
        # The free class is transparent and is never projected as a surface.
        for i, name in enumerate(class_names):
            if i == 17:
                continue
            x, y = 12 + (i % 3) * panel_width, top + (i // 3) * 24
            color = tuple(int(c) for c in palette[i])
            draw.rectangle((x, y, x + 15, y + 15), fill=color)
            draw.text((x + 22, y), f'{i}: {name}', fill='black', font=font)
        draw.text((12, canvas.height - 20), 'Free voxels are transparent; nearest occupied voxel per camera ray; no hit preserves the original image.',
                  fill='black', font=font)
    return canvas


def render_frame(dataset: NuScenesMini, frame_dir: Path, output_dir: Path, alpha: float = .45,
                 device: str = 'cuda', min_depth: float = .5, reference_camera: str | None = None) -> dict:
    from tasks.effocc.python.effocc_projection import raycast_occupancy

    frame_dir, output_dir = Path(frame_dir), Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f'Projection directory is not empty: {output_dir}; choose a new directory')
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('alpha must be in [0,1]')
    with np.load(frame_dir / 'occupancy.npz', allow_pickle=False) as archive:
        semantics = archive['semantics'].copy()
        bounds = archive['point_cloud_range'].copy()
        size = archive['voxel_size'].copy()
        class_names = tuple(str(name) for name in archive['class_names'])
    sample = json.loads((frame_dir / 'sample.json').read_text(encoding='utf-8'))
    info = sample['input']
    reference_camera = reference_camera or sample.get('occupancy_reference_camera', info.get('camera_keyego_reference', 'CAM_FRONT_LEFT'))
    cameras = camera_geometry(dataset, info['sample_token'], reference_camera)
    output_dir.mkdir(parents=True, exist_ok=True)
    palette = make_palette()
    palette[0] = (100, 100, 100)
    original_panels, overlay_panels, semantic_panels = [], [], []
    report = {'sample_token': info['sample_token'], 'reference_camera': reference_camera,
              'grid_frame': f'ego at {reference_camera} sample_data timestamp',
              'camera_transform': 'inverse(key_ego_to_global) @ camera_ego_to_global @ camera_sensor_to_ego',
              'projection': 'first occupied voxel along each original camera pixel-center ray',
              'free_class_id': 17, 'opaque_class_ids': list(range(17)), 'alpha': alpha,
              'min_camera_depth_m': min_depth, 'device': device, 'cameras': [],
              'limitations': 'Visualization of predicted occupancy, not image ground truth or a projection accuracy metric. '
                             'Uses the adapter image key-ego convention; its LiDAR branch uses a slightly different CAM_FRONT timestamp. '
                             'Camera poses account for timestamps, but moving-object motion is not compensated.'}
    save_json(output_dir / 'projection_config.json', {**report, 'status': 'running'})
    try:
        for camera in cameras:
            started = time.perf_counter()
            with Image.open(camera['image_path']) as source:
                image = source.convert('RGB')
            result = raycast_occupancy(semantics, camera['camera_to_grid'], camera['intrinsic'], image.size,
                                      bounds, size, min_depth=min_depth, device=device)
            labels, depth = result['labels'], result['depth']
            overlay = overlay_projection(image, labels, palette, alpha)
            semantic_image = projection_colors(labels, palette)
            name = camera['camera']
            image.save(output_dir / f'{name}_original.jpg', quality=94)
            overlay.save(output_dir / f'{name}_overlay.jpg', quality=94)
            semantic_image.save(output_dir / f'{name}_semantics.png')
            Image.fromarray(labels).save(output_dir / f'{name}_class_ids.png')
            np.savez_compressed(output_dir / f'{name}_projection.npz', labels=labels, depth=depth,
                                voxel_index=result['voxel_index'], camera_to_grid=camera['camera_to_grid'],
                                intrinsic=camera['intrinsic'])
            original_panels.append((name, image))
            overlay_panels.append((name, overlay))
            semantic_panels.append((name, semantic_image))
            hit = labels != 255
            details = {key: value for key, value in camera.items() if key not in ('camera_to_grid', 'intrinsic')}
            details['image_path'] = str(details['image_path'])
            details.update(image_size_wh=list(image.size), projected_pixels=int(hit.sum()),
                           projected_fraction=float(hit.mean()), seconds=time.perf_counter() - started,
                           raycaster=result['metadata'])
            report['cameras'].append(details)
            print(f"{name}: {hit.sum():,}/{hit.size:,} pixels, {details['seconds']:.1f}s", flush=True)
        contact_sheet(original_panels, class_names, palette, 'Original nuScenes cameras', legend=False).save(output_dir / 'six_cameras_original.jpg', quality=94)
        contact_sheet(overlay_panels, class_names, palette, 'EFFOcc occupancy projected onto original cameras | nearest occupied voxel', legend=True).save(output_dir / 'six_cameras_overlay.jpg', quality=94)
        contact_sheet(semantic_panels, class_names, palette, 'EFFOcc rendered semantic surfaces | white = no ray hit', legend=True).save(output_dir / 'six_cameras_semantics.jpg', quality=94)
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        raise
    finally:
        save_json(output_dir / 'projection_config.json', report)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    result.add_argument('--input-dir', type=Path, required=True, help='An EFFOcc run directory, or one sample directory containing occupancy.npz')
    result.add_argument('--output-dir', type=Path, help='New/empty output root; default <input-dir>/camera_projections')
    result.add_argument('--dataset-dir', type=Path, default=ROOT / 'datasets/nuscenes')
    result.add_argument('--max-samples', type=int, default=1, help='0 renders all saved samples')
    result.add_argument('--alpha', type=float, default=.45)
    result.add_argument('--min-depth', type=float, default=.5)
    result.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    result.add_argument('--reference-camera', choices=(*CAMERAS, 'LIDAR_TOP'), help='Override occupancy ego timestamp convention for diagnostic comparison')
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    ensure_environment()
    if args.max_samples < 0:
        raise ValueError('max-samples must be nonnegative')
    source = args.input_dir.resolve()
    frames = [source] if (source / 'occupancy.npz').is_file() else sorted(path.parent for path in source.glob('*/occupancy.npz'))
    if not frames:
        raise FileNotFoundError(f'No saved EFFOcc occupancy predictions found under {source}')
    frames = frames[:args.max_samples or len(frames)]
    output = (args.output_dir or source / 'camera_projections').resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Output directory is not empty: {output}')
    dataset = NuScenesMini(args.dataset_dir.resolve())
    for frame in frames:
        render_frame(dataset, frame, output / frame.name, args.alpha, args.device, args.min_depth, args.reference_camera)
    print(f'Six-camera projection results: {output}', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, FileNotFoundError, FileExistsError) as error:
        print(f'Error: {error}', file=sys.stderr)
        raise SystemExit(1)

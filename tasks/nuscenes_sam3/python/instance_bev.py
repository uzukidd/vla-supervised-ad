#!/usr/bin/env python
"""Experiment: six-camera SAM3 instance points covered by minimal convex BEV hulls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

from shared.python.nuscenes_data import CAMERAS, COARSE_CLASS_NAMES, NuScenesMini
from shared.python.runtime import ROOT, ensure_environment, save_json
from shared.python.sam3 import Sam3Segmenter
from tasks.nuscenes_sam3.python.sam3_nuscenes import load_categories
from tasks.nuscenes_sam3.python.instance_hulls import instance_rgb, save_hull_bev


class InstancePointCollector:
    """Keep original masks and confidence-exclusive ownership within one camera.

    Class/prompt aliases may return overlapping detections. Highest instance
    confidence wins; exact ties preserve earlier detection order. No cross-camera
    association is attempted. Packed masks retain even the pre-fusion detections.
    """
    def __init__(self, image_size: tuple[int, int], point_indices: np.ndarray, pixels: np.ndarray):
        width, height = image_size
        self.image_ids = np.zeros((height, width), dtype=np.uint16)
        self.image_scores = np.full((height, width), -np.inf, dtype=np.float32)
        self.point_indices = np.asarray(point_indices, dtype=np.int64)
        self.pixels = np.asarray(pixels, dtype=np.int64)
        if self.point_indices.ndim != 1 or self.pixels.shape != (len(self.point_indices), 2):
            raise ValueError('Projected indices and pixels must align')
        if len(np.unique(self.point_indices)) != len(self.point_indices):
            raise ValueError('Each projected point must occur once within a camera')
        if len(self.pixels) and (np.any(self.pixels < 0) or np.any(self.pixels[:, 0] >= width) or np.any(self.pixels[:, 1] >= height)):
            raise ValueError('Projection pixels must be inside the original image')
        self.point_ids = np.zeros(len(point_indices), dtype=np.uint16)
        self.point_scores = np.full(len(point_indices), -np.inf, dtype=np.float32)
        self.packed_masks: list[np.ndarray] = []
        self.records: list[dict] = []

    def __call__(self, mask: np.ndarray, metadata: dict) -> None:
        if mask.shape != self.image_ids.shape or mask.dtype != bool:
            raise ValueError('SAM3 instance mask must match the original image')
        instance_id = int(metadata['instance_id'])
        if instance_id != len(self.records) + 1 or instance_id > 65535:
            raise ValueError('Instance IDs must be consecutive 1..65535 within a camera')
        score = float(metadata['score'])
        if not np.isfinite(score):
            raise ValueError('Instance score must be finite')
        selected = mask[self.pixels[:, 1], self.pixels[:, 0]]
        update_points = selected & (score > self.point_scores)
        self.point_ids[update_points] = instance_id
        self.point_scores[update_points] = score
        update_image = mask & (score > self.image_scores)
        self.image_ids[update_image] = instance_id
        self.image_scores[update_image] = score
        self.packed_masks.append(np.packbits(mask.reshape(-1), bitorder='little'))
        self.records.append({**metadata, 'raw_projected_point_count': int(selected.sum())})

    def instance_points(self, camera: str, global_offset: int) -> list[dict]:
        return [{**record, 'camera': camera, 'camera_instance_id': record['instance_id'],
                 'global_instance_id': global_offset + record['instance_id'],
                 'point_indices': self.point_indices[self.point_ids == record['instance_id']].tolist()}
                for record in self.records]

    def save_masks(self, path: Path) -> None:
        packed_size = (self.image_ids.size + 7) // 8
        masks = np.stack(self.packed_masks) if self.packed_masks else np.empty((0, packed_size), dtype=np.uint8)
        np.savez_compressed(path, packed_masks=masks, mask_shape_hw=np.array(self.image_ids.shape, dtype=np.int32),
                            mask_ids=np.arange(1, len(self.records) + 1, dtype=np.uint16),
                            instance_map=self.image_ids, projected_point_indices=self.point_indices,
                            projected_pixels=self.pixels, projected_instance_ids=self.point_ids)


def instance_overlay(image: Image.Image, image_ids: np.ndarray, global_offset: int, alpha: float = .5) -> Image.Image:
    if image_ids.shape != (image.height, image.width):
        raise ValueError('Instance map must match the original image')
    colors = np.zeros((*image_ids.shape, 3), dtype=np.uint8)
    for local_id in np.unique(image_ids):
        if local_id:
            colors[image_ids == local_id] = instance_rgb(global_offset + int(local_id))
    pixels = np.asarray(image).astype(np.float32)
    selected = image_ids != 0
    pixels[selected] = pixels[selected] * (1 - alpha) + colors[selected] * alpha
    return Image.fromarray(np.rint(pixels).clip(0, 255).astype(np.uint8))


def camera_panel(image: Image.Image, overlay: Image.Image, projection: dict, collector: InstancePointCollector,
                 global_offset: int, title: str) -> Image.Image:
    point_image = image.copy()
    draw = ImageDraw.Draw(point_image)
    for (x, y), local_id in zip(projection['pixels'].tolist(), collector.point_ids.tolist()):
        if local_id:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=instance_rgb(global_offset + local_id))
    width, height = 640, max(1, round(image.height * 640 / image.width))
    panel = Image.new('RGB', (width * 3, height + 65), 'white')
    draw = ImageDraw.Draw(panel)
    for i, (label, source) in enumerate((('Original', image), ('SAM3 local instances', overlay), ('LiDAR points assigned to instances', point_image))):
        panel.paste(source.resize((width, height), Image.Resampling.BILINEAR), (i * width, 30))
        draw.text((i * width + 8, 8), label, fill='black')
    draw.text((8, height + 40), title + ' | matching instance colors are used in BEV; IDs are camera-local, not tracks', fill='black')
    return panel


def write_result_page(output: Path, frames: list[Path]) -> None:
    blocks = []
    for frame in frames:
        images = []
        if (frame / 'ground/ground_comparison.png').is_file():
            images.append(('Road and sidewalk: cross-camera semantic planes', frame / 'ground/ground_comparison.png'))
        images.append(('BEV: observed points and minimal convex hulls', frame / 'bev_instance_comparison.png'))
        if (frame / 'merged/merge_comparison.png').is_file():
            images.extend([('Cross-camera shared-point merge: before / after', frame / 'merged/merge_comparison.png'),
                           ('Matched groups in detail', frame / 'merged/matched_groups.png')])
        images.extend((f'{camera}: original / SAM3 instances / projected LiDAR points', frame / 'cameras' / f'{camera}_comparison.jpg')
                      for camera in CAMERAS)
        blocks.append(f'<h2>{escape(frame.name)}</h2>')
        for title, path in images:
            relative = path.relative_to(output).as_posix()
            blocks.append(f'<section><h3>{escape(title)}</h3><a href="{escape(relative, quote=True)}"><img loading="lazy" src="{escape(relative, quote=True)}"></a></section>')
    page = ('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>SAM3 instance point hulls</title><style>body{font:16px system-ui;margin:24px auto;max-width:1500px;padding:0 18px;background:#f6f7fa}'
            'img{width:100%;height:auto}section{padding:12px;background:white;margin:18px 0}a{color:#1454b0}</style>'
            '<h1>SAM3 instances → LiDAR points → convex BEV polygons</h1>'
            '<p>凸包是包含观测点的最小凸多边形；填充区域不是新增观测或真值。Convex hulls fill unobserved interiors and may bridge gaps.</p>'
            '<p>Foreground instances (IDs 1–10) are the main map. Road/building/vegetation masks are shown separately as semantic regions. '
            'Camera-local detections are preserved; optional shared-point associations are shown separately and are not verified object identities.</p>' + ''.join(blocks))
    (output / 'result.html').write_text(page, encoding='utf-8')


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    result.add_argument('--dataset-dir', type=Path, default=ROOT / 'datasets/nuscenes')
    result.add_argument('--split', choices=('mini_val', 'mini_train', 'all'), default='mini_val')
    result.add_argument('--scenes', nargs='+')
    result.add_argument('--start-index', type=int, default=0, help='Six-camera LiDAR frame index')
    result.add_argument('--samples', type=int, default=1, help='Number of complete six-camera frames, minimum 1')
    result.add_argument('--checkpoint', type=Path, default=ROOT / 'models/sam3/sam3.pt')
    result.add_argument('--classes-file', type=Path, default=ROOT / 'tasks/nuscenes_sam3/config/nuscenes_classes.txt')
    result.add_argument('--score-threshold', type=float, default=.5)
    result.add_argument('--mask-threshold', type=float, default=.5)
    result.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    result.add_argument('--precision', choices=('bf16', 'fp32', 'fp16'), default='bf16')
    result.add_argument('--bev-resolution', type=float, default=.2)
    result.add_argument('--bev-roi-size', type=float, default=100.)
    result.add_argument('--no-ground-plane', action='store_true', help='Skip merged road/sidewalk ground-region interpolation')
    result.add_argument('--ground-max-edge', type=float, default=6., help='Maximum ground triangle edge length in meters')
    result.add_argument('--no-merge-overlap', action='store_true', help='Skip the separate cross-camera shared-point merge comparison')
    result.add_argument('--min-shared-points', type=int, default=3)
    result.add_argument('--min-overlap', type=float, default=.3, help='Shared points / smaller instance point set')
    result.add_argument('--output-dir', type=Path, help='New/empty directory; default outputs/nuscenes_instance_hulls/<timestamp>')
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    ensure_environment()
    from tasks.nuscenes_sam3.python.instance_merge import merge_instances_by_points
    merge_instances_by_points([], args.min_shared_points, args.min_overlap)
    if not np.isfinite(args.ground_max_edge) or args.ground_max_edge <= 0:
        raise ValueError('ground-max-edge must be finite and positive')
    if args.samples < 1 or args.start_index < 0:
        raise ValueError('samples must be >=1 and start-index nonnegative')
    if not all(np.isfinite(x) and 0 <= x <= 1 for x in (args.score_threshold, args.mask_threshold)):
        raise ValueError('Thresholds must be finite in [0,1]')
    if not np.isfinite(args.bev_resolution) or args.bev_resolution <= 0 or not np.isfinite(args.bev_roi_size) or args.bev_roi_size <= 0:
        raise ValueError('BEV resolution and ROI size must be finite positive values')
    cells = args.bev_roi_size / args.bev_resolution
    if not np.isfinite(cells) or not np.isclose(cells, round(cells), rtol=0, atol=1e-8):
        raise ValueError('BEV ROI size must be an integer multiple of resolution')
    output = (args.output_dir or ROOT / 'outputs/nuscenes_instance_hulls' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Output directory is not empty: {output}')
    categories, class_names = load_categories('lidarseg', 'lidarseg16', args.classes_file.resolve())
    dataset = NuScenesMini(args.dataset_dir.resolve())
    all_views = dataset.select_views(list(CAMERAS), args.split, args.scenes)
    frames = [all_views[i:i + 6] for i in range(0, len(all_views), 6)]
    if args.start_index >= len(frames):
        raise ValueError(f'start-index must be below {len(frames)} selected frames')
    selected = frames[args.start_index:args.start_index + args.samples]
    segmenter = Sam3Segmenter(args.checkpoint.resolve(), args.device, args.precision, args.score_threshold, args.mask_threshold)
    output.mkdir(parents=True, exist_ok=True)
    run = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    run.update(status='running', processed_frames=0, selected_frames=len(selected), python=sys.executable,
               started_at=datetime.now(timezone.utc).isoformat(),
               instance_scope='camera-local masks preserved; separate constrained shared-point cross-camera association unless disabled',
               overlapping_masks='highest confidence wins within a camera; ties preserve earlier detection',
               polygon='minimal convex hull of exclusive ROI projected XY points; no point interpolation or outlier trimming',
               evaluation=None, limitations='Sparse semantic lidarseg has no matching SAM3 instance truth; convex interior is hypothesized coverage.')
    save_json(output / 'run_config.json', run)
    frame_outputs = []
    try:
        for frame_index, views in enumerate(selected):
            first = views[0]
            full = dataset.load_lidar(first['lidar_token'])
            xyz = full['xyz']
            stem = f"{first['scene_name']}_{args.start_index + frame_index:06d}_{first['sample_token']}"
            frame_dir = output / stem
            cameras_dir = frame_dir / 'cameras'
            cameras_dir.mkdir(parents=True)
            instances = []
            point_camera_ids = np.zeros((6, len(xyz)), dtype=np.int32)
            for camera_index, view in enumerate(views):
                started = time.perf_counter()
                with Image.open(view['image_path']) as source:
                    image = source.convert('RGB')
                projection = dataset.projection(view)
                collector = InstancePointCollector(image.size, projection['point_indices'], projection['pixels'])
                offset = len(instances)
                prediction, _ = segmenter.predict(image, categories, instance_callback=collector)
                camera_instances = collector.instance_points(view['camera'], offset)
                instances.extend(camera_instances)
                point_camera_ids[camera_index, projection['point_indices']] = np.where(collector.point_ids != 0, collector.point_ids.astype(np.int32) + offset, 0)
                collector.save_masks(cameras_dir / f"{view['camera']}_instances.npz")
                save_json(cameras_dir / f"{view['camera']}_instances.json", camera_instances)
                Image.fromarray(prediction).save(cameras_dir / f"{view['camera']}_semantic_ids.png")
                overlay = instance_overlay(image, collector.image_ids, offset)
                overlay.save(cameras_dir / f"{view['camera']}_instances_overlay.jpg", quality=94)
                camera_panel(image, overlay, projection, collector, offset, view['camera']).save(cameras_dir / f"{view['camera']}_comparison.jpg", quality=94)
                print(f"{view['camera']}: {len(camera_instances)} masks, {(collector.point_ids != 0).sum()} assigned points, {time.perf_counter() - started:.1f}s", flush=True)
            np.savez_compressed(frame_dir / 'points.npz', xyz=xyz, raw_labels=full['raw_labels'],
                                camera_instance_ids=point_camera_ids, camera_names=np.array([view['camera'] for view in views]))
            hulls = save_hull_bev(frame_dir, xyz, instances, class_names, args.bev_resolution, args.bev_roi_size)
            summary = {'sample_token': first['sample_token'], 'lidar_token': first['lidar_token'],
                       'camera_names': [view['camera'] for view in views], 'instances': len(instances),
                       'point_observations': int(np.count_nonzero(point_camera_ids)),
                       'unique_assigned_points': int(np.any(point_camera_ids != 0, axis=0).sum()),
                       'hull_metadata': hulls['metadata']}
            save_json(frame_dir / 'summary.json', summary)
            if not args.no_merge_overlap:
                from tasks.nuscenes_sam3.python.merge_saved_instances import merge_frame
                merge_frame(frame_dir, frame_dir / 'merged', args.min_shared_points, args.min_overlap)
            if not args.no_ground_plane:
                from tasks.nuscenes_sam3.python.ground_regions import process_frame
                process_frame(frame_dir, frame_dir / 'ground', args.ground_max_edge)
            frame_outputs.append(frame_dir)
            run['processed_frames'] += 1
        write_result_page(output, frame_outputs)
        run['status'] = 'completed'
    except BaseException as error:
        run.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        raise
    finally:
        run['finished_at'] = datetime.now(timezone.utc).isoformat()
        save_json(output / 'run_config.json', run)
    print(f"Instance hull results: {output / 'result.html'}", flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, FileNotFoundError, FileExistsError) as error:
        print(f'Error: {error}', file=sys.stderr)
        raise SystemExit(1)

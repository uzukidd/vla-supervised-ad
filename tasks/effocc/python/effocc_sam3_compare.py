#!/usr/bin/env python
"""Compare same-image SAM3 segmentation with EFFOcc six-camera occupancy renders."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tasks.effocc.python.effocc_cameras import CAMERAS, camera_geometry, contact_sheet, projection_colors, render_frame
from shared.python.nuscenes_data import NuScenesMini, COARSE_CLASS_NAMES
from tasks.nuscenes_sam3.python.sam3_nuscenes import load_categories
from shared.python.runtime import ROOT, ensure_environment, save_json
from shared.python.sam3 import Sam3Segmenter, make_palette


def sam3_overlay(image: Image.Image, prediction: np.ndarray, palette: np.ndarray, alpha: float) -> Image.Image:
    if prediction.shape != (image.height, image.width) or prediction.dtype != np.uint8 or np.any(prediction > 16):
        raise ValueError('SAM3 prediction must be an original-size uint8 class map with IDs 0..16')
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('alpha must be in [0,1]')
    pixels = np.asarray(image.convert('RGB')).astype(np.float32)
    assigned = prediction != 0
    pixels[assigned] = pixels[assigned] * (1 - alpha) + palette[prediction[assigned]] * alpha
    return Image.fromarray(np.rint(pixels).clip(0, 255).astype(np.uint8))


def comparison_sheet(rows: list[tuple[str, Image.Image, Image.Image, Image.Image]], palette: np.ndarray,
                     title: str) -> Image.Image:
    """Each row is one camera; columns are original, SAM3 and occupancy overlay."""
    width, height, label_height = 640, 360, 28
    footer = 180
    result = Image.new('RGB', (3 * width, 45 + len(rows) * (height + label_height) + footer), 'white')
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 15)
    except OSError:
        font = ImageFont.load_default()
    draw.text((12, 10), title, fill='black', font=font)
    for row_index, (camera, original, sam3, occupancy) in enumerate(rows):
        if original.size != sam3.size or original.size != occupancy.size:
            raise ValueError('Original, SAM3 and occupancy panels must use the same image size')
        y = 45 + row_index * (height + label_height)
        for column, (name, panel) in enumerate((('Original', original), ('SAM3 image segmentation', sam3), ('EFFOcc occupancy projection', occupancy))):
            x = column * width
            draw.text((x + 10, y + 5), f'{camera} | {name}', fill='black', font=font)
            result.paste(panel.resize((width, height), Image.Resampling.BILINEAR), (x, y + label_height))
    top = result.height - footer
    for i in range(1, 17):
        x, y = 12 + ((i - 1) % 3) * width, top + 5 + ((i - 1) // 3) * 24
        draw.rectangle((x, y, x + 15, y + 15), fill=tuple(int(c) for c in palette[i]))
        draw.text((x + 22, y), f'{i}: {COARSE_CLASS_NAMES[i]}', fill='black', font=font)
    draw.text((12, result.height - 25), 'Shared IDs/colors 1..16. SAM3 0 = unassigned; EFFOcc 0 = others (gray), free = transparent, no ray hit = original.',
              fill='black', font=font)
    return result


def find_projections(dataset: NuScenesMini, frame: Path, projection_root: Path | None, device: str, alpha: float,
                     fallback_output: Path) -> Path:
    sample = json.loads((frame / 'sample.json').read_text(encoding='utf-8'))
    token = sample['input']['sample_token']
    candidates = []
    if projection_root is not None:
        candidates.extend([projection_root, projection_root / frame.name, projection_root / frame.name / 'camera_projections'])
    else:
        candidates.append(frame / 'camera_projections')
    for candidate in candidates:
        if (candidate / 'projection_config.json').is_file():
            config = json.loads((candidate / 'projection_config.json').read_text(encoding='utf-8'))
            if config.get('sample_token') == token and config.get('status') == 'completed':
                return candidate
    if projection_root is not None:
        raise ValueError(f'No completed projections for sample {token} found under {projection_root}')
    render_frame(dataset, frame, fallback_output, alpha=alpha, device=device)
    return fallback_output


def compare_frame(dataset: NuScenesMini, frame: Path, projection_dir: Path, output: Path, segmenter,
                  categories, alpha: float = .45) -> dict:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Output directory is not empty: {output}')
    sample = json.loads((frame / 'sample.json').read_text(encoding='utf-8'))
    info = sample['input']
    config = json.loads((projection_dir / 'projection_config.json').read_text(encoding='utf-8'))
    if config.get('status') != 'completed' or config.get('sample_token') != info['sample_token']:
        raise ValueError('Occupancy projection must be completed and match this exact sample token')
    projection_records = {camera['camera']: camera for camera in config['cameras']}
    if set(projection_records) != set(CAMERAS) or len(config['cameras']) != 6:
        raise ValueError('Occupancy projection must record exactly six distinct cameras')
    cameras = camera_geometry(dataset, info['sample_token'], config['reference_camera'])
    output.mkdir(parents=True, exist_ok=True)
    palette = make_palette()
    palette[0] = (100, 100, 100)
    report = {'sample_token': info['sample_token'], 'status': 'running', 'alpha': alpha, 'cameras': [],
              'source_occupancy': str(frame / 'occupancy.npz'), 'source_projections': str(projection_dir),
              'comparison': 'Same original image per camera, unchanged occupancy render and independent SAM3 text prompts',
              'shared_class_ids': list(range(1, 17)),
              'special_labels': {'sam3_0': 'unassigned (no text mask)', 'effocc_0': 'others (occupied)',
                                 'effocc_17': 'free (transparent)', 'projection_255': 'no ray hit'},
              'evaluation': None, 'limitations': 'Side-by-side predictions only; neither model is ground truth. '
                              'SAM3 sees the full original image; EFFOcc uses its configured crop and bounded occupancy volume.'}
    save_json(output / 'comparison_config.json', report)
    save_json(output / 'classes.json', [{'id': i, 'name': COARSE_CLASS_NAMES[i] if i else 'SAM3 unassigned / EFFOcc others',
              'color_rgb': palette[i].tolist(), 'sam3_prompts': list(next((category.prompts for category in categories if category.id == i), ()))}
             for i in range(17)])
    sam3_panels, rows = [], []
    try:
        for camera in cameras:
            started = time.perf_counter()
            name = camera['camera']
            projection_record = projection_records[name]
            if projection_record['camera_token'] != camera['camera_token']:
                raise ValueError(f'{name}: projection belongs to a different camera frame')
            with Image.open(camera['image_path']) as source:
                image = source.convert('RGB')
            with Image.open(projection_dir / f'{name}_class_ids.png') as source:
                occ_labels = np.asarray(source).copy()
            if occ_labels.shape != (image.height, image.width) or occ_labels.dtype != np.uint8:
                raise ValueError(f'{name}: occupancy projection dimensions/dtype do not match the original image')
            if np.any((occ_labels > 16) & (occ_labels != 255)):
                raise ValueError(f'{name}: expected occupancy surface IDs 0..16 or no-hit 255')
            from tasks.effocc.python.effocc_cameras import overlay_projection
            occ_overlay = overlay_projection(image, occ_labels, palette, alpha)
            prediction, instances = segmenter.predict(image, categories)
            segmentation_overlay = sam3_overlay(image, prediction, palette, alpha)
            semantic = palette[prediction].copy()
            semantic[prediction == 0] = 255
            image.save(output / f'{name}_original.jpg', quality=94)
            Image.fromarray(prediction).save(output / f'{name}_sam3_class_ids.png')
            Image.fromarray(semantic).save(output / f'{name}_sam3_semantics.png')
            segmentation_overlay.save(output / f'{name}_sam3_overlay.jpg', quality=94)
            Image.fromarray(occ_labels).save(output / f'{name}_occupancy_class_ids.png')
            projection_colors(occ_labels, palette).save(output / f'{name}_occupancy_semantics.png')
            occ_overlay.save(output / f'{name}_occupancy_overlay.jpg', quality=94)
            save_json(output / f'{name}_sam3_instances.json', instances)
            row = (name, image, segmentation_overlay, occ_overlay)
            comparison_sheet([row], palette, 'Original | SAM3 | EFFOcc occupancy projection').save(output / f'{name}_comparison.jpg', quality=94)
            rows.append(row)
            sam3_panels.append((name, segmentation_overlay))
            report['cameras'].append({'camera': name, 'camera_token': camera['camera_token'],
                                      'image_path': str(camera['image_path']), 'image_size_wh': list(image.size),
                                      'sam3_instance_count': len(instances), 'sam3_assigned_fraction': float((prediction != 0).mean()),
                                      'occupancy_hit_fraction': float((occ_labels != 255).mean()),
                                      'seconds': time.perf_counter() - started})
            print(f'{name}: SAM3 instances={len(instances)}, assigned={(prediction != 0).mean():.1%}, {time.perf_counter() - started:.1f}s', flush=True)
        comparison_sheet(rows, palette, 'Same sample / six cameras | original vs SAM3 vs EFFOcc occupancy projection').save(output / 'six_cameras_comparison.jpg', quality=94)
        contact_sheet(sam3_panels, tuple(COARSE_CLASS_NAMES.values()), palette, 'SAM3 segmentation | nuScenes 16 prompts | class 0 unassigned', legend=False).save(output / 'six_cameras_sam3.jpg', quality=94)
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        raise
    finally:
        save_json(output / 'comparison_config.json', report)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    result.add_argument('--input-dir', type=Path, required=True, help='Saved EFFOcc run root or single sample directory')
    result.add_argument('--projection-dir', type=Path, help='Existing projection root or sample projection directory; otherwise reuse automatic projections or render them')
    result.add_argument('--dataset-dir', type=Path, default=ROOT / 'datasets/nuscenes')
    result.add_argument('--sam3-checkpoint', type=Path, default=ROOT / 'models/sam3/sam3.pt')
    result.add_argument('--classes-file', type=Path, default=ROOT / 'tasks/nuscenes_sam3/config/nuscenes_classes.txt', help='Complete nuScenes 16-class prompt mapping, retaining original class names')
    result.add_argument('--max-samples', type=int, default=1, help='0 compares all saved samples')
    result.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    result.add_argument('--precision', choices=('bf16', 'fp32', 'fp16'), default='bf16')
    result.add_argument('--score-threshold', type=float, default=.5)
    result.add_argument('--mask-threshold', type=float, default=.5)
    result.add_argument('--alpha', type=float, default=.45)
    result.add_argument('--output-dir', type=Path, help='New/empty directory; default <input-dir>/sam3_comparison')
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    ensure_environment()
    if args.max_samples < 0 or not all(np.isfinite(x) and 0 <= x <= 1 for x in (args.alpha, args.score_threshold, args.mask_threshold)):
        raise ValueError('max-samples must be nonnegative; alpha and thresholds must be finite in [0,1]')
    source = args.input_dir.resolve()
    frames = [source] if (source / 'occupancy.npz').is_file() else sorted(path.parent for path in source.glob('*/occupancy.npz'))
    if not frames:
        raise FileNotFoundError(f'No saved EFFOcc occupancy found under {source}')
    frames = frames[:args.max_samples or len(frames)]
    output = (args.output_dir or source / 'sam3_comparison').resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Output directory is not empty: {output}')
    dataset = NuScenesMini(args.dataset_dir.resolve())
    categories, _ = load_categories('lidarseg', 'lidarseg16', args.classes_file.resolve())
    # Render missing occupancy views before allocating SAM3 GPU memory.
    projections = [find_projections(dataset, frame, args.projection_dir, args.device, args.alpha,
                                   output / 'occupancy_projections' / frame.name) for frame in frames]
    segmenter = Sam3Segmenter(args.sam3_checkpoint.resolve(), args.device, args.precision, args.score_threshold, args.mask_threshold)
    output.mkdir(parents=True, exist_ok=True)
    run = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    run.update(status='running', processed_samples=0, python=sys.executable, started_at=datetime.now(timezone.utc).isoformat())
    save_json(output / 'run_config.json', run)
    try:
        for frame, projection in zip(frames, projections):
            compare_frame(dataset, frame, projection, output / frame.name, segmenter, categories, args.alpha)
            run['processed_samples'] += 1
        run['status'] = 'completed'
    except BaseException as error:
        run.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        raise
    finally:
        run['finished_at'] = datetime.now(timezone.utc).isoformat()
        save_json(output / 'run_config.json', run)
    print(f'SAM3 / EFFOcc comparisons: {output}', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, FileNotFoundError, FileExistsError) as error:
        print(f'Error: {error}', file=sys.stderr)
        raise SystemExit(1)

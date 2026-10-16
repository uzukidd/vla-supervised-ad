#!/usr/bin/env python
"""Merge saved camera-local SAM3 instances by shared original LiDAR points."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

from shared.python.runtime import ROOT, ensure_environment, save_json
from tasks.nuscenes_sam3.python.instance_hulls import convex_hull_xy, instance_rgb, polygon_area, points_in_convex_polygon
from tasks.nuscenes_sam3.python.instance_merge import merge_instances_by_points


def enrich_hulls(xyz: np.ndarray, rows: list[dict], half: float, merged: bool) -> list[dict]:
    result = []
    inside = ((xyz[:, :2] >= -half) & (xyz[:, :2] < half)).all(1)
    for row in rows:
        indices = np.asarray(row['point_indices'], dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(xyz)):
            raise ValueError('Instance point indices must refer to this original LiDAR scan')
        roi_indices = indices[inside[indices]]
        vertices = convex_hull_xy(xyz[roi_indices, :2])
        if not points_in_convex_polygon(xyz[roi_indices, :2], vertices).all():
            raise RuntimeError('Merged hull does not cover every supporting point')
        identifier = row['merged_instance_id'] if merged else row['global_instance_id']
        result.append({**row, 'display_id': int(identifier), 'roi_point_indices': roi_indices.tolist(),
                       'support_count': len(roi_indices), 'hull_xy': vertices.tolist(), 'area_m2': polygon_area(vertices),
                       'geometry_type': ('empty', 'point', 'segment')[len(vertices)] if len(vertices) < 3 else 'polygon'})
    return result


def render_hulls(xyz: np.ndarray, rows: list[dict], bounds: tuple[float, float, float, float],
                 size: tuple[int, int], label: bool = True) -> Image.Image:
    xmin, ymin, xmax, ymax = bounds
    width, height = size
    def pixels(points):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        xy = np.empty_like(points, dtype=np.int64)
        xy[:, 0] = np.clip(np.floor((points[:, 0] - xmin) / (xmax - xmin) * width), 0, width - 1)
        xy[:, 1] = height - 1 - np.clip(np.floor((points[:, 1] - ymin) / (ymax - ymin) * height), 0, height - 1)
        return [tuple(int(value) for value in pair) for pair in xy]
    image = Image.new('RGB', size, 'white')
    mask = (xyz[:, 0] >= xmin) & (xyz[:, 0] < xmax) & (xyz[:, 1] >= ymin) & (xyz[:, 1] < ymax)
    ImageDraw.Draw(image).point(pixels(xyz[mask, :2]), fill=(195, 195, 195))
    for row in sorted(rows, key=lambda row: (row['score'], row['display_id'])):
        if not row['support_count']:
            continue
        rgb = instance_rgb(row['display_id'])
        vertices = pixels(row['hull_xy'])
        draw = ImageDraw.Draw(image, 'RGBA')
        if len(vertices) >= 3:
            draw.polygon(vertices, fill=(*rgb, 48), outline=(*rgb, 255), width=2)
        elif len(vertices) == 2:
            draw.line(vertices, fill=(*rgb, 255), width=2)
        ImageDraw.Draw(image).point(pixels(xyz[row['roi_point_indices'], :2]), fill=rgb)
        if label:
            x, y = pixels(np.asarray(row['hull_xy']).mean(axis=0, keepdims=True))[0]
            text = f"#{row['display_id']} {row['class_name']}"
            if row.get('merged'):
                text += ' [' + ','.join(str(value) for value in row['member_instance_ids']) + ']'
            draw = ImageDraw.Draw(image)
            box = draw.textbbox((0, 0), text)
            x = max(0, min(x + 3, width - (box[2] - box[0]) - 2))
            y = max(0, min(y + 3, height - 15))
            draw.text((x, y), text, fill=tuple(int(c * .65) for c in rgb))
    return image


def merge_frame(frame: Path, output: Path, min_shared_points: int = 3, min_overlap: float = .3) -> dict:
    frame, output = Path(frame), Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Merge output is not empty: {output}')
    source = json.loads((frame / 'instance_hulls.json').read_text(encoding='utf-8'))
    with np.load(frame / 'points.npz', allow_pickle=False) as archive:
        xyz = archive['xyz'].copy()
        camera_ids = archive['camera_instance_ids'].copy() if 'camera_instance_ids' in archive else None
        camera_names = archive['camera_names'].copy() if 'camera_names' in archive else None
    original_rows = source['instances']
    merged = merge_instances_by_points(original_rows, min_shared_points, min_overlap)
    half = source['metadata']['roi_size_m'] / 2
    width, height = source['metadata']['width'], source['metadata']['height']
    bounds = (-half, -half, half, half)
    original = enrich_hulls(xyz, original_rows, half, False)
    groups = enrich_hulls(xyz, merged['groups'], half, True)
    before_fg = [row for row in original if row['class_id'] <= 10]
    after_fg = [row for row in groups if row['class_id'] <= 10]
    before_image = render_hulls(xyz, before_fg, bounds, (width, height))
    after_image = render_hulls(xyz, after_fg, bounds, (width, height))
    all_image = render_hulls(xyz, groups, bounds, (width, height))
    output.mkdir(parents=True, exist_ok=True)
    before_image.save(output / 'bev_before_merge.png')
    after_image.save(output / 'bev_after_merge.png')
    all_image.save(output / 'bev_all_groups.png')
    comparison = Image.new('RGB', (width * 2, height + 150), 'white')
    comparison.paste(before_image, (0, 30))
    comparison.paste(after_image, (width, 30))
    draw = ImageDraw.Draw(comparison)
    draw.text((8, 8), 'Before: camera-local foreground instances', fill='black')
    draw.text((width + 8, 8), 'After: shared-point merged foreground instances', fill='black')
    active_before = sum(bool(row['support_count']) for row in before_fg)
    active_after = sum(bool(row['support_count']) for row in after_fg)
    merged_groups = [row for row in groups if row['merged']]
    lines = [f'ROI {half*2:g}x{half*2:g}m | {source["metadata"]["resolution_m_per_pixel"]:g} m/cell | foreground groups with ROI points {active_before} -> {active_after}',
             f'Same class, different cameras, shared >= {min_shared_points}, overlap/smaller >= {min_overlap:.0%}; {len(merged_groups)} merged groups',
             'At most one detection per camera in a group; all member pairs must pass the point-overlap thresholds.',
             'Group ID/color = smallest member ID. Shared point indices are deduplicated before recomputing the hull.',
             'Association is a hypothesis, not GT identity. Hulls can still contain unobserved regions and outlier points.']
    for index, line in enumerate(lines):
        draw.text((8, height + 40 + index * 19), line, fill='black')
    comparison.save(output / 'merge_comparison.png')
    by_id = {row['global_instance_id']: row for row in original}
    zoom_rows = max(1, len(merged_groups))
    zoom = Image.new('RGB', (1000, zoom_rows * 550 + 35), 'white')
    zoom_draw = ImageDraw.Draw(zoom)
    zoom_draw.text((8, 8), 'Matched groups | exact source points; each row uses its own XY zoom in meters', fill='black')
    if not merged_groups:
        zoom_draw.text((20, 80), 'No cross-camera instance pairs met the configured evidence thresholds.', fill='black')
    for i, group in enumerate(merged_groups):
        vertices = np.asarray(group['hull_xy'], dtype=float).reshape(-1, 2)
        if not len(vertices):
            # Match used full point sets; a merged group can be wholly outside the displayed ROI.
            zoom_draw.text((10, 40 + i * 550), f"Group #{group['display_id']} has no support points inside the ROI", fill='black')
            continue
        lo, hi = vertices.min(axis=0), vertices.max(axis=0)
        center = (lo + hi) / 2
        side = max(float((hi - lo).max()) * 1.25, 4.)
        crop = (center[0] - side / 2, center[1] - side / 2, center[0] + side / 2, center[1] + side / 2)
        members = [by_id[identifier] for identifier in group['member_instance_ids']]
        before = render_hulls(xyz, members, crop, (500, 500))
        after = render_hulls(xyz, [group], crop, (500, 500))
        top = 35 + i * 550
        zoom.paste(before, (0, top + 25))
        zoom.paste(after, (500, top + 25))
        zoom_draw.text((8, top + 5), 'Before: ' + ','.join(f"#{row['display_id']} {row['camera']}" for row in members), fill='black')
        zoom_draw.text((508, top + 5), f"After #{group['display_id']} | unique points={len(group['point_indices'])} | {group['class_name']}", fill='black')
        zoom_draw.text((8, top + 528), f'XY zoom: x=[{crop[0]:.2f},{crop[2]:.2f}], y=[{crop[1]:.2f},{crop[3]:.2f}]m; hull area={group["area_m2"]:.3f}m2', fill='black')
    zoom.save(output / 'matched_groups.png')
    # Preserve every camera membership as well as the group-level deduplicated point sets.
    arrays = {'xyz': xyz}
    if camera_ids is not None:
        mapping = {int(old): int(new) for old, new in merged['mapping'].items()}
        mapped = np.zeros_like(camera_ids)
        for old in np.unique(camera_ids):
            if old:
                mapped[camera_ids == old] = mapping[int(old)]
        arrays.update(camera_instance_ids=camera_ids, camera_merged_ids=mapped, camera_names=camera_names)
    np.savez_compressed(output / 'merged_points.npz', **arrays)
    save_json(output / 'merged_hulls.json', {'metadata': merged['metadata'], 'groups': groups})
    save_json(output / 'match_pairs.json', merged['pairs'])
    save_json(output / 'instance_mapping.json', merged['mapping'])
    if merged['pairs']:
        fields = list(merged['pairs'][0])
        with (output / 'match_pairs.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(merged['pairs'])
    summary = {'source_frame': str(frame), 'status': 'completed', 'thresholds': {'min_shared_points': min_shared_points, 'min_overlap_smaller': min_overlap},
               'foreground_with_roi_points_before': active_before, 'foreground_with_roi_points_after': active_after,
               'merged_group_count': len(merged_groups), 'metadata': merged['metadata'],
               'merged_groups': [{'merged_instance_id': group['merged_instance_id'], 'members': group['member_instance_ids'],
                                  'cameras': group['source_cameras'], 'class_name': group['class_name'],
                                  'unique_points': len(group['point_indices']), 'hull_area_m2': group['area_m2']} for group in merged_groups]}
    save_json(output / 'summary.json', summary)
    print(f"Shared-point merge: {active_before} -> {active_after} foreground groups with ROI points; merged groups={len(merged_groups)}", flush=True)
    return summary


def write_merge_page(output: Path, frames: list[Path]) -> None:
    sections = []
    for frame in frames:
        summary = json.loads((frame / 'summary.json').read_text())
        sections.append(f'<h2>{escape(frame.name)}</h2><p>前景组 / foreground groups: {summary["foreground_with_roi_points_before"]} → {summary["foreground_with_roi_points_after"]}</p>')
        for title, filename in [('Before / after cross-camera merge', 'merge_comparison.png'), ('Matched group detail', 'matched_groups.png')]:
            path = (frame / filename).relative_to(output).as_posix()
            sections.append(f'<section><h3>{escape(title)}</h3><a href="{escape(path, quote=True)}"><img src="{escape(path, quote=True)}"></a></section>')
        record = (frame / 'match_pairs.json').relative_to(output).as_posix()
        sections.append(f'<p><a href="{escape(record, quote=True)}">Pairwise evidence and merge decisions / 逐对关联证据</a></p>')
    text = ('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Shared-point instance merging</title><style>body{font:16px system-ui;max-width:1400px;margin:24px auto;padding:0 18px;background:#f5f6fa}'
            'img{width:100%;height:auto}section{background:white;padding:14px;margin:20px 0}</style>'
            '<h1>Cross-camera instance merging by shared LiDAR points</h1>'
            '<p>Same-class foreground only. Shared original point indices, no GT or BEV-polygon-overlap matching. '
            'Groups contain at most one instance per camera; every member pair must meet the thresholds. '
            'Unmatched instances and stuff masks remain separate. Associations are hypotheses, not verified identities.</p>' + ''.join(sections))
    (output / 'result.html').write_text(text, encoding='utf-8')


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    result.add_argument('--input-dir', type=Path, required=True, help='Existing SAM3 instance experiment root or one saved frame')
    result.add_argument('--output-dir', type=Path, help='New/empty directory; default outputs/nuscenes_instance_merge/<timestamp>')
    result.add_argument('--min-shared-points', type=int, default=3)
    result.add_argument('--min-overlap', type=float, default=.3, help='Intersection / smaller point-set size')
    result.add_argument('--samples', type=int, default=1, help='Saved LiDAR frames, minimum 1')
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    ensure_environment()
    if args.samples < 1:
        raise ValueError('samples must be at least 1')
    # Validate thresholds before writing a partial run.
    merge_instances_by_points([], args.min_shared_points, args.min_overlap)
    source = args.input_dir.resolve()
    frames = [source] if (source / 'instance_hulls.json').is_file() else sorted(path.parent for path in source.glob('*/instance_hulls.json'))
    if not frames:
        raise FileNotFoundError(f'No saved instance hulls under {source}')
    output = (args.output_dir or ROOT / 'outputs/nuscenes_instance_merge' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Output directory is not empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    config = {'input_dir': str(source), 'status': 'running', 'processed_frames': 0,
              'min_shared_points': args.min_shared_points, 'min_overlap': args.min_overlap}
    save_json(output / 'run_config.json', config)
    completed = []
    try:
        for frame in frames[:args.samples]:
            target = output / frame.name
            merge_frame(frame, target, args.min_shared_points, args.min_overlap)
            completed.append(target)
            config['processed_frames'] += 1
        write_merge_page(output, completed)
        config['status'] = 'completed'
    except BaseException as error:
        config.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        raise
    finally:
        save_json(output / 'run_config.json', config)
    print(f'Merge results: {output / "result.html"}', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, FileNotFoundError, FileExistsError) as error:
        print(f'Error: {error}', file=sys.stderr)
        raise SystemExit(1)

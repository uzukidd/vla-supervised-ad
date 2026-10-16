"""One command per task: run required stages and collect final comparison images."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import shutil
import subprocess
import sys

from shared.python.runtime import ROOT, ensure_environment, save_json

TASKS = ('sidewalk', 'nuimages', 'nuscenes_sam3', 'effocc')
DESCRIPTIONS = {
    'sidewalk': 'Original image / SAM3 segmentation / dataset ground truth',
    'nuimages': 'Original image / SAM3 segmentation / dataset ground truth / errors',
    'nuscenes_sam3': 'Six-camera SAM3 segmentation / projected lidarseg ground truth / semantic BEV',
    'effocc': 'Same six camera images: original / SAM3 / EFFOcc occupancy projection',
}


def build_stages(task: str, samples: int, start: int, output: Path, device: str) -> list[dict]:
    """Translate user sample units to each existing entry point without changing models."""
    if task not in TASKS or samples < 1 or start < 0 or device not in ('cuda', 'cpu'):
        raise ValueError('Choose a supported task/device, samples >= 1 and start-index >= 0')
    raw = output / 'raw'
    precision = 'bf16' if device == 'cuda' else 'fp32'
    common = ['--device', device]
    sam3 = common + ['--precision', precision]
    if task in ('sidewalk', 'nuimages'):
        module = 'tasks.sidewalk.python.sam3_sidewalk' if task == 'sidewalk' else 'tasks.nuimages.python.sam3_nuimages'
        return [{'name': 'SAM3 segmentation, GT comparison and evaluation',
                 'module': module,
                 'args': ['--max-images', str(samples), '--start-index', str(start), '--output-dir', str(raw), *sam3]}]
    if task == 'nuscenes_sam3':
        return [{'name': 'Six-camera SAM3 segmentation, lidarseg evaluation and BEV',
                 'module': 'tasks.nuscenes_sam3.python.sam3_nuscenes',
                 'args': ['--cameras', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
                          '--max-images', str(samples * 6), '--start-index', str(start * 6), '--export-pointcloud',
                          '--bev-roi-size', '100', '--bev-resolution', '0.2', '--output-dir', str(raw), *sam3]}]
    return [
        {'name': 'EFFOcc dual-modal inference and six-camera occupancy projection',
         'module': 'tasks.effocc.python.effocc_mini',
         'args': ['--max-samples', str(samples), '--start-index', str(start), '--output-dir', str(raw), *common]},
        {'name': 'SAM3 segmentation and same-frame EFFOcc comparison',
         'module': 'tasks.effocc.python.effocc_sam3_compare',
         'args': ['--input-dir', str(raw), '--max-samples', str(samples), '--output-dir', str(output / 'comparison'), *sam3]},
    ]


def collect_results(task: str, output: Path) -> list[dict]:
    """Keep a small final-image folder; retain all intermediate labels under raw/."""
    results = output / 'results'
    results.mkdir(exist_ok=True)
    images = []
    if task in ('sidewalk', 'nuimages'):
        candidates = [(path, 'SAM3 and dataset ground truth') for path in sorted((output / 'raw/visualizations').glob('*.jpg'))]
    elif task == 'effocc':
        comparison = output / 'comparison'
        candidates = [(path, 'All six cameras: original / SAM3 / EFFOcc') for path in sorted(comparison.glob('*/six_cameras_comparison.jpg'))]
        candidates += [(path, 'Camera comparison: original / SAM3 / EFFOcc') for path in sorted(comparison.glob('*/CAM_*_comparison.jpg'))]
    else:
        bev_root = output / 'raw/pointcloud_results/pointclouds'
        bev_frames = sorted(path.parent for path in bev_root.glob('*/points.npz'))
        if not bev_frames:
            raise RuntimeError('nuScenes BEV output is missing; expected raw/pointcloud_results/pointclouds/<frame>/points.npz')
        candidates = []
        for frame in bev_frames:
            required = (
                ('bev_prediction.png', 'BEV: SAM3 semantic majority (500x500 by default)'),
                ('bev_ground_truth.png', 'BEV: sparse lidarseg ground-truth majority'),
                ('bev_comparison.png', 'BEV: prediction / GT / cell comparison / camera coverage'),
                ('ground/ground_plane.png', 'BEV: continuous road and sidewalk semantic plane (inferred fill)'),
                ('ground/ground_sources.png', 'BEV: observed / inferred / non-ground veto source map'),
                ('ground/ground_comparison.png', 'BEV: sparse ground / continuous plane / source comparison'),
            )
            for relative, caption in required:
                source = frame / relative
                if not source.is_file():
                    raise RuntimeError(f'nuScenes final BEV image is missing: {source}. Refresh saved results to fill the ground outputs.')
                candidates.append((source, caption))
        candidates += [(path, 'Camera: SAM3 and projected lidarseg ground truth') for path in sorted((output / 'raw/visualizations').glob('*.jpg'))]
    for index, (path, caption) in enumerate(candidates, 1):
        # Parent name distinguishes different frames without exposing the raw directory tree.
        if task == 'nuscenes_sam3' and path.suffix.lower() == '.png':
            frame = path.parent.parent if path.parent.name == 'ground' else path.parent
            destination = results / 'BEV' / frame.name / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
        else:
            filename = f'{index:03d}_{path.parent.name}_{path.name}'
            destination = results / filename
        shutil.copyfile(path, destination)
        images.append({'path': destination.relative_to(output).as_posix(), 'caption': caption})
    if not images:
        raise RuntimeError('The pipeline completed without final comparison images; inspect raw/ and run_config.json')
    # Metrics are already in JSON/CSV. Copy them without merging incompatible scopes.
    metric_files = [
        ('image_or_camera_metrics', output / 'raw'),
        ('unique_point_metrics', output / 'raw/pointcloud_results/evaluation'),
    ]
    for prefix, directory in metric_files:
        for filename in ('metrics.json', 'metrics.csv', 'confusion_matrix.csv', 'coverage.csv'):
            source = directory / filename
            if source.is_file():
                shutil.copyfile(source, results / f'{prefix}_{filename}')
    save_json(results / 'images.json', images)
    description = DESCRIPTIONS[task]
    cards = ''
    if task == 'nuscenes_sam3':
        cards += '<h2 id="bev">BEV 俯视图 / Semantic BEV and ground planes</h2><p>每帧包含原始语义BEV、GT和连续地面。Continuous planes contain inferred fill; use the source map to distinguish observations.</p>'
        cards += ''.join(f'<section><h3>{escape(row["caption"])} — {escape(Path(row["path"]).parent.name)}</h3><a href="{escape(row["path"], quote=True)}">'
                         f'<img loading="lazy" src="{escape(row["path"], quote=True)}" alt="{escape(row["caption"], quote=True)}"></a></section>'
                         for row in images if row['path'].startswith('results/BEV/'))
        cards += '<h2 id="cameras">六相机图像 / Camera comparisons</h2>'
        cards += ''.join(f'<section><h3>{escape(row["caption"])}</h3><a href="{escape(row["path"], quote=True)}">'
                         f'<img loading="lazy" src="{escape(row["path"], quote=True)}" alt="{escape(row["caption"], quote=True)}"></a></section>'
                         for row in images if not row['path'].startswith('results/BEV/'))
    else:
        cards = ''.join(f'<section><h2>{escape(row["caption"])}</h2><a href="{escape(row["path"], quote=True)}">'
                        f'<img loading="lazy" src="{escape(row["path"], quote=True)}" alt="{escape(row["caption"], quote=True)}"></a></section>'
                        for row in images)
    metrics = sorted(results.glob('*metrics*'))
    links = ''.join(f'<li><a href="{escape(path.relative_to(output).as_posix(), quote=True)}">{escape(path.name)}</a></li>' for path in metrics)
    notice = ('SAM3 and EFFOcc are independent predictions. Shared class colors 1–16; this comparison has no ground-truth accuracy metric.'
              if task == 'effocc' else 'Metrics use the existing dataset-specific ignore policy. Sparse lidarseg projection is not dense image ground truth.'
              if task == 'nuscenes_sam3' else 'Dataset ground truth is compared with SAM3; only valid annotated pixels enter evaluation.')
    html = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{escape(task)} comparison results</title>'
            '<style>body{font:16px system-ui,sans-serif;max-width:1500px;margin:28px auto;padding:0 18px;background:#f4f6f9;color:#182536}'
            'section{background:white;padding:16px;margin:24px 0;border-radius:10px}h2{font-size:18px}img{width:100%;height:auto}a{color:#1953b0}</style>'
            f'<h1>{escape(task)} — results</h1><p>{escape(description)}</p><p>{escape(notice)}</p>'
            '<p>点击图片打开完整结果。Click an image for the full-size comparison.</p>'
            + ('<p><a href="#bev">BEV俯视图 / BEV</a> · <a href="#cameras">六相机 / Cameras</a></p>' if task == 'nuscenes_sam3' else '')
            + (f'<h2>Evaluation files / 评估文件</h2><ul>{links}</ul>' if links else '') + cards + '</html>')
    (output / 'result.html').write_text(html, encoding='utf-8')
    return images


def refresh_results(task: str, output: Path) -> dict:
    """Add missing presentation products to a completed run without model execution."""
    output = output.resolve()
    config_path = output / 'run_config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config.get('task') != task or config.get('status') != 'completed':
        raise ValueError('Refresh requires a completed one-click output of the selected task')
    if task == 'nuscenes_sam3':
        from tasks.nuscenes_sam3.python.ground_regions import process_frame
        bev_frames = sorted(path.parent for path in (output / 'raw/pointcloud_results/pointclouds').glob('*/points.npz'))
        raw_config = json.loads((output / 'raw/run_config.json').read_text(encoding='utf-8'))
        taxonomy = raw_config.get('taxonomy', 'lidarseg16')
        for frame in bev_frames:
            ground = frame / 'ground'
            if not (ground / 'ground_comparison.png').is_file():
                process_frame(frame, ground, taxonomy=taxonomy)
    images = collect_results(task, output)
    config.update(comparison_images=len(images), result_page=str(output / 'result.html'),
                  results_refreshed_at=datetime.now(timezone.utc).isoformat())
    save_json(config_path, config)
    print(f'Refreshed without running models: {output / "result.html"}', flush=True)
    return config


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description='Run a complete task and save final comparison images.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    result.add_argument('--task', choices=TASKS, required=True)
    result.add_argument('--samples', type=int, default=1, help='Image count for Sidewalk/nuImages; six-camera frame count for nuScenes/EFFOcc')
    result.add_argument('--start-index', type=int, default=0, help='Starting image or six-camera frame index')
    result.add_argument('--output-dir', type=Path, help='New/empty output directory; default outputs/<task>/<timestamp>')
    result.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    result.add_argument('--refresh-results', action='store_true', help='Refresh an existing completed --output-dir, including missing ground planes, without running models')
    result.add_argument('--dry-run', action='store_true', help='Print the pipeline without loading models or creating output')
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    ensure_environment()
    if args.refresh_results:
        if args.output_dir is None or args.dry_run:
            raise ValueError('--refresh-results requires the existing --output-dir and cannot be combined with --dry-run')
        refresh_results(args.task, args.output_dir)
        return 0
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = (args.output_dir or ROOT / 'outputs' / args.task / stamp).resolve()
    stages = build_stages(args.task, args.samples, args.start_index, output, args.device)
    if args.dry_run:
        print(json.dumps({'task': args.task, 'output': str(output), 'stages': stages, 'final_result': str(output / 'result.html')}, indent=2))
        return 0
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Output directory is not empty: {output}. Omit --output-dir to create a new timestamped run.')
    output.mkdir(parents=True, exist_ok=True)
    run = {'task': args.task, 'samples': args.samples, 'start_index': args.start_index, 'device': args.device,
           'output_dir': str(output), 'status': 'running', 'stages': stages, 'completed_stages': 0,
           'started_at': datetime.now(timezone.utc).isoformat()}
    save_json(output / 'run_config.json', run)
    try:
        for index, stage in enumerate(stages, 1):
            print(f'\n[{index}/{len(stages)}] {stage["name"]}', flush=True)
            # Separate stages release the EFFOcc model before SAM3 allocates GPU memory.
            subprocess.run([sys.executable, '-m', stage['module'], *stage['args']], cwd=ROOT, check=True)
            run['completed_stages'] += 1
            save_json(output / 'run_config.json', run)
        images = collect_results(args.task, output)
        run.update(status='completed', comparison_images=len(images), result_page=str(output / 'result.html'))
    except BaseException as error:
        run.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        raise
    finally:
        run['finished_at'] = datetime.now(timezone.utc).isoformat()
        save_json(output / 'run_config.json', run)
    print(f'\n完成 / Completed: {output / "result.html"}\n对比图 / Comparison images: {output / "results"}', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, FileExistsError, subprocess.CalledProcessError) as error:
        print(f'Error: {error}', file=sys.stderr)
        raise SystemExit(1)

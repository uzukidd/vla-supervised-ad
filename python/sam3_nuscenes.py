#!/usr/bin/env python
"""SAM3 nuScenes mini image segmentation with projected lidarseg point evaluation."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

from nuscenes_data import CAMERAS, COARSE_CLASS_NAMES, RAW_CLASS_NAMES, RAW_TO_COARSE, NuScenesMini
from sam3_sidewalk import (ROOT, Category, Sam3Segmenter, colorize, draw_text, ensure_environment,
                          make_palette, overlay, parse_class_file, save_json)
from segmentation_metrics import ConfusionMatrix


def load_categories(mode: str, taxonomy: str, class_file: Path) -> tuple[list[Category], dict[int, str]]:
    entries = parse_class_file(class_file)
    if mode == "custom":
        categories = [Category(i, name, prompts) for i, (name, prompts) in enumerate(entries, 1)]
        return categories, {0: "unassigned", **{c.id: c.name for c in categories}}
    labels = COARSE_CLASS_NAMES if taxonomy == "lidarseg16" else RAW_CLASS_NAMES
    expected = {name for i, name in labels.items() if i != 0}
    prompts = dict(entries)
    missing, extra = expected - prompts.keys(), prompts.keys() - expected
    if missing or extra:
        raise ValueError(f"Prompt names must match {taxonomy}: missing={sorted(missing)}, extra={sorted(extra)}")
    return [Category(i, name, prompts[name]) for i, name in sorted(labels.items()) if i != 0], dict(labels)


def point_overlay(image: Image.Image, pixels: np.ndarray, labels: np.ndarray, palette: np.ndarray, radius: int = 2) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for (x, y), label in zip(pixels.tolist(), labels.tolist()):
        color = tuple(int(value) for value in palette[label])
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return result


def save_visualization(output: Path, stem: str, image: Image.Image, prediction: np.ndarray,
                       labels: dict[int, str], palette: np.ndarray, alpha: float,
                       projection: dict | None, target: np.ndarray | None) -> None:
    panels = [("Image", image), ("SAM3 overlay", overlay(image, prediction, palette, alpha)),
              ("SAM3 semantic map", colorize(prediction, palette))]
    present = set(np.unique(prediction).tolist())
    if projection is not None:
        gt_image = point_overlay(image, projection["pixels"], target, palette)
        predicted_points = prediction[projection["pixels"][:, 1], projection["pixels"][:, 0]]
        pred_image = point_overlay(image, projection["pixels"], predicted_points, palette)
        gt_image.save(output / "point_overlays" / f"{stem}_gt.jpg", quality=92)
        pred_image.save(output / "point_overlays" / f"{stem}_prediction.jpg", quality=92)
        panels.append(("Projected LiDAR GT (sparse)", gt_image))
        present.update(np.unique(target).tolist())
    width = min(640, image.width)
    height = max(1, round(image.height * width / image.width))
    present = sorted(present)
    legend_height = ((len(present) + 2) // 3) * 25 + 15
    canvas = Image.new("RGB", (width * len(panels), height + 30 + legend_height), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (title, panel) in enumerate(panels):
        resampling = Image.Resampling.NEAREST if title == "SAM3 semantic map" else Image.Resampling.BILINEAR
        canvas.paste(panel.resize((width, height), resampling), (i * width, 30))
        draw_text(draw, (i * width + 10, 6), title)
    cell_width = canvas.width // 3
    for i, label in enumerate(present):
        x, y = (i % 3) * cell_width + 10, height + 40 + (i // 3) * 25
        draw.rectangle((x, y, x + 15, y + 15), fill=tuple(int(c) for c in palette[label]))
        draw_text(draw, (x + 22, y), f"{label}: {labels[label]}")
    canvas.save(output / "visualizations" / f"{stem}.jpg", quality=92)
    overlay(image, prediction, palette, alpha).save(output / "overlays" / f"{stem}.jpg", quality=92)
    colorize(prediction, palette).save(output / "color_masks" / f"{stem}.png")


def save_point_metrics(metrics: ConfusionMatrix, output: Path, configuration: dict, totals: dict) -> None:
    metrics.save(output)
    report = metrics.report()
    report.update({
        "point_accuracy": report["pixel_accuracy"],
        "valid_point_observations": report["valid_pixels"],
        "evaluation_scope": {
            "unit": "retained projected LiDAR point observations per camera image",
            "taxonomy": configuration["taxonomy"], "split": configuration["split"],
            "cameras": configuration["cameras"], "processed_camera_images": configuration["processed_images"],
            "run_status": configuration["status"], "projection_counts": totals,
            "ground_truth": "lidarseg labels from the same sample's LIDAR_TOP keyframe",
            "ignored_gt_ids": [0],
            "prediction_sampling": "floor(u), floor(v) on the full-resolution semantic mask",
            "projection": "LiDAR sensor -> LiDAR-time ego -> global -> camera-time ego -> camera sensor -> intrinsic",
            "filter": f"camera depth > {configuration['min_depth']}m, 1-pixel image margin; nearest point per integer pixel",
            "limitations": "Sparse camera projection evaluation, not dense-image or official full-360-degree lidarseg benchmark. "
                           "A point in multiple selected cameras counts once per camera. Nearest-point filtering only resolves "
                           "same-pixel collisions; occlusion across different pixels and moving-object time offsets remain.",
        },
    })
    report["metric_definitions"]["pixel_accuracy"] = "Alias of point_accuracy: correct retained non-ignored point observations / valid point observations."
    report["metric_definitions"]["count_unit"] = "Fields named pixels in the JSON compatibility aliases are projected point observations in this run; CSV uses point field names."
    for row in report["per_class"]:
        row["gt_points"] = row["gt_pixels"]
        row["predicted_points"] = row["predicted_pixels"]
    save_json(output / "metrics.json", report)
    fields = ("scope", "unit", "metric", "value", "id", "name", "ignored", "iou", "dice", "precision", "recall",
              "gt_points", "predicted_points", "tp", "fp", "fn")
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name in ("point_accuracy", "mean_iou", "mean_accuracy", "frequency_weighted_iou", "macro_dice"):
            writer.writerow({"scope": "aggregate", "unit": "projected_point_observations", "metric": name, "value": report[name]})
        for row in report["per_class"]:
            writer.writerow({"scope": "per_class", "unit": "projected_point_observations",
                             **{key: row[key] for key in fields[4:]}})


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    result.add_argument("--mode", choices=("lidarseg", "custom"), default="lidarseg")
    result.add_argument("--dataset-dir", type=Path, default=ROOT / "datasets/nuscenes")
    result.add_argument("--version", choices=("v1.0-mini",), default="v1.0-mini")
    result.add_argument("--split", choices=("all", "mini_train", "mini_val"), default="mini_val")
    result.add_argument("--scenes", nargs="+", help="Scene names within the selected split, e.g. scene-0103")
    result.add_argument("--cameras", nargs="+", choices=CAMERAS, default=["CAM_FRONT"])
    result.add_argument("--taxonomy", choices=("lidarseg16", "lidarseg32"), default="lidarseg16")
    result.add_argument("--classes-file", type=Path, help="TXT prompts; default selected by mode and taxonomy")
    result.add_argument("--checkpoint", type=Path, default=ROOT / "models/sam3/sam3.pt")
    result.add_argument("--output-dir", type=Path, help="New or empty output directory")
    result.add_argument("--start-index", type=int, default=0, help="Index into selected camera views, not LiDAR frames")
    result.add_argument("--max-images", type=int, default=10, help="Number of camera images; 0 processes all selected views")
    result.add_argument("--min-depth", type=float, default=1.0, help="Minimum positive camera-frame depth in meters")
    result.add_argument("--score-threshold", type=float, default=0.5)
    result.add_argument("--mask-threshold", type=float, default=0.5)
    result.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    result.add_argument("--precision", choices=("bf16", "fp32", "fp16"), default="bf16")
    result.add_argument("--alpha", type=float, default=0.5)
    result.add_argument("--vis-every", type=int, default=1)
    result.add_argument("--no-vis", action="store_true")
    result.add_argument("--export-pointcloud", action="store_true", help="Fuse camera labels into original LiDAR points; export PLY/NPZ/3D viewer and unique-point metrics")
    result.add_argument("--html-max-points", type=int, default=60000, help="Maximum rendered HTML points; exports and metrics retain every point")
    result.add_argument("--inspect-data", action="store_true", help="Inspect first selected view and point projection without loading SAM3")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    ensure_environment()
    if args.start_index < 0 or args.max_images < 0 or args.vis_every < 1:
        raise ValueError("start-index/max-images must be nonnegative; vis-every must be positive")
    if args.html_max_points < 1:
        raise ValueError("html-max-points must be positive")
    if not np.isfinite(args.min_depth) or args.min_depth <= 0:
        raise ValueError("min-depth must be finite and positive")
    if not all(0 <= value <= 1 for value in (args.score_threshold, args.mask_threshold, args.alpha)):
        raise ValueError("thresholds/alpha must be in [0, 1]")
    args.dataset_dir = args.dataset_dir.resolve()
    args.checkpoint = args.checkpoint.resolve()
    default_class_file = "classes.txt" if args.mode == "custom" else (
        "nuscenes_classes.txt" if args.taxonomy == "lidarseg16" else "nuscenes_lidarseg32_classes.txt")
    args.classes_file = (args.classes_file or ROOT / "config" / default_class_file).resolve()
    args.output_dir = (args.output_dir or ROOT / f"outputs/nuscenes_{args.mode}").resolve()
    categories, labels = load_categories(args.mode, args.taxonomy, args.classes_file)
    dataset = NuScenesMini(args.dataset_dir, args.version)
    views = dataset.select_views(args.cameras, args.split, args.scenes)
    if args.start_index >= len(views):
        raise ValueError(f"start-index {args.start_index} must be below selected camera view count {len(views)}")
    count = min(args.max_images or len(views), len(views) - args.start_index)
    selected = views[args.start_index:args.start_index + count]
    evaluate = args.mode == "lidarseg"
    if args.inspect_data:
        first = selected[0]
        with Image.open(first["image_path"]) as image:
            size = image.size
        report = {"python": sys.executable, "version": args.version, "split": args.split,
                  "selected_camera_views": len(views), "requested_image_count": count,
                  "first_view": {key: str(value) if isinstance(value, Path) else value for key, value in first.items()},
                  "image_size_wh": list(size), "taxonomy": args.taxonomy, "class_count": len(categories)}
        if evaluate:
            projection = dataset.projection(first, args.min_depth)
            report.update({"projection_counts": projection["counts"], "raw_label_ids": np.unique(projection["raw_labels"]).tolist(),
                           "mapped_label_ids": np.unique(RAW_TO_COARSE[projection["raw_labels"]]).tolist() if args.taxonomy == "lidarseg16" else None})
        print(json.dumps(report, indent=2))
        return 0
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}; choose a new --output-dir")
    # Preflight the first projection before loading the large model.
    first_projection = dataset.projection(selected[0], args.min_depth) if evaluate else None
    segmenter = Sam3Segmenter(args.checkpoint, args.device, args.precision, args.score_threshold, args.mask_threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    directories = ["predictions", "instances"]
    if evaluate:
        directories.extend(["projected_points", "sparse_gt"])
    if not args.no_vis:
        directories.extend(["visualizations", "overlays", "color_masks"])
        if evaluate:
            directories.append("point_overlays")
    for directory in directories:
        (args.output_dir / directory).mkdir()
    palette = make_palette()
    save_json(args.output_dir / "classes.json", [{"id": i, "name": name, "color_rgb": palette[i].tolist(),
              "prompts": list(next((c.prompts for c in categories if c.id == i), ()))} for i, name in sorted(labels.items())])
    if evaluate:
        save_json(args.output_dir / "lidarseg_mapping.json", [{"raw_id": i, "raw_name": name,
                  "prediction_id": int(RAW_TO_COARSE[i]) if args.taxonomy == "lidarseg16" else i,
                  "prediction_name": labels[int(RAW_TO_COARSE[i]) if args.taxonomy == "lidarseg16" else i]}
                 for i, name in sorted(RAW_CLASS_NAMES.items())])
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    configuration.update({"python": sys.executable, "selected_view_count": len(views), "selected_image_count": count,
                          "started_at": datetime.now(timezone.utc).isoformat(), "status": "running", "processed_images": 0,
                          "evaluation": "sparse_projected_lidarseg_point_observations" if evaluate else None,
                          "torch_version": segmenter.torch.__version__, "numpy_version": np.__version__,
                          "fusion": "highest instance confidence; ties preserve earlier category/prompt/instance"})
    metrics = ConfusionMatrix(labels, ignore_ids={0}) if evaluate else None
    totals = {}
    save_json(args.output_dir / "run_config.json", configuration)
    started = time.perf_counter()
    print(f"Mode={args.mode}, taxonomy={args.taxonomy}, images={count}/{len(views)}, output={args.output_dir}", flush=True)
    try:
        with (args.output_dir / "samples.jsonl").open("w", encoding="utf-8") as manifest:
            for offset, view in enumerate(selected):
                sample_started = time.perf_counter()
                with Image.open(view["image_path"]) as source:
                    image = source.convert("RGB")
                projection = (first_projection if offset == 0 else dataset.projection(view, args.min_depth)) if evaluate else None
                if projection is not None and list(image.size) != list(projection["image_size_wh"]):
                    raise ValueError(f"Image dimensions do not match camera metadata for {view['camera_token']}")
                prediction, instances = segmenter.predict(image, categories)
                stem = f"{view['scene_name']}_{args.start_index + offset:06d}_{view['camera']}_{view['sample_token'][:8]}"
                target = None
                observation_count = None
                if evaluate:
                    target = RAW_TO_COARSE[projection["raw_labels"]] if args.taxonomy == "lidarseg16" else projection["raw_labels"]
                    pixels = projection["pixels"]
                    point_prediction = prediction[pixels[:, 1], pixels[:, 0]]
                    metrics.update(target, point_prediction)
                    observation_count = int((target != 0).sum())
                    np.savez_compressed(args.output_dir / "projected_points" / f"{stem}.npz",
                                        point_indices=projection["point_indices"], uv=projection["uv"], pixels=pixels,
                                        depth=projection["depth"], raw_labels=projection["raw_labels"],
                                        target=target, prediction=point_prediction)
                    sparse = np.full(prediction.shape, 255, dtype=np.uint8)
                    sparse[pixels[:, 1], pixels[:, 0]] = target
                    Image.fromarray(sparse).save(args.output_dir / "sparse_gt" / f"{stem}.png")
                    for key, value in projection["counts"].items():
                        totals[key] = totals.get(key, 0) + int(value)
                Image.fromarray(prediction).save(args.output_dir / "predictions" / f"{stem}.png")
                save_json(args.output_dir / "instances" / f"{stem}.json", instances)
                visualized = not args.no_vis and offset % args.vis_every == 0
                if visualized:
                    save_visualization(args.output_dir, stem, image, prediction, labels, palette, args.alpha, projection, target)
                entry = {key: str(value) if isinstance(value, Path) else value for key, value in view.items()}
                entry.update({"index": args.start_index + offset, "stem": stem, "image_size_wh": list(image.size),
                              "instances": len(instances), "assigned_pixel_fraction": float((prediction != 0).mean()),
                              "valid_point_observations": observation_count, "projection_counts": projection["counts"] if evaluate else None,
                              "visualized": visualized, "seconds": time.perf_counter() - sample_started})
                manifest.write(json.dumps(entry) + "\n")
                manifest.flush()
                configuration["processed_images"] += 1
                print(f"[{offset + 1}/{count}] {stem}: instances={len(instances)}, eval_points={observation_count}, "
                      f"{time.perf_counter() - sample_started:.1f}s", flush=True)
        configuration["status"] = "completed"
    except BaseException as error:
        configuration.update({"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "error": str(error)})
        raise
    finally:
        configuration.update({"elapsed_seconds": time.perf_counter() - started,
                              "finished_at": datetime.now(timezone.utc).isoformat(), "projection_counts": totals})
        save_json(args.output_dir / "run_config.json", configuration)
        if metrics is not None:
            save_point_metrics(metrics, args.output_dir, configuration, totals)
    if metrics is not None:
        report = metrics.report()
        print(f"Projected-point mIoU={report['mean_iou']}, point accuracy={report['pixel_accuracy']}, "
              f"valid observations={report['valid_pixels']}", flush=True)
    if args.export_pointcloud:
        from nuscenes_pointcloud import export_run
        export_run(args.output_dir, args.output_dir / "pointcloud_results", html_max_points=args.html_max_points,
                   make_preview=not args.no_vis, make_html=not args.no_vis)
    print(f"Results saved to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)

#!/usr/bin/env python
"""SAM3 nuImages segmentation, decoded semantic GT comparison and evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

from tasks.nuimages.python.nuimages_data import NuImagesData
from shared.python.nuscenes_data import CAMERAS
from shared.python.runtime import ROOT, ensure_environment, save_json
from shared.python.sam3 import (Category, Sam3Segmenter, colorize, draw_text,
                                make_palette, overlay, parse_class_file)
from shared.python.segmentation_metrics import ConfusionMatrix


def load_categories(dataset: NuImagesData, mode: str, class_file: Path, include_ego: bool) -> tuple[list[Category], dict[int, str]]:
    entries = parse_class_file(class_file)
    if mode == "custom":
        categories = [Category(i, name, prompts) for i, (name, prompts) in enumerate(entries, 1)]
        return categories, {0: "unassigned", **{c.id: c.name for c in categories}}
    labels = dict(dataset.class_names)
    expected = {labels[i] for i in dataset.active_class_ids}
    prompts = dict(entries)
    missing, extra = expected - prompts.keys(), prompts.keys() - expected
    if missing or extra:
        raise ValueError(f"nuImages prompt taxonomy mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    categories = [Category(i, labels[i], prompts[labels[i]]) for i in sorted(dataset.active_class_ids)
                  if include_ego or i != 31]
    return categories, labels


def error_image(target: np.ndarray, prediction: np.ndarray, ignore_ids: set[int]) -> Image.Image:
    pixels = np.full((*target.shape, 3), (70, 75, 85), dtype=np.uint8)
    valid = ~np.isin(target, list(ignore_ids))
    pixels[valid & (target == prediction)] = (45, 190, 90)
    pixels[valid & (target != prediction)] = (240, 70, 65)
    return Image.fromarray(pixels)


def save_visualization(output: Path, stem: str, image: Image.Image, prediction: np.ndarray,
                       target: np.ndarray | None, labels: dict[int, str], palette: np.ndarray,
                       alpha: float, ignore_ids: set[int]) -> None:
    panels = [("Image", image), ("SAM3 overlay", overlay(image, prediction, palette, alpha)),
              ("SAM3 semantic map", colorize(prediction, palette))]
    if target is not None:
        gt_overlay = overlay(image, target, palette, alpha)
        errors = error_image(target, prediction, ignore_ids)
        panels.extend([("Ground truth overlay", gt_overlay), ("Ground truth semantic map", colorize(target, palette)),
                       ("Errors: green correct / red wrong", errors)])
        gt_overlay.save(output / "gt_overlays" / f"{stem}.jpg", quality=92)
        colorize(target, palette).save(output / "gt_color_masks" / f"{stem}.png")
        errors.save(output / "errors" / f"{stem}.png")
    width = min(640, image.width)
    height = max(1, round(image.height * width / image.width))
    columns = 3
    rows = (len(panels) + columns - 1) // columns
    present = sorted(set(np.unique(prediction).tolist()) | (set(np.unique(target).tolist()) if target is not None else set()))
    legend_height = ((len(present) + columns - 1) // columns) * 25 + (40 if target is not None else 15)
    canvas = Image.new("RGB", (width * columns, (height + 30) * rows + legend_height), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (title, panel) in enumerate(panels):
        x, y = (i % columns) * width, (i // columns) * (height + 30)
        resample = Image.Resampling.NEAREST if "map" in title or "Errors" in title else Image.Resampling.BILINEAR
        canvas.paste(panel.resize((width, height), resample), (x, y + 30))
        draw_text(draw, (x + 10, y + 6), title)
    legend_top = (height + 30) * rows + 10
    for i, label in enumerate(present):
        x, y = (i % columns) * width + 10, legend_top + (i // columns) * 25
        draw.rectangle((x, y, x + 15, y + 15), fill=tuple(int(c) for c in palette[label]))
        suffix = " (ignored GT)" if target is not None and label in ignore_ids else ""
        draw_text(draw, (x + 22, y), f"{label}: {labels.get(label, 'unknown')}{suffix}")
    if target is not None:
        draw_text(draw, (10, canvas.height - 25), "Error map: gray = ignored/unannotated; all metrics use annotated non-ignored GT pixels.")
    canvas.save(output / "visualizations" / f"{stem}.jpg", quality=92)
    overlay(image, prediction, palette, alpha).save(output / "overlays" / f"{stem}.jpg", quality=92)
    colorize(prediction, palette).save(output / "color_masks" / f"{stem}.png")


def save_metrics(metrics: ConfusionMatrix, output: Path, configuration: dict, totals: dict) -> None:
    metrics.save(output)
    report = metrics.report()
    report["evaluation_scope"] = {
        "dataset": "nuImages", "version": configuration["version"], "mode": "semantic pixel evaluation",
        "processed_images": configuration["processed_images"], "run_status": configuration["status"],
        "ignored_gt_ids": sorted(metrics.ignore_ids), "annotation_counts": totals,
        "image_pixels": totals.get("image_pixels", 0),
        "annotated_pixels": totals.get("annotated_pixels", 0),
        "valid_gt_pixel_fraction": report["valid_pixels"] / totals["image_pixels"] if totals.get("image_pixels") else None,
        "annotated_pixel_fraction": totals["annotated_pixels"] / totals["image_pixels"] if totals.get("image_pixels") else None,
        "gt_composition": "Base64 COCO RLE; surfaces in JSON order first, objects sorted by annotation token overwrite surfaces",
        "class_mapping": "Foreground alphabetical IDs 1..23; driveable_surface=24; ego=31; 25..30 unused",
        "prediction_zero": "On valid GT pixels, prediction0 remains an error; no filtering by successful SAM3 coverage",
        "limitations": "Semantic evaluation of annotated foreground/surface pixels. Unlabeled background is ignored; "
                       "this does not provide full-scene labels for sky/buildings/vegetation or instance AP evaluation.",
    }
    save_json(output / "metrics.json", report)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    result.add_argument("--mode", choices=("nuimages", "custom"), default="nuimages")
    result.add_argument("--dataset-dir", type=Path, default=ROOT / "datasets/nuimages")
    result.add_argument("--version", choices=("v1.0-mini", "v1.0-train", "v1.0-val", "v1.0-test"), default="v1.0-mini")
    result.add_argument("--cameras", choices=CAMERAS, nargs="+", help="Filter annotated key cameras; default all channels")
    result.add_argument("--classes-file", type=Path, help="TXT prompts; default selected by mode")
    result.add_argument("--include-ego", action="store_true", help="Prompt and evaluate vehicle.ego (31); default ignores ego")
    result.add_argument("--checkpoint", type=Path, default=ROOT / "models/sam3/sam3.pt")
    result.add_argument("--output-dir", type=Path, help="New or empty output directory")
    result.add_argument("--start-index", type=int, default=0)
    result.add_argument("--max-images", type=int, default=10, help="0 processes all remaining selected keyframes")
    result.add_argument("--score-threshold", type=float, default=0.5)
    result.add_argument("--mask-threshold", type=float, default=0.5)
    result.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    result.add_argument("--precision", choices=("bf16", "fp32", "fp16"), default="bf16")
    result.add_argument("--alpha", type=float, default=0.5)
    result.add_argument("--vis-every", type=int, default=1)
    result.add_argument("--no-vis", action="store_true")
    result.add_argument("--inspect-data", action="store_true", help="Inspect keyframes and decode first GT without loading SAM3")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    ensure_environment()
    if args.start_index < 0 or args.max_images < 0 or args.vis_every < 1:
        raise ValueError("start-index/max-images must be nonnegative; vis-every must be positive")
    if not all(0 <= value <= 1 for value in (args.alpha, args.score_threshold, args.mask_threshold)):
        raise ValueError("thresholds and alpha must be in [0, 1]")
    evaluate = args.mode == "nuimages"
    if evaluate and args.version == "v1.0-test":
        raise ValueError("nuImages test GT is unavailable; use custom mode or a labelled mini/train/val version")
    args.dataset_dir = args.dataset_dir.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.classes_file = (args.classes_file or (ROOT / "tasks/nuimages/config/nuimages_classes.txt" if evaluate else ROOT / "shared/config/classes.txt")).resolve()
    args.output_dir = (args.output_dir or ROOT / f"outputs/nuimages_{args.mode}").resolve()
    dataset = NuImagesData(args.dataset_dir, args.version, with_annotations=evaluate)
    categories, labels = load_categories(dataset, args.mode, args.classes_file, args.include_ego)
    views = dataset.select_images(args.cameras)
    if args.start_index >= len(views):
        raise ValueError(f"start-index {args.start_index} must be below selected image count {len(views)}")
    count = min(args.max_images or len(views), len(views) - args.start_index)
    selected = views[args.start_index:args.start_index + count]
    ignore_ids = {0, 255, *range(25, 31)} | (set() if args.include_ego else {31})
    if args.inspect_data:
        first = selected[0]
        report = {"python": sys.executable, "version": args.version, "selected_image_count": len(views),
                  "requested_image_count": count, "camera_counts": {channel: sum(view["camera"] == channel for view in views)
                                                                      for channel in sorted({view["camera"] for view in views})},
                  "first_image": {key: str(value) if isinstance(value, Path) else value for key, value in first.items()},
                  "class_count": len(categories), "ignored_gt_ids": sorted(ignore_ids) if evaluate else []}
        if evaluate:
            target, stats = dataset.semantic_mask(first)
            report.update({"label_ids": np.unique(target).tolist(), "annotation_stats": stats,
                           "valid_gt_pixels": int((~np.isin(target, list(ignore_ids))).sum())})
        print(json.dumps(report, indent=2))
        return 0
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}; choose a new --output-dir")
    if evaluate:
        dataset.semantic_mask(selected[0])  # Validate RLE and image dimensions before loading SAM3.
    segmenter = Sam3Segmenter(args.checkpoint, args.device, args.precision, args.score_threshold, args.mask_threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    directories = ["predictions", "instances"]
    if evaluate:
        directories.append("ground_truth")
    if not args.no_vis:
        directories.extend(["visualizations", "overlays", "color_masks"])
        if evaluate:
            directories.extend(["gt_overlays", "gt_color_masks", "errors"])
    for directory in directories:
        (args.output_dir / directory).mkdir()
    palette = make_palette()
    save_json(args.output_dir / "classes.json", [{"id": i, "name": name, "active": i in dataset.active_class_ids if evaluate else i != 0,
              "color_rgb": palette[i].tolist(), "prompts": list(next((c.prompts for c in categories if c.id == i), ()))}
             for i, name in sorted(labels.items())])
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    configuration.update({"status": "running", "processed_images": 0, "python": sys.executable,
                          "selected_image_count": count, "available_selected_images": len(views),
                          "started_at": datetime.now(timezone.utc).isoformat(), "ignored_gt_ids": sorted(ignore_ids) if evaluate else [],
                          "fusion": "highest instance confidence; ties preserve earlier category/prompt/instance",
                          "torch_version": segmenter.torch.__version__, "numpy_version": np.__version__})
    save_json(args.output_dir / "run_config.json", configuration)
    metrics = ConfusionMatrix(labels, ignore_ids=ignore_ids) if evaluate else None
    totals = {"image_pixels": 0, "annotated_pixels": 0}
    started = time.perf_counter()
    print(f"Mode={args.mode}, version={args.version}, images={count}/{len(views)}, classes={len(categories)}, output={args.output_dir}", flush=True)
    try:
        with (args.output_dir / "samples.jsonl").open("w", encoding="utf-8") as manifest:
            for offset, view in enumerate(selected):
                sample_started = time.perf_counter()
                with Image.open(view["image_path"]) as source:
                    image = source.convert("RGB")
                target, stats = dataset.semantic_mask(view) if evaluate else (None, None)
                prediction, instances = segmenter.predict(image, categories)
                if evaluate:
                    metrics.update(target, prediction)
                    for name, value in stats.items():
                        totals[name] = totals.get(name, 0) + int(value)
                totals["image_pixels"] += image.width * image.height
                stem = f"{args.version}_{args.start_index + offset:06d}_{view['camera']}_{view['camera_token'][:8]}"
                Image.fromarray(prediction).save(args.output_dir / "predictions" / f"{stem}.png")
                save_json(args.output_dir / "instances" / f"{stem}.json", instances)
                if target is not None:
                    Image.fromarray(target).save(args.output_dir / "ground_truth" / f"{stem}.png")
                visualized = not args.no_vis and offset % args.vis_every == 0
                if visualized:
                    save_visualization(args.output_dir, stem, image, prediction, target, labels, palette, args.alpha, ignore_ids)
                record = {key: str(value) if isinstance(value, Path) else value for key, value in view.items()}
                record.update({"stem": stem, "index": args.start_index + offset, "instances": len(instances),
                               "assigned_pixel_fraction": float((prediction != 0).mean()), "annotation_stats": stats,
                               "valid_gt_pixels": int((~np.isin(target, list(ignore_ids))).sum()) if evaluate else None,
                               "visualized": visualized, "seconds": time.perf_counter() - sample_started})
                manifest.write(json.dumps(record) + "\n")
                manifest.flush()
                configuration["processed_images"] += 1
                print(f"[{offset + 1}/{count}] {stem}: instances={len(instances)}, valid_pixels={record['valid_gt_pixels']}, "
                      f"{time.perf_counter() - sample_started:.1f}s", flush=True)
        configuration["status"] = "completed"
    except BaseException as error:
        configuration.update({"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "error": str(error)})
        raise
    finally:
        configuration.update({"elapsed_seconds": time.perf_counter() - started,
                              "finished_at": datetime.now(timezone.utc).isoformat(), "annotation_counts": totals})
        save_json(args.output_dir / "run_config.json", configuration)
        if metrics is not None:
            save_metrics(metrics, args.output_dir, configuration, totals)
    if metrics is not None:
        report = metrics.report()
        print(f"mIoU={report['mean_iou']}, pixel accuracy={report['pixel_accuracy']}, valid_pixels={report['valid_pixels']}", flush=True)
    print(f"Results saved to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)

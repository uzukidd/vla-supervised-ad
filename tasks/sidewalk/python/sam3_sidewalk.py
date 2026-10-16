#!/usr/bin/env python
"""SAM3 text-prompt segmentation for local Sidewalk Parquet data (WSL only)."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

from shared.python.segmentation_metrics import ConfusionMatrix

from shared.python.runtime import ROOT, ensure_environment, save_json
from shared.python.sam3 import (Category, parse_class_file, merge_instance, Sam3Segmenter,
                                make_palette, colorize, overlay, draw_text)


def load_categories(mode: str, dataset_dir: Path, class_file: Path) -> tuple[list[Category], dict[int, str]]:
    entries = parse_class_file(class_file)
    if mode == "custom":
        categories = [Category(i, name, prompts) for i, (name, prompts) in enumerate(entries, 1)]
        return categories, {0: "unassigned", **{c.id: c.name for c in categories}}
    labels = {int(k): v for k, v in json.loads((dataset_dir / "id2label.json").read_text()).items()}
    if sorted(labels) != list(range(len(labels))) or len(labels) > 255 or labels.get(0) != "unlabeled":
        raise ValueError("Expected contiguous Sidewalk class IDs starting at 0 (unlabeled).")
    by_name = dict(entries)
    expected = set(labels.values()) - {labels[0]}
    missing, extra = expected - by_name.keys(), by_name.keys() - expected
    if missing or extra:
        raise ValueError(f"Sidewalk prompt file must retain every taxonomy name; missing={sorted(missing)}, extra={sorted(extra)}")
    categories = [Category(i, name, by_name[name]) for i, name in sorted(labels.items()) if i != 0]
    return categories, labels


def parquet_files(dataset_dir: Path, split: str) -> list[Path]:
    files = sorted((dataset_dir / "data").glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No data/{split}-*.parquet files found under {dataset_dir}")
    return files


def dataset_info(files: list[Path], with_labels: bool) -> dict:
    import pyarrow.parquet as pq

    rows = 0
    shards = []
    for path in files:
        parquet = pq.ParquetFile(path)
        columns = parquet.schema_arrow.names
        required = {"pixel_values", "label"} if with_labels else {"pixel_values"}
        if not required.issubset(columns):
            raise ValueError(f"{path}: required columns {required}, found {columns}")
        rows += parquet.metadata.num_rows
        shards.append({"path": str(path), "rows": parquet.metadata.num_rows, "schema": str(parquet.schema_arrow)})
    return {"rows": rows, "shards": shards}


def decode_image(cell, dataset_dir: Path, shard: Path) -> Image.Image:
    if isinstance(cell, dict):
        image_bytes = cell.get("bytes")
        if image_bytes is not None:
            with Image.open(BytesIO(image_bytes)) as image:
                return image.copy()
        cell = cell.get("path")
    if isinstance(cell, (bytes, bytearray, memoryview)):
        with Image.open(BytesIO(cell)) as image:
            return image.copy()
    if isinstance(cell, str):
        path = Path(cell)
        candidates = [path] if path.is_absolute() else [dataset_dir / path, shard.parent / path]
        for candidate in candidates:
            if candidate.is_file():
                with Image.open(candidate) as image:
                    return image.copy()
        raise FileNotFoundError(f"Image path {cell!r} is absent from {dataset_dir}")
    raise ValueError(f"Unsupported Parquet image cell: {type(cell).__name__}")


def iter_samples(files: list[Path], dataset_dir: Path, start: int, count: int, with_labels: bool):
    import pyarrow.parquet as pq

    absolute_index = 0
    stop = start + count
    columns = ["pixel_values", "label"] if with_labels else ["pixel_values"]
    for shard in files:
        parquet = pq.ParquetFile(shard)
        if absolute_index + parquet.metadata.num_rows <= start:
            absolute_index += parquet.metadata.num_rows
            continue
        for batch in parquet.iter_batches(batch_size=8, columns=columns):
            end = absolute_index + batch.num_rows
            lo, hi = max(0, start - absolute_index), min(batch.num_rows, stop - absolute_index)
            if lo < hi:
                for offset, row in enumerate(batch.slice(lo, hi - lo).to_pylist(), lo):
                    index = absolute_index + offset
                    image = decode_image(row["pixel_values"], dataset_dir, shard).convert("RGB")
                    target = None
                    if with_labels:
                        target = np.asarray(decode_image(row["label"], dataset_dir, shard))
                        if target.ndim != 2 or target.dtype.kind not in "ui":
                            raise ValueError(f"Sample {index}: GT must be a 2D integer class-ID mask, found {target.shape}/{target.dtype}")
                        if target.shape != (image.height, image.width):
                            raise ValueError(f"Sample {index}: image/GT dimensions differ")
                    yield index, image, target
            absolute_index = end
            if absolute_index >= stop:
                return


def save_visualization(output: Path, stem: str, image: Image.Image, prediction: np.ndarray, target: np.ndarray | None,
                       labels: dict[int, str], palette: np.ndarray, alpha: float) -> None:
    panels = [("Image", image), ("SAM3 overlay", overlay(image, prediction, palette, alpha)),
              ("SAM3 semantic map", colorize(prediction, palette))]
    if target is not None:
        panels.append(("Ground truth", colorize(target, palette)))
    preview_width = min(image.width, 640)
    preview_height = max(1, round(image.height * preview_width / image.width))
    present = sorted(set(np.unique(prediction).tolist()) | (set(np.unique(target).tolist()) if target is not None else set()))
    legend_columns = 3
    legend_height = ((len(present) + legend_columns - 1) // legend_columns) * 25 + 15
    canvas = Image.new("RGB", (preview_width * len(panels), preview_height + 30 + legend_height), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (title, panel) in enumerate(panels):
        canvas.paste(panel.resize((preview_width, preview_height), Image.Resampling.NEAREST if "map" in title or "truth" in title else Image.Resampling.BILINEAR),
                     (i * preview_width, 30))
        draw_text(draw, (i * preview_width + 10, 6), title)
    cell_width = canvas.width // legend_columns
    for i, label_id in enumerate(present):
        x, y = (i % legend_columns) * cell_width + 10, preview_height + 40 + (i // legend_columns) * 25
        draw.rectangle((x, y, x + 15, y + 15), fill=tuple(int(c) for c in palette[label_id]))
        name = labels.get(label_id, "ignored" if label_id == 255 else "unknown")
        draw_text(draw, (x + 22, y), f"{label_id}: {name}")
    canvas.save(output / "visualizations" / f"{stem}.jpg", quality=92)
    overlay(image, prediction, palette, alpha).save(output / "overlays" / f"{stem}.jpg", quality=92)
    colorize(prediction, palette).save(output / "color_masks" / f"{stem}.png")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    result.add_argument("--mode", choices=("sidewalk", "custom"), default="sidewalk")
    result.add_argument("--dataset-dir", type=Path, default=ROOT / "datasets/sidewalk-semantic")
    result.add_argument("--checkpoint", type=Path, default=ROOT / "models/sam3/sam3.pt")
    result.add_argument("--classes-file", type=Path, help="TXT class prompts; default selected by mode")
    result.add_argument("--output-dir", type=Path, help="Empty output directory; default outputs/sam3_<mode>")
    result.add_argument("--split", choices=("train", "validation", "test"), default="train")
    result.add_argument("--start-index", type=int, default=0)
    result.add_argument("--max-images", type=int, default=0, help="0 processes all remaining images")
    result.add_argument("--score-threshold", type=float, default=0.5)
    result.add_argument("--mask-threshold", type=float, default=0.5)
    result.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    result.add_argument("--precision", choices=("bf16", "fp32", "fp16"), default="bf16")
    result.add_argument("--alpha", type=float, default=0.5, help="Overlay opacity")
    result.add_argument("--vis-every", type=int, default=1, help="Visualize every Nth processed image")
    result.add_argument("--no-vis", action="store_true", help="Save class-ID masks and metrics without visualization")
    result.add_argument("--inspect-data", action="store_true", help="Inspect Parquet schema and first selected image without loading SAM3")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    ensure_environment()
    if args.start_index < 0 or args.max_images < 0 or args.vis_every < 1:
        raise ValueError("start-index/max-images must be nonnegative and vis-every must be positive")
    if not all(0 <= x <= 1 for x in (args.score_threshold, args.mask_threshold, args.alpha)):
        raise ValueError("score-threshold, mask-threshold and alpha must be in [0, 1]")
    args.dataset_dir = args.dataset_dir.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.classes_file = (args.classes_file or ROOT / ("tasks/sidewalk/config/sidewalk_classes.txt" if args.mode == "sidewalk" else "shared/config/classes.txt")).resolve()
    args.output_dir = (args.output_dir or ROOT / f"outputs/sam3_{args.mode}").resolve()
    categories, labels = load_categories(args.mode, args.dataset_dir, args.classes_file)
    with_labels = args.mode == "sidewalk"
    files = parquet_files(args.dataset_dir, args.split)
    info = dataset_info(files, with_labels)
    if args.start_index >= info["rows"]:
        raise ValueError(f"start-index {args.start_index} must be smaller than dataset size {info['rows']}")
    count = min(args.max_images or info["rows"], info["rows"] - args.start_index)
    if args.inspect_data:
        index, image, target = next(iter_samples(files, args.dataset_dir, args.start_index, 1, with_labels))
        info.update({"first_selected_index": index, "image_size_wh": list(image.size),
                     "label_ids": np.unique(target).tolist() if target is not None else None,
                     "class_count": len(categories), "prompt_count": sum(len(c.prompts) for c in categories),
                     "python": sys.executable})
        print(json.dumps(info, indent=2))
        return 0
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}. Choose a new --output-dir.")
    segmenter = Sam3Segmenter(args.checkpoint, args.device, args.precision, args.score_threshold, args.mask_threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for directory in ("predictions", "instances") + (() if args.no_vis else ("visualizations", "overlays", "color_masks")):
        (args.output_dir / directory).mkdir(exist_ok=True)
    palette = make_palette()
    save_json(args.output_dir / "classes.json", [{"id": i, "name": name, "color_rgb": palette[i].tolist(),
               "prompts": list(next((c.prompts for c in categories if c.id == i), ()))} for i, name in sorted(labels.items())])
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    configuration.update({"python": sys.executable, "started_at": datetime.now(timezone.utc).isoformat(),
                          "selected_image_count": count, "dataset_image_count": info["rows"],
                          "fusion": "highest instance confidence; ties preserve earlier category/prompt/instance",
                          "unassigned_prediction_id": 0, "evaluation_ignore_gt_ids": [0, 255] if with_labels else [],
                          "status": "running", "processed_images": 0,
                          "torch_version": segmenter.torch.__version__, "numpy_version": np.__version__})
    save_json(args.output_dir / "run_config.json", configuration)
    metrics = ConfusionMatrix(labels, ignore_ids={0, 255}) if with_labels else None
    print(f"Mode={args.mode}, images={count}, classes={len(categories)}, output={args.output_dir}", flush=True)
    started = time.perf_counter()
    processed = 0
    try:
        with (args.output_dir / "samples.jsonl").open("w", encoding="utf-8") as manifest:
            for index, image, target in iter_samples(files, args.dataset_dir, args.start_index, count, with_labels):
                sample_started = time.perf_counter()
                prediction, instances = segmenter.predict(image, categories)
                if metrics is not None:
                    metrics.update(target, prediction)
                stem = f"{args.split}_{index:06d}"
                Image.fromarray(prediction).save(args.output_dir / "predictions" / f"{stem}.png")
                save_json(args.output_dir / "instances" / f"{stem}.json", instances)
                visualized = not args.no_vis and processed % args.vis_every == 0
                if visualized:
                    save_visualization(args.output_dir, stem, image, prediction, target, labels, palette, args.alpha)
                manifest.write(json.dumps({"index": index, "stem": stem, "width": image.width, "height": image.height,
                                           "instances": len(instances), "assigned_pixel_fraction": float((prediction != 0).mean()),
                                           "visualized": visualized, "seconds": time.perf_counter() - sample_started}) + "\n")
                manifest.flush()
                processed += 1
                print(f"[{processed}/{count}] {stem}: instances={len(instances)}, assigned={(prediction != 0).mean():.1%}, "
                      f"{time.perf_counter() - sample_started:.1f}s", flush=True)
        if processed != count:
            raise RuntimeError(f"Expected {count} samples but read {processed}")
        configuration["status"] = "completed"
    except BaseException as error:
        configuration.update({"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "error": str(error)})
        raise
    finally:
        configuration.update({"processed_images": processed, "elapsed_seconds": time.perf_counter() - started,
                              "finished_at": datetime.now(timezone.utc).isoformat()})
        if metrics is not None:
            metrics.save(args.output_dir)
        save_json(args.output_dir / "run_config.json", configuration)
    if metrics is not None:
        report = metrics.report()
        print(f"mIoU={report['mean_iou']}, pixel accuracy={report['pixel_accuracy']}", flush=True)
    print(f"Results saved to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)

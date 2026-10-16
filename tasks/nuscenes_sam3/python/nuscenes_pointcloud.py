#!/usr/bin/env python
"""Flatten camera-colored nuScenes LiDAR points into semantic BEV images."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from shared.python.nuscenes_data import NuScenesMini, RAW_TO_COARSE
from tasks.nuscenes_sam3.python.pointcloud_semantics import UniquePointMetrics, fuse_point_labels
from tasks.nuscenes_sam3.python.pointcloud_bev import save_bev_images
from shared.python.runtime import ensure_environment, save_json


def export_run(input_dir: Path, output_dir: Path, dataset_dir: Path | None = None,
               max_frames: int = 0, bev_resolution: float = 0.2, make_preview: bool = True,
               bev_density_cap: int = 32, bev_roi_size: float = 100.0) -> dict:
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if max_frames < 0 or not np.isfinite(bev_resolution) or bev_resolution <= 0:
        raise ValueError("max-frames must be nonnegative; bev-resolution must be finite and positive")
    if not isinstance(bev_density_cap, int) or isinstance(bev_density_cap, bool) or bev_density_cap < 2:
        raise ValueError("bev-density-cap must be an integer of at least 2")
    if not np.isfinite(bev_roi_size) or bev_roi_size < 0:
        raise ValueError("bev-roi-size must be finite and nonnegative (0 selects automatic extent)")
    if bev_roi_size:
        cells = bev_roi_size / bev_resolution
        if not np.isfinite(cells) or not np.isclose(cells, round(cells), rtol=0, atol=1e-8) or round(cells) < 1:
            raise ValueError("bev-roi-size must be an integer multiple of bev-resolution")
    half_size = bev_roi_size / 2
    bev_bounds = (-half_size, -half_size, half_size, half_size) if bev_roi_size else None
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Output directory is not empty: {output_dir}; choose a new output directory")
    configuration = json.loads((input_dir / "run_config.json").read_text(encoding="utf-8"))
    source_status = configuration.get("status")
    if source_status not in ("completed", "interrupted", "failed"):
        raise ValueError("Wait for the segmentation run to finish before exporting its BEV results")
    classes = json.loads((input_dir / "classes.json").read_text(encoding="utf-8"))
    labels = {int(row["id"]): row["name"] for row in classes}
    if sorted(labels) != list(range(len(labels))):
        raise ValueError("Source class IDs must be contiguous from 0")
    palette = np.zeros((256, 3), dtype=np.uint8)
    for row in classes:
        palette[int(row["id"])] = row["color_rgb"]
    evaluate = configuration["mode"] == "lidarseg"
    dataset = NuScenesMini(dataset_dir or Path(configuration["dataset_dir"]), configuration["version"])
    groups = {}
    with (input_dir / "samples.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                groups.setdefault(record["lidar_token"], []).append(record)
    if not groups:
        raise ValueError("No completed camera images recorded in the source samples.jsonl")
    selected = list(groups.items())[:max_frames or len(groups)]
    output_dir.mkdir(parents=True, exist_ok=True)
    clouds_dir = output_dir / "pointclouds"
    clouds_dir.mkdir()
    save_json(output_dir / "classes.json", classes)
    state = {"input_dir": str(input_dir), "dataset_dir": str(dataset.root), "source_run_status": source_status,
             "source_processed_images": configuration.get("processed_images"),
             "mode": configuration["mode"], "taxonomy": configuration["taxonomy"],
             "coordinate_frame": "native LIDAR_TOP sensor coordinates, meters",
             "bev_roi_bounds": bev_bounds, "bev_roi_size_m": bev_roi_size,
             "fusion": "nonzero majority vote; ties prefer earlier camera in samples.jsonl; zero abstains",
             "bev_resolution_m_per_pixel": bev_resolution, "visualization": "2d_xy_bev",
             "bev_collision_rule": "majority_count_then_lowest_class_id",
             "bev_density_cap": bev_density_cap,
             "status": "running", "processed_frames": 0,
             "selected_frames": len(selected), "started_at": datetime.now(timezone.utc).isoformat()}
    save_json(output_dir / "run_config.json", state)
    metrics = UniquePointMetrics(labels) if evaluate else None
    try:
        with (output_dir / "frames.jsonl").open("w", encoding="utf-8") as manifest:
            for frame_index, (lidar_token, views) in enumerate(selected):
                full = dataset.load_lidar(lidar_token)
                xyz = full["xyz"]
                observations = []
                camera_tokens = set()
                for view in views:
                    if view["camera_token"] in camera_tokens:
                        raise ValueError(f"Repeated camera image for LiDAR token {lidar_token}")
                    camera_tokens.add(view["camera_token"])
                    view["image_path"] = Path(view["image_path"])
                    projection = dataset.projection(view, configuration["min_depth"])
                    with Image.open(input_dir / "predictions" / f"{view['stem']}.png") as image:
                        prediction = np.asarray(image)
                    width, height = projection["image_size_wh"]
                    if prediction.shape != (height, width) or prediction.dtype.kind not in "ui":
                        raise ValueError(f"Source prediction must be a full-resolution integer class map: {view['stem']}")
                    if prediction.size and (int(prediction.min()) < 0 or int(prediction.max()) >= len(labels)):
                        raise ValueError(f"Source prediction contains unknown class IDs: {view['stem']}")
                    pixels = projection["pixels"]
                    observations.append({"point_indices": projection["point_indices"],
                                         "prediction": prediction[pixels[:, 1], pixels[:, 0]]})
                fused = fuse_point_labels(len(xyz), observations, len(labels))
                prediction, covered = fused["prediction"], fused["covered"]
                target = (RAW_TO_COARSE[full["raw_labels"]] if configuration["taxonomy"] == "lidarseg16" else full["raw_labels"]) if evaluate else None
                frame_metrics = UniquePointMetrics(labels) if evaluate else None
                if evaluate:
                    frame_metrics.update(target, prediction, covered)
                    metrics.update(target, prediction, covered)
                    report = frame_metrics.report()
                else:
                    report = {"coverage": {"total_points": len(xyz), "covered_points": int(covered.sum()),
                                           "assigned_points": int((prediction != 0).sum()),
                                           "geometric_coverage": float(covered.mean()) if len(xyz) else None}}
                stem = f"{views[0]['scene_name']}_{frame_index:06d}_{views[0]['sample_token']}"
                frame_dir = clouds_dir / stem
                frame_dir.mkdir()
                arrays = {"xyz": xyz, "raw_labels": full["raw_labels"], "intensity": full["intensity"], "ring": full["ring"],
                          "point_indices": np.arange(len(xyz), dtype=np.int64), **fused,
                          "camera_names": np.asarray([view["camera"] for view in views])}
                if target is not None:
                    arrays["target"] = target
                np.savez_compressed(frame_dir / "points.npz", **arrays)
                if target is not None:
                    frame_metrics.save(frame_dir / "evaluation")
                bev_metadata = save_bev_images(frame_dir, xyz, prediction, target, covered, labels, palette, report,
                                               resolution=bev_resolution, make_images=make_preview, density_cap=bev_density_cap,
                                               bounds=bev_bounds)
                ground_metadata = None
                if evaluate and bev_roi_size:
                    from tasks.nuscenes_sam3.python.ground_regions import save_ground_planes
                    ground_prediction = RAW_TO_COARSE[prediction] if configuration['taxonomy'] == 'lidarseg32' else prediction
                    ground_metadata = save_ground_planes(
                        frame_dir / 'ground', xyz, ground_prediction,
                        resolution=bev_resolution, roi_size=bev_roi_size, make_images=make_preview,
                        source_info={'kind': 'fused_semantic_points', 'taxonomy': 'lidarseg16',
                                     'source_taxonomy': configuration['taxonomy'],
                                     'fusion': 'same original LiDAR point merged across cameras by nonzero class majority vote'})
                save_json(frame_dir / "summary.json", {"sample_token": views[0]["sample_token"], "lidar_token": lidar_token,
                          "cameras": [view["camera"] for view in views], "source_run_status": source_status,
                          "bev": bev_metadata, "ground": ground_metadata, **report})
                entry = {"stem": stem, "sample_token": views[0]["sample_token"], "lidar_token": lidar_token,
                         "cameras": [view["camera"] for view in views], "camera_count": len(views),
                         "bev": bev_metadata, **report["coverage"]}
                manifest.write(json.dumps(entry) + "\n")
                manifest.flush()
                state["processed_frames"] += 1
                print(f"[{frame_index + 1}/{len(selected)}] {stem}: points={len(xyz)}, covered={covered.sum()}, "
                      f"BEV={bev_metadata['width']}x{bev_metadata['height']}", flush=True)
        state["status"] = "completed"
    except BaseException as error:
        state.update({"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "error": str(error)})
        raise
    finally:
        state["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_json(output_dir / "run_config.json", state)
        if metrics is not None:
            metrics.save(output_dir / "evaluation")
    print(f"2D BEV results: {output_dir}", flush=True)
    return state


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    result.add_argument("--input-dir", type=Path, required=True, help="Finished SAM3 nuScenes output directory")
    result.add_argument("--output-dir", type=Path, help="New/empty directory; default <input-dir>/pointcloud_results")
    result.add_argument("--dataset-dir", type=Path, help="Override source run's dataset root")
    result.add_argument("--max-frames", type=int, default=0, help="0 exports all LiDAR frames recorded by the source run")
    result.add_argument("--bev-resolution", type=float, default=0.2, help="XY grid meters per pixel; does not change unique-point metrics")
    result.add_argument("--bev-density-cap", type=int, default=32, help="Winning-class point count at darkest shade; logarithmic density coloring")
    result.add_argument("--bev-roi-size", type=float, default=100.0, help="Centered square side in meters; default100/.2=500x500, 0 restores automatic extent")
    result.add_argument("--no-preview", action="store_true", help="Skip BEV PNGs; keep full point data, raster arrays and metrics")
    return result


def main() -> int:
    args = parser().parse_args()
    ensure_environment()
    export_run(args.input_dir, args.output_dir or args.input_dir / "pointcloud_results", args.dataset_dir,
               args.max_frames, args.bev_resolution, not args.no_preview, args.bev_density_cap, args.bev_roi_size)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)

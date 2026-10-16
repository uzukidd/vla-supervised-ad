"""Deterministic LiDAR XY majority rasters and offline Pillow previews.

Rasterization is display-only: metrics retain the original unique points.
Prediction and target vote independently; count ties choose the lowest class ID.
"""

from __future__ import annotations

import json
import math
from numbers import Integral, Real
from pathlib import Path

import numpy as np


MAX_CELLS = 16_000_000
BACKGROUND_COLOR = (255, 255, 255)
UNCOVERED_COLOR = (95, 105, 120)
UNLABELED_COLOR = (145, 145, 145)
IGNORED_COLOR = (145, 145, 145)
DENSITY_COLOR = (65, 65, 65)
CORRECT_COLOR = (45, 190, 95)
ERROR_COLOR = (235, 65, 65)
COVERED_COLOR = (56, 189, 248)


def _validate(xyz, prediction, target, covered, resolution):
    for name, value in (("xyz", xyz), ("prediction", prediction), ("covered", covered)):
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")
    if not (np.issubdtype(xyz.dtype, np.integer) or np.issubdtype(xyz.dtype, np.floating)):
        raise TypeError("xyz must have a real numeric dtype")
    if not np.all(np.isfinite(xyz)):
        raise ValueError("xyz must contain finite coordinates; no points are silently removed")
    for name, labels in (("prediction", prediction), ("target", target)):
        if labels is None and name == "target":
            continue
        if not isinstance(labels, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        if labels.shape != (len(xyz),):
            raise ValueError(f"{name} must have shape (N,) matching xyz")
        if labels.dtype != np.uint8:
            raise TypeError(f"{name} must have dtype uint8")
        if np.any(labels == 255):
            raise ValueError(f"{name} labels must be in [0, 254]; 255 marks empty cells")
    if covered.shape != (len(xyz),):
        raise ValueError("covered must have shape (N,) matching xyz")
    if covered.dtype != np.bool_:
        raise TypeError("covered must have dtype bool")
    if not isinstance(resolution, Real) or isinstance(resolution, (bool, np.bool_)):
        raise TypeError("resolution must be a finite positive real number")
    if not math.isfinite(float(resolution)) or resolution <= 0:
        raise ValueError("resolution must be finite and greater than zero")


def _validate_density_cap(cap):
    if not isinstance(cap, Integral) or isinstance(cap, (bool, np.bool_)):
        raise TypeError("density_cap must be an integer greater than or equal to 2")
    if cap < 2:
        raise ValueError("density_cap must be greater than or equal to 2")
    return int(cap)


def _density_metadata(cap):
    return {
        "cap": cap, "minimum_alpha": 0.25,
        "alpha_formula": "0.25 + 0.75 * log1p(min(n, cap)) / log1p(cap)",
        "rgb_formula": "round(255 * (1 - alpha) + base_rgb * alpha)",
        "empty_rgb": [255, 255, 255],
        "count_basis": "absolute winning class count; density uses total count; coverage uses covered count",
        "saturation": "n >= cap reaches base RGB; not a probability; no frame normalization",
    }


def _class_votes(flat_cells, labels, shape):
    """Reduce sparse (cell, class) counts without allocating H x W x classes."""
    encoded, counts = np.unique(flat_cells * 256 + labels, return_counts=True)
    vote_cells = (encoded // 256).astype(np.int64)
    vote_classes = (encoded % 256).astype(np.uint8)
    vote_counts = counts.astype(np.uint32)
    winners = np.full(shape, 255, dtype=np.uint8)
    winning_counts = np.zeros(shape, dtype=np.uint32)
    if len(encoded):
        order = np.lexsort((vote_classes, -counts, vote_cells))
        ordered_cells = vote_cells[order]
        first = np.r_[True, ordered_cells[1:] != ordered_cells[:-1]]
        selected = order[first]
        winners.flat[vote_cells[selected]] = vote_classes[selected]
        winning_counts.flat[vote_cells[selected]] = vote_counts[selected]
    return winners, winning_counts, vote_cells, vote_classes, vote_counts


def _fixed_bounds(bounds, resolution):
    if not isinstance(bounds, tuple) or len(bounds) != 4:
        raise TypeError("bounds must be a tuple (x_min, y_min, x_max, y_max)")
    if any(not isinstance(value, Real) or isinstance(value, (bool, np.bool_)) for value in bounds):
        raise TypeError("bounds must contain finite real numbers")
    bounds = tuple(float(value) for value in bounds)
    if not all(math.isfinite(value) for value in bounds):
        raise ValueError("bounds must contain finite real numbers")
    if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
        raise ValueError("bounds must be strictly ordered on both axes")
    sizes = []
    for low, high in ((bounds[0], bounds[2]), (bounds[1], bounds[3])):
        span = (high - low) / resolution
        if not math.isfinite(span):
            raise ValueError("BEV coordinate range is too large; increase resolution")
        size = round(span)
        if size < 1 or not math.isclose(span, size, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("each bounds range / resolution must be an integer (tolerance 1e-8)")
        sizes.append(size)
    return bounds, sizes[0], sizes[1]


def rasterize_bev(
    xyz: np.ndarray, prediction: np.ndarray, target: np.ndarray | None,
    covered: np.ndarray, resolution: float = 0.2,
    bounds: tuple[float, float, float, float] | None = None,
) -> dict:
    """Vote points into aligned HxW grids without changing inputs.

    Fixed bounds are (x_min, y_min, x_max, y_max), lower-inclusive and
    upper-exclusive. With bounds=None the original automatic extent is used,
    including the maximum-coordinate cell and a single cell for empty clouds.
    Columns increase with X; rows decrease with Y. Coordinates are caller-owned:
    no ego-frame transformation or height filtering is assumed here.
    """
    _validate(xyz, prediction, target, covered, resolution)
    resolution = float(resolution)
    input_count = len(xyz)
    fixed = bounds is not None
    if fixed:
        bounds, width, height = _fixed_bounds(bounds, resolution)
        inside = ((xyz[:, 0] >= bounds[0]) & (xyz[:, 0] < bounds[2])
                  & (xyz[:, 1] >= bounds[1]) & (xyz[:, 1] < bounds[3]))
        xyz = xyz[inside]
        prediction = prediction[inside]
        target = None if target is None else target[inside]
        covered = covered[inside]
    count = len(xyz)
    if fixed:
        # Subtraction/division can put a mathematical grid edge a few ULPs
        # below its integer index (e.g. (0.4 + 50) / 0.2). Snap only floating
        # roundoff before floor; the strict ROI mask still excludes upper edges.
        offsets = (xyz[:, :2].astype(np.float64) - np.asarray(bounds[:2])) / resolution
        nearest = np.rint(offsets)
        tolerance = 8 * np.finfo(np.float64).eps * np.maximum(1.0, np.abs(offsets))
        offsets = np.where(np.abs(offsets - nearest) <= tolerance, nearest, offsets)
        columns = np.clip(np.floor(offsets[:, 0]), 0, width - 1).astype(np.int64)
        y_cells = np.clip(np.floor(offsets[:, 1]), 0, height - 1).astype(np.int64)
        rows = height - 1 - y_cells
    elif count:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            cells = np.floor(xyz[:, :2].astype(np.float64) / resolution)
        if not np.all(np.isfinite(cells)):
            raise ValueError("BEV coordinate range is too large; increase resolution")
        x_low, y_low = (int(value) for value in cells.min(axis=0))
        x_high, y_high = (int(value) for value in cells.max(axis=0))
        width, height = x_high - x_low + 1, y_high - y_low + 1
    else:
        x_low = y_low = x_high = y_high = 0
        width = height = 1
    if width * height > MAX_CELLS:
        raise ValueError(f"BEV grid requires {width * height:,} cells (limit {MAX_CELLS:,}); "
                         "increase resolution rather than cropping or sampling points")
    if not fixed:
        try:
            bounds = (x_low * resolution, y_low * resolution,
                      (x_high + 1) * resolution, (y_high + 1) * resolution)
        except OverflowError as error:
            raise ValueError("BEV bounds cannot be represented; increase resolution") from error
        if (not all(math.isfinite(value) for value in bounds)
                or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]):
            raise ValueError("BEV bounds cannot be represented accurately; increase resolution")
    valid_extent = ({"x_min_m": float(xyz[:, 0].min()), "x_max_m": float(xyz[:, 0].max()),
                     "y_min_m": float(xyz[:, 1].min()), "y_max_m": float(xyz[:, 1].max())}
                    if count else None)
    shape = (height, width)
    flat_cells = np.empty(0, dtype=np.int64)
    if count:
        if not fixed:
            columns = (cells[:, 0] - x_low).astype(np.int64)
            rows = (y_high - cells[:, 1]).astype(np.int64)
        flat_cells = rows * width + columns
    point_count = np.bincount(flat_cells, minlength=height * width).reshape(shape).astype(np.uint32)
    covered_count = np.bincount(flat_cells[covered], minlength=height * width).reshape(shape).astype(np.uint32)
    occupied = point_count > 0
    raster_prediction, prediction_count, pred_cells, pred_classes, pred_counts = _class_votes(
        flat_cells, prediction, shape)
    target_arrays = (None,) * 5 if target is None else _class_votes(flat_cells, target, shape)
    raster_target, target_count, target_cells, target_classes, target_counts = target_arrays
    occupied_cells = int(np.count_nonzero(occupied))
    metadata = {
        "projection": "LiDAR XY BEV", "resolution_m_per_pixel": resolution,
        "x_min_m": bounds[0], "y_min_m": bounds[1],
        "x_max_m": bounds[2], "y_max_m": bounds[3],
        "width": width, "height": height, "point_count": count,
        "roi_input_point_count": input_count, "roi_outside_point_count": input_count - count,
        "grid_mode": "fixed_roi" if fixed else "automatic_extent",
        "occupied_cells": occupied_cells, "collision_points": count - occupied_cells,
        "rule": "majority_count_then_lowest_class_id", "tie_break": "lowest_class_id",
        "zero_class_policy": "class 0 participates as unassigned/ignored",
        "orientation": "+x right, +y up", "valid_extent_m": valid_extent,
        "density_scaling": _density_metadata(32),
        "error_basis": "cell majority comparison; covered cells with target majority != 0; not a point metric",
    }
    return {
        "occupied": occupied, "prediction": raster_prediction, "target": raster_target,
        "point_count": point_count, "prediction_count": prediction_count, "target_count": target_count,
        "covered_count": covered_count, "covered": covered_count > 0,
        "prediction_vote_cells": pred_cells, "prediction_vote_classes": pred_classes,
        "prediction_vote_counts": pred_counts, "target_vote_cells": target_cells,
        "target_vote_classes": target_classes, "target_vote_counts": target_counts,
        "metadata": metadata,
    }


def density_tint(base_colors, counts, cap=32):
    """Blend RGB bases with white using absolute counts; zero is pure white.

    Counts broadcast against the leading dimensions of base_colors. A count
    at or above cap reaches the base color. Inputs are never modified.
    """
    cap = _validate_density_cap(cap)
    base = np.asarray(base_colors)
    counts = np.asarray(counts)
    if base.ndim < 1 or base.shape[-1] != 3:
        raise ValueError("base_colors must have a final RGB dimension of length 3")
    if (not np.issubdtype(base.dtype, np.number) or np.iscomplexobj(base)
            or not np.all(np.isfinite(base)) or np.any(base < 0) or np.any(base > 255)):
        raise ValueError("base_colors must contain real RGB values in [0, 255]")
    if not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
        raise ValueError("counts must contain nonnegative integers")
    alpha = 0.25 + 0.75 * np.log1p(np.minimum(counts, cap)) / math.log1p(cap)
    alpha = np.where(counts == 0, 0.0, alpha)[..., None]
    return np.rint(255 * (1 - alpha) + base.astype(np.float64) * alpha).astype(np.uint8)


def _json(value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Not JSON serializable: {type(item).__name__}")
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=convert)


def _colors(raster, palette, density_cap=32):
    occupied, covered = raster["occupied"], raster["covered"]
    prediction, target = raster["prediction"], raster["target"]
    prediction_base = palette[prediction].copy()
    prediction_base[occupied & (prediction == 0) & covered] = UNLABELED_COLOR
    prediction_base[occupied & (prediction == 0) & ~covered] = UNCOVERED_COLOR
    predicted = density_tint(prediction_base, raster["prediction_count"], density_cap)
    coverage_base = np.full((*occupied.shape, 3), UNCOVERED_COLOR, dtype=np.uint8)
    coverage_base[covered] = COVERED_COLOR
    coverage_counts = np.where(covered, raster["covered_count"], raster["point_count"])
    coverage = density_tint(coverage_base, coverage_counts, density_cap)
    colors = {"prediction": predicted, "coverage": coverage,
              "density": density_tint(DENSITY_COLOR, raster["point_count"], density_cap)}
    if target is not None:
        target_base = palette[target].copy()
        target_base[occupied & (target == 0)] = IGNORED_COLOR
        gt = density_tint(target_base, raster["target_count"], density_cap)
        error_base = np.full((*occupied.shape, 3), IGNORED_COLOR, dtype=np.uint8)
        valid = occupied & covered & (target != 0)
        error_base[valid & (prediction == target)] = CORRECT_COLOR
        error_base[valid & (prediction != target)] = ERROR_COLOR
        errors = density_tint(error_base, raster["prediction_count"], density_cap)
        colors.update(ground_truth=gt, errors=errors)
    return colors


def _summary_lines(summary):
    lines = []

    def visit(value, path=()):
        for key, item in value.items():
            parts = (*path, str(key))
            if isinstance(item, dict):
                visit(item, parts)
            elif (key in {"mean_iou", "miou", "mIoU", "geometric_coverage", "coverage",
                          "coverage_ratio", "coverage_fraction"}
                  or "coverage" in str(key).lower()):
                if isinstance(item, (int, float)) and not isinstance(item, bool):
                    text = f"{item:.2%}" if 0 <= item <= 1 else str(item)
                elif item is None:
                    text = "n/a"
                else:
                    continue
                label = " / ".join(parts).replace("mean_iou", "mIoU").replace("_", " ")
                lines.append(f"Point metric | {label}: {text}")
    visit(summary)
    return lines


def _comparison(colors, metadata, class_names, palette, summary):
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default()
    text_color = (30, 40, 55)
    modes = (["prediction", "ground_truth", "errors", "coverage"]
             if "ground_truth" in colors else ["prediction", "coverage"])
    titles = {"prediction": "Prediction (cell majority)", "ground_truth": "Ground truth (independent cell majority)",
              "errors": "Cell majority comparison (covered, GT nonzero)", "coverage": "Camera coverage (any covered point)"}
    panel_size = 640
    panel_rows = 2 if len(modes) == 4 else 1
    legend = [(BACKGROUND_COLOR, "Empty cell: 0 points"), (UNCOVERED_COLOR, "Uncovered prediction 0"),
              (UNLABELED_COLOR, "Covered prediction 0"), (IGNORED_COLOR, "GT 0 / unevaluated"),
              (COVERED_COLOR, "Covered"), (CORRECT_COLOR, "Correct cell majority"), (ERROR_COLOR, "Wrong cell majority")]
    legend.extend((tuple(int(c) for c in palette[key]), f"{key}: {class_names[key]}")
                  for key in sorted(class_names) if key != 0)
    cap = metadata["density_scaling"]["cap"]
    density_rows = [("Total points", DENSITY_COLOR), ("Covered points", COVERED_COLOR),
                    ("Uncovered / pred 0", UNCOVERED_COLOR), ("Class 0 / GT 0", UNLABELED_COLOR)]
    density_rows.extend((f"{key}: {class_names[key]}", palette[key]) for key in sorted(class_names) if key != 0)
    header_lines = [
        "LiDAR XY BEV | +X right, +Y up | point metrics use original unique points",
        (f"X [{metadata['x_min_m']:.3f}, {metadata['x_max_m']:.3f}) m; "
         f"Y [{metadata['y_min_m']:.3f}, {metadata['y_max_m']:.3f}) m; "
         f"resolution {metadata['resolution_m_per_pixel']:g} m/pixel"),
        (f"Points {metadata['point_count']:,}; occupied cells {metadata['occupied_cells']:,}; "
         f"collisions {metadata['collision_points']:,}; majority count, ties lowest class ID; class 0 votes"),
        f"Color depth: absolute winning class count; log scale; cap {cap} points (saturation, not probability)",
        * _summary_lines(summary),
    ]
    header_height = 16 * len(header_lines) + 16
    legend_height = 24 * math.ceil(len(legend) / 3) + 16
    footer_height = legend_height + 44 + 24 * len(density_rows)
    result = Image.new("RGB", (2 * panel_size, header_height + panel_rows * panel_size + footer_height),
                       BACKGROUND_COLOR)
    draw = ImageDraw.Draw(result)
    for row, text in enumerate(header_lines):
        draw.text((12, 8 + row * 16), text, font=font, fill=text_color)
    for index, mode in enumerate(modes):
        left = (index % 2) * panel_size
        top = header_height + (index // 2) * panel_size
        draw.text((left + 16, top + 8), titles[mode], font=font, fill=text_color)
        image = Image.fromarray(colors[mode])
        scale = min(590 / image.width, 574 / image.height)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(size, resample=Image.Resampling.NEAREST)
        image_left, image_top = left + 38 + (590 - size[0]) // 2, top + 30 + (574 - size[1]) // 2
        result.paste(image, (image_left, image_top))
        draw.line((left + 18, top + 610, left + 18, top + 580), fill=text_color, width=2)
        draw.polygon(((left + 18, top + 576), (left + 14, top + 583), (left + 22, top + 583)), fill=text_color)
        draw.text((left + 4, top + 562), "+Y", font=font, fill=text_color)
        draw.line((left + 38, top + 618, left + 85, top + 618), fill=text_color, width=2)
        draw.polygon(((left + 89, top + 618), (left + 82, top + 614), (left + 82, top + 622)), fill=text_color)
        draw.text((left + 96, top + 612), "+X", font=font, fill=text_color)
    footer_top = header_height + panel_rows * panel_size
    for index, (color, label) in enumerate(legend):
        x, y = 12 + (index % 3) * 420, footer_top + 8 + (index // 3) * 24
        draw.rectangle((x, y, x + 14, y + 14), fill=color)
        draw.text((x + 22, y + 2), label, font=font, fill=text_color)
    density_top = footer_top + legend_height
    draw.text((12, density_top), f"Density examples (points per cell): cap = {cap}; counts >= cap have the same saturated color", font=font, fill=text_color)
    sample_counts = np.array([1, 2, 4, 8, 16, 32], dtype=np.uint32)
    for column, count in enumerate(sample_counts):
        draw.text((280 + column * 150, density_top + 20), f"{count} points" + (" (sat.)" if count >= cap else ""), font=font, fill=text_color)
    for row, (label, base) in enumerate(density_rows):
        y = density_top + 42 + row * 24
        draw.text((12, y + 2), label, font=font, fill=text_color)
        samples = density_tint(base, sample_counts, cap)
        for column, color in enumerate(samples):
            draw.rectangle((280 + column * 150, y, 340 + column * 150, y + 16), fill=tuple(color))
    return result


def save_bev_images(
    output_dir: Path, xyz: np.ndarray, prediction: np.ndarray, target: np.ndarray | None,
    covered: np.ndarray, class_names: dict[int, str], palette: np.ndarray, summary: dict,
    resolution: float = 0.2, make_images: bool = True, density_cap: int = 32,
    bounds: tuple[float, float, float, float] | None = None,
) -> dict:
    """Save dense rasters, sparse votes and metadata; optionally native RGB PNGs.

    Comparison uses the same grid for every panel and nearest-neighbor display
    scaling. Caller arrays and point metrics are never changed or recomputed.
    """
    density_cap = _validate_density_cap(density_cap)
    if not isinstance(palette, np.ndarray) or palette.dtype != np.uint8:
        raise TypeError("palette must be a numpy.ndarray with dtype uint8")
    if palette.shape != (256, 3):
        raise ValueError("palette must have shape (256, 3)")
    if not isinstance(class_names, dict):
        raise TypeError("class_names must be a dict")
    for key, name in class_names.items():
        if (not isinstance(key, (int, np.integer)) or isinstance(key, (bool, np.bool_))
                or not 0 <= key <= 254):
            raise ValueError("class IDs must be integers in [0, 254]")
        if not isinstance(name, str):
            raise TypeError("class names must be strings")
    if not isinstance(summary, dict):
        raise TypeError("summary must be a dict")
    if not isinstance(make_images, (bool, np.bool_)):
        raise TypeError("make_images must be bool")
    normalized_summary = json.loads(_json(summary))
    raster = rasterize_bev(xyz, prediction, target, covered, resolution, bounds)
    metadata = {**raster["metadata"], "class_names": {int(k): v for k, v in class_names.items()},
                "density_scaling": _density_metadata(density_cap),
                "summary": normalized_summary,
                "metrics_basis": "original unique points; BEV cells are display-only"}
    encoded_metadata = _json(metadata)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in raster.items() if key != "metadata" and value is not None}
    np.savez_compressed(output_dir / "bev_raster.npz", **arrays)
    (output_dir / "bev_metadata.json").write_text(encoded_metadata + "\n", encoding="utf-8")
    if make_images:
        from PIL import Image
        colors = _colors(raster, palette, density_cap)
        for mode, rgb in colors.items():
            Image.fromarray(rgb).save(output_dir / f"bev_{mode}.png")
        _comparison(colors, metadata, class_names, palette, normalized_summary).save(
            output_dir / "bev_comparison.png")
    return metadata

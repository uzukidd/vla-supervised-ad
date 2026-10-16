"""Camera-local instance convex hulls and deterministic, fixed-ROI BEV previews.

Geometry uses every assigned point inside the ROI, without outlier removal,
clustering, buffering, ground truth, or merging instances across cameras.
"""

from __future__ import annotations

import colorsys
import json
import math
from numbers import Integral, Real
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw


WARNINGS = [
    "Convex hull fills unobserved interior and can bridge gaps.",
    "Only camera-local instance IDs; no merging across cameras.",
    "Stuff masks are semantic regions, not object instances.",
    "Overlapping projections are not ground truth; no evaluation is fabricated.",
]


def _coordinates(value, dimensions, name):
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != dimensions:
        raise ValueError(f"{name} must have shape (N, {dimensions})")
    if not (np.issubdtype(array.dtype, np.integer)
            or np.issubdtype(array.dtype, np.floating)):
        raise TypeError(f"{name} must contain real numeric coordinates")
    result = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite coordinates")
    return result


def _cross(origin, left, right):
    a, b = left - origin, right - origin
    return a[0] * b[1] - a[1] * b[0]


def convex_hull_xy(points: np.ndarray) -> np.ndarray:
    """Return the CCW monotone-chain hull, starting at the smallest XY pair.

    Duplicate vertices and collinear interior vertices are removed. Empty,
    singleton and collinear inputs produce zero, one and two vertices.
    """
    points = _coordinates(points, 2, "points")
    ordered = np.unique(points, axis=0)
    if len(ordered) <= 1:
        return ordered.copy()
    lower, upper = [], []
    for point in ordered:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in ordered[::-1]:
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.array(lower[:-1] + upper[:-1], dtype=np.float64).reshape(-1, 2)


def polygon_area(vertices: np.ndarray) -> float:
    """Return unsigned square-metre area; degenerate hulls have zero area."""
    vertices = _coordinates(vertices, 2, "vertices")
    if len(vertices) < 3:
        return 0.0
    shifted = vertices - vertices[0]
    return float(abs(np.sum(shifted[:, 0] * np.roll(shifted[:, 1], -1)
                            - shifted[:, 1] * np.roll(shifted[:, 0], -1))) * 0.5)


def points_in_convex_polygon(points, vertices, tolerance=1e-7) -> np.ndarray:
    """Test closed hull containment with a boundary-distance tolerance in metres.

    Polygon winding may be clockwise or counterclockwise. A singleton uses
    Euclidean distance, and a segment uses distance to its clamped projection.
    """
    points = _coordinates(points, 2, "points")
    vertices = _coordinates(vertices, 2, "vertices")
    if not isinstance(tolerance, Real) or isinstance(tolerance, (bool, np.bool_)):
        raise TypeError("tolerance must be a finite nonnegative real number")
    if not math.isfinite(float(tolerance)) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if not len(vertices):
        return np.zeros(len(points), dtype=bool)
    if len(vertices) == 1:
        return np.linalg.norm(points - vertices[0], axis=1) <= tolerance
    if len(vertices) == 2 or polygon_area(vertices) == 0:
        endpoints = convex_hull_xy(vertices)
        if len(endpoints) == 1:
            return np.linalg.norm(points - endpoints[0], axis=1) <= tolerance
        vector = endpoints[-1] - endpoints[0]
        fraction = np.clip(((points - endpoints[0]) @ vector) / (vector @ vector), 0, 1)
        nearest = endpoints[0] + fraction[:, None] * vector
        return np.linalg.norm(points - nearest, axis=1) <= tolerance
    shifted = vertices - vertices[0]
    signed = np.sum(shifted[:, 0] * np.roll(shifted[:, 1], -1)
                    - shifted[:, 1] * np.roll(shifted[:, 0], -1))
    orientation = 1 if signed > 0 else -1
    inside = np.ones(len(points), dtype=bool)
    for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
        edge = end - start
        length = np.linalg.norm(edge)
        cross = edge[0] * (points[:, 1] - start[1]) - edge[1] * (points[:, 0] - start[0])
        inside &= orientation * cross >= -float(tolerance) * length
    return inside


def instance_rgb(global_instance_id: int) -> tuple[int, int, int]:
    """Generate the same RGB for an ID in camera and BEV views, beyond 255 IDs."""
    if (not isinstance(global_instance_id, Integral)
            or isinstance(global_instance_id, (bool, np.bool_))):
        raise TypeError("global_instance_id must be a positive integer")
    if global_instance_id < 1:
        raise ValueError("global_instance_id must be a positive integer")
    identifier = int(global_instance_id)
    hue = (identifier * 0.6180339887498949) % 1.0
    saturation = 0.65 + 0.25 * ((identifier * 0.4142135623730951) % 1.0)
    value = 0.72 + 0.23 * ((identifier * 0.7320508075688772) % 1.0)
    return tuple(round(channel * 255) for channel in colorsys.hsv_to_rgb(hue, saturation, value))


def _positive_real(value, name):
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive finite real number")
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be a positive finite real number")
    return float(value)


def save_hull_bev(
    output: Path, xyz: np.ndarray, instances: list[dict], class_names: dict[int, str],
    resolution: float = 0.2, roi_size: float = 100,
) -> dict:
    """Save metric hulls, four previews and camera views; return their JSON data.

    point_count is the unique assigned input count before ROI filtering;
    support_count is the unique assigned count inside the lower-inclusive,
    upper-exclusive square ROI. All indices refer to the original xyz array.
    Assignments must be exclusive within each camera. Different cameras may
    observe the same original point; repeats within one instance are deduplicated.
    Validation completes before files are saved.
    """
    xyz = _coordinates(xyz, 3, "xyz")
    resolution = _positive_real(resolution, "resolution")
    roi_size = _positive_real(roi_size, "roi_size")
    cells = roi_size / resolution
    if not math.isfinite(cells) or not math.isclose(cells, round(cells), rel_tol=0, abs_tol=1e-8):
        raise ValueError("roi_size / resolution must be an integer")
    size = round(cells)
    if size < 1 or size * size > 16_000_000:
        raise ValueError("BEV grid must contain between 1 and 16,000,000 pixels")
    if not isinstance(instances, list):
        raise TypeError("instances must be a list of dictionaries")
    if not isinstance(class_names, dict):
        raise TypeError("class_names must be a dictionary")
    half = roi_size / 2
    inside = np.all((xyz[:, :2] >= -half) & (xyz[:, :2] < half), axis=1)
    results, identifiers, claimed = [], set(), set()
    claimed_by_camera = {}
    for instance in instances:
        if not isinstance(instance, dict):
            raise TypeError("each instance must be a dictionary")
        identifier = instance["global_instance_id"]
        rgb = instance_rgb(identifier)
        identifier = int(identifier)
        if identifier in identifiers:
            raise ValueError("global_instance_id must be unique")
        identifiers.add(identifier)
        camera = instance["camera"]
        if (not isinstance(camera, str) or not camera or camera in (".", "..")
                or any(character in camera for character in '/\\<>:"|?*')):
            raise ValueError("camera must be a nonempty safe filename component")
        class_id = instance["class_id"]
        if (not isinstance(class_id, Integral) or isinstance(class_id, (bool, np.bool_))
                or not 1 <= class_id <= 16):
            raise ValueError("class_id must be an integer in [1, 16]")
        class_name = instance["class_name"]
        if not isinstance(class_name, str):
            raise TypeError("class_name must be a string")
        score = instance["score"]
        if not isinstance(score, Real) or isinstance(score, (bool, np.bool_)):
            raise TypeError("score must be finite and numeric")
        if not math.isfinite(float(score)):
            raise ValueError("score must be finite")
        indices = instance["point_indices"]
        if not isinstance(indices, (list, tuple, np.ndarray)):
            raise TypeError("point_indices must be a one-dimensional integer sequence")
        if isinstance(indices, np.ndarray) and indices.ndim != 1:
            raise ValueError("point_indices must be one-dimensional")
        if any(not isinstance(index, Integral) or isinstance(index, (bool, np.bool_)) for index in indices):
            raise TypeError("point_indices must contain integers")
        indices = sorted({int(index) for index in indices})
        if any(index < 0 or index >= len(xyz) for index in indices):
            raise ValueError("point_indices contains an index outside xyz")
        camera_claimed = claimed_by_camera.setdefault(camera, set())
        if camera_claimed.intersection(indices):
            raise ValueError("point_indices must be exclusively assigned within each camera after conflict resolution")
        camera_claimed.update(indices)
        claimed.update(indices)
        roi_indices = [index for index in indices if inside[index]]
        vertices = convex_hull_xy(xyz[roi_indices, :2])
        if not np.all(points_in_convex_polygon(xyz[roi_indices, :2], vertices)):
            raise ValueError("convex hull failed to contain all assigned ROI points")
        geometry = ("empty", "point", "segment")[len(vertices)] if len(vertices) < 3 else "polygon"
        result = dict(instance)
        result.update(global_instance_id=identifier, camera=camera, class_id=int(class_id),
                      class_name=class_name, score=float(score), point_indices=indices,
                      roi_point_indices=roi_indices, hull_xy=vertices.tolist(),
                      area_m2=polygon_area(vertices), geometry_type=geometry,
                      point_count=len(indices), support_count=len(roi_indices),
                      outside_roi_point_count=len(indices) - len(roi_indices), rgb=list(rgb),
                      is_stuff=int(class_id) >= 11)
        results.append(result)
    results.sort(key=lambda item: item["global_instance_id"])
    cameras = sorted({item["camera"] for item in results})
    summary = {
        "num_instances": len(results),
        "num_polygons": sum(item["geometry_type"] == "polygon" for item in results),
        "num_degenerate": sum(item["geometry_type"] in ("point", "segment") for item in results),
        "num_empty": sum(item["geometry_type"] == "empty" for item in results),
        "num_points": sum(item["support_count"] for item in results),
        "num_unique_points": sum(bool(inside[index]) for index in claimed),
        "per_camera_count": {camera: sum(item["camera"] == camera for item in results) for camera in cameras},
    }
    metadata = {
        "rule": "minimum_convex_polygon_covering_all_assigned_roi_points",
        "hull_method": "monotone_chain; exact input XY vertices; CCW; no outlier trimming",
        "roi_size_m": roi_size, "resolution_m_per_pixel": resolution,
        "width": size, "height": size, "bounds_xy_m": [-half, -half, half, half],
        "x_min_m": -half, "y_min_m": -half, "x_max_m": half, "y_max_m": half,
        "roi_boundary": "lower-inclusive, upper-exclusive on both axes",
        "orientation": "+x right, +y up", "coordinate_frame": "input xyz frame",
        "input_point_count": len(xyz), "roi_point_count": int(inside.sum()),
        "outside_roi_point_count": int((~inside).sum()),
        "assigned_point_count": len(claimed), "assigned_roi_point_count": summary["num_unique_points"],
        "assigned_point_observation_count": sum(item["point_count"] for item in results),
        "assigned_roi_point_observation_count": summary["num_points"],
        "assignment_policy": "exclusive within each camera; shared original points across cameras retained without merging",
        "point_count_definition": "instance point_count: unique assigned input before ROI; support_count: inside ROI; metadata assigned counts: unique original points; observation counts: camera-local repetitions included",
        "foreground_class_ids": list(range(1, 11)), "stuff_class_ids": list(range(11, 17)),
        "class_names": {str(key): str(value) for key, value in sorted(class_names.items())},
        "polygon_fill_alpha": 0.18, "outline_width_px": 2,
        "draw_order": "ascending score then global_instance_id; higher score is drawn last",
        "color_rule": "deterministic HSV from global_instance_id, shared with camera views",
        "warnings": WARNINGS.copy(), "summary": summary,
    }

    def pixels(points):
        coordinates = np.floor((points + half) / resolution)
        coordinates = np.clip(coordinates, 0, size - 1).astype(np.int64)
        coordinates[:, 1] = size - 1 - coordinates[:, 1]
        return [tuple(int(value) for value in pair) for pair in coordinates]

    raw = Image.new("RGB", (size, size), "white")
    ImageDraw.Draw(raw).point(pixels(xyz[inside, :2]), fill=(175, 175, 175))

    def render(selected, hulls):
        image = raw.copy()
        ordered = sorted(selected, key=lambda item: (item["score"], item["global_instance_id"]))
        for item in ordered:
            if not item["support_count"]:
                continue
            color = tuple(item["rgb"])
            if hulls:
                draw = ImageDraw.Draw(image, "RGBA")
                vertices = pixels(np.asarray(item["hull_xy"], dtype=np.float64))
                dark = tuple(round(channel * 0.65) for channel in color)
                if item["geometry_type"] == "polygon":
                    draw.polygon(vertices, fill=(*color, round(255 * 0.18)), outline=(*dark, 255), width=2)
                elif item["geometry_type"] == "segment":
                    draw.line(vertices, fill=(*dark, 255), width=2)
        draw = ImageDraw.Draw(image)
        for item in ordered:
            if item["support_count"]:
                draw.point(pixels(xyz[item["roi_point_indices"], :2]), fill=tuple(item["rgb"]))
        if hulls:
            for item in ordered:
                if not item["support_count"]:
                    continue
                anchor = np.asarray(item["hull_xy"], dtype=np.float64).mean(axis=0, keepdims=True)
                x, y = pixels(anchor)[0]
                label = f"#{item['global_instance_id']} {item['class_name'][:16]}"
                label = label.encode("ascii", "replace").decode("ascii")
                box = draw.textbbox((0, 0), label)
                x = max(0, min(x + 3, size - (box[2] - box[0]) - 2))
                y = max(0, min(y + 3, size - (box[3] - box[1]) - 3))
                draw.text((x, y), label, fill=tuple(round(c * 0.55) for c in item["rgb"]))
        return image

    points_image = render(results, False)
    foreground_image = render([item for item in results if item["class_id"] <= 10], True)
    all_image = render(results, True)
    titles = ["Raw ROI LiDAR points", "Assigned points: instance ID colors",
              "Foreground instance hulls (classes 1-10)", "All regions (11-16: semantic stuff masks)"]
    legend_lines = [
        f"ROI {roi_size:g} x {roi_size:g} m | {resolution:g} m/pixel | +x right, +y up",
        f"Polygons: {summary['num_polygons']} | Point observations: {summary['num_points']} | "
        f"Unique points: {summary['num_unique_points']} | Degenerate: {summary['num_degenerate']} | Empty: {summary['num_empty']}",
        "Camera region counts: " + (", ".join(f"{key}: {value}" for key, value in summary["per_camera_count"].items()) or "none"),
        "Legend: gray = raw LiDAR; colors = instance IDs; light fill = convex interior; dark = boundary.",
        *WARNINGS,
    ]
    wrapped = [line for text in legend_lines for line in textwrap.wrap(text, width=max(30, (2 * size - 20) // 7))]
    title_height = 26
    comparison = Image.new("RGB", (2 * size, 2 * (size + title_height) + 18 * len(wrapped) + 16), "white")
    draw = ImageDraw.Draw(comparison)
    for index, (title, image) in enumerate(zip(titles, (raw, points_image, foreground_image, all_image))):
        left, top = (index % 2) * size, (index // 2) * (size + title_height)
        draw.text((left + 8, top + 7), title, fill=(30, 30, 30))
        comparison.paste(image, (left, top + title_height))
    for index, line in enumerate(wrapped):
        draw.text((8, 2 * (size + title_height) + 8 + 18 * index), line, fill=(45, 45, 45))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    points_image.save(output / "bev_instance_points.png")
    foreground_image.save(output / "bev_instance_hulls.png")
    all_image.save(output / "bev_all_regions.png")
    comparison.save(output / "bev_instance_comparison.png")
    if cameras:
        (output / "per_camera").mkdir(exist_ok=True)
        for camera in cameras:
            render([item for item in results if item["camera"] == camera], True).save(
                output / "per_camera" / f"{camera}_hulls.png")
    data = {"metadata": metadata, "instances": results}
    (output / "instance_hulls.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return data

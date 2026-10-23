#!/usr/bin/env python
"""Merge camera-local ground masks into bounded, triangulated semantic BEV regions."""
from __future__ import annotations

import argparse
from datetime import datetime
from html import escape
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

from shared.python.nuscenes_data import RAW_TO_COARSE
from shared.python.runtime import ROOT, ensure_environment, save_json
from shared.python.sam3 import make_palette

GROUND_IDS = (11, 12, 13, 14)


def fuse_semantic_points(point_count: int, instances: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Deduplicate original points across cameras, resolving classes by score."""
    labels = np.zeros(point_count, dtype=np.uint8)
    scores = np.full(point_count, -np.inf, dtype=np.float64)
    for row in sorted(instances, key=lambda row: row['global_instance_id']):
        indices = np.unique(np.asarray(row['point_indices'], dtype=np.int64))
        if np.any(indices < 0) or np.any(indices >= point_count) or not 1 <= int(row['class_id']) <= 16:
            raise ValueError('Instance class or original point index is invalid')
        score = float(row['score'])
        if not np.isfinite(score):
            raise ValueError('Instance score must be finite')
        update = score > scores[indices]
        labels[indices[update]] = int(row['class_id'])
        scores[indices[update]] = score
    return labels, np.isfinite(scores)


def ground_planes(xyz: np.ndarray, labels: np.ndarray, resolution: float = .2, roi_size: float = 100.,
                  max_edge: float = 6., *, ground_ids=GROUND_IDS, num_classes: int = 17) -> dict:
    """Fill bounded per-class triangles from finite Nx3 points and aligned uint8 labels.

    Labels must be below ``num_classes`` (including unassigned 0, at most 255).
    ``ground_ids`` must contain unique integer IDs in 1..num_classes-1. IDs are
    preserved and sorted so equal votes and nearest-support distances prefer the
    lowest ID. Other assigned classes veto filling; unassigned 0 does not.
    Defaults retain the existing coarse16 ground classes and behavior.
    """
    if (isinstance(num_classes, (bool, np.bool_)) or not isinstance(num_classes, (int, np.integer))
            or not 2 <= num_classes <= 255):
        raise ValueError('num_classes must be an integer in [2, 255], including unassigned 0')
    num_classes = int(num_classes)
    try:
        ground_ids = tuple(ground_ids)
    except TypeError as error:
        raise ValueError('ground_ids must contain unique integer IDs in 1..num_classes-1') from error
    if (not ground_ids or any(isinstance(cid, (bool, np.bool_)) or not isinstance(cid, (int, np.integer))
                              or not 1 <= cid < num_classes for cid in ground_ids)
            or len(set(ground_ids)) != len(ground_ids)):
        raise ValueError('ground_ids must contain unique integer IDs in 1..num_classes-1')
    ground_ids = tuple(sorted(int(cid) for cid in ground_ids))
    if (not isinstance(xyz, np.ndarray) or not isinstance(labels, np.ndarray)
            or xyz.ndim != 2 or xyz.shape[1] != 3 or labels.shape != (len(xyz),)
            or labels.dtype != np.uint8 or np.any(labels >= num_classes) or not np.isfinite(xyz).all()):
        raise ValueError('Expected finite Nx3 xyz and aligned uint8 semantic labels below num_classes')
    if not all(np.isfinite(x) and x > 0 for x in (resolution, roi_size, max_edge)):
        raise ValueError('Resolution, ROI and maximum edge must be finite and positive')
    n = round(roi_size / resolution)
    if n < 1 or n*n > 16000000 or not np.isclose(roi_size / resolution, n, rtol=0, atol=1e-8):
        raise ValueError('ROI/resolution must be integral and grid no larger than 16 million cells')
    half = roi_size / 2
    inside = ((xyz[:, :2] >= -half) & (xyz[:, :2] < half)).all(1)
    points, values = xyz[inside], labels[inside]
    xy = np.floor((points[:, :2].astype(float) + half) / resolution).astype(int)
    xy = np.clip(xy, 0, n-1)
    cols, rows = xy[:, 0], n-1-xy[:, 1]
    flat = rows*n+cols
    observed = np.zeros((n,n), dtype=np.uint8)
    votes = np.zeros((num_classes,n,n), dtype=np.uint32)
    for cid in range(1,num_classes):
        votes[cid] = np.bincount(flat[values==cid], minlength=n*n).reshape(n,n)
    ground_support = np.stack([votes[cid] for cid in ground_ids])
    support_count = ground_support.sum(0)
    observed[support_count > 0] = np.asarray(ground_ids)[ground_support.argmax(0)[support_count > 0]]
    blocked = votes[[cid for cid in range(1,num_classes) if cid not in ground_ids]].sum(0) > 0
    # Ground labels at occupied cells are preserved; other semantic evidence is a veto.
    observed[blocked] = 0
    candidate_masks, distance_maps, triangle_records = [], [], []
    for cid in ground_ids:
        support = (observed == cid)
        pixel_points = np.column_stack(np.nonzero(support)[::-1]).astype(np.float32)
        candidate = support.astype(np.uint8)
        accepted = []
        if len(pixel_points) >= 3:
            subdiv = cv2.Subdiv2D((0,0,n,n))
            for x,y in pixel_points:
                subdiv.insert((float(x)+.5, float(y)+.5))
            for triangle in np.asarray(subdiv.getTriangleList(), dtype=np.float32).reshape(-1,3,2):
                if not ((triangle >= 0).all() and (triangle < n).all()):
                    continue
                edges = np.linalg.norm(triangle - np.roll(triangle,1,axis=0),axis=1)*resolution
                if edges.max() > max_edge + 1e-6:
                    continue
                vertices = np.floor(triangle).astype(np.int32)
                # Confirm triangle vertices are actual observed cells, not artificial boundary vertices.
                if not support[vertices[:,1],vertices[:,0]].all():
                    continue
                cv2.fillConvexPoly(candidate, vertices, 1)
                accepted.append(vertices.tolist())
        candidate[blocked] = 0
        # Never paint over another observed ground class, even with a nearby triangle.
        candidate[(observed != 0) & (observed != cid)] = 0
        distance = cv2.distanceTransform((~support).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) if support.any() else np.full((n,n),np.inf,dtype=np.float32)
        candidate_masks.append(candidate.astype(bool))
        distance_maps.append(distance)
        triangle_records.append({'class_id':cid,'support_cells':int(support.sum()),'accepted_triangles':len(accepted),'triangles_pixel_xy':accepted})
    filled = np.zeros_like(observed)
    nearest = np.full((n,n),np.inf,dtype=np.float32)
    for cid, mask, distance in zip(ground_ids,candidate_masks,distance_maps):
        update = mask & (distance < nearest)
        filled[update] = cid
        nearest[update] = distance[update]
    filled[observed != 0] = observed[observed != 0]
    filled[blocked] = 0
    inferred = (filled != 0) & (observed == 0)
    contours = []
    for cid in ground_ids:
        loops, hierarchy = cv2.findContours((filled==cid).astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        for i, contour in enumerate(loops):
            pixel_xy = contour[:,0,:]
            metric_xy = np.column_stack((-half+(pixel_xy[:,0]+.5)*resolution,half-(pixel_xy[:,1]+.5)*resolution))
            contours.append({'class_id':cid,'is_hole':bool(hierarchy[0,i,3]>=0),
                             'parent_contour_index':int(hierarchy[0,i,3]),'pixel_xy':pixel_xy.tolist(),'xy_m':metric_xy.tolist()})
    return {'observed':observed,'filled':filled,'inferred':inferred,'blocked':blocked,
            'point_count':np.bincount(flat,minlength=n*n).reshape(n,n).astype(np.uint32),
            'ground_point_count':support_count,'triangles':triangle_records,'contours':contours,
            'metadata':{'width':n,'height':n,'roi_size_m':roi_size,'resolution_m_per_pixel':resolution,
                        'bounds_xy_m':[-half,-half,half,half],'orientation':'+x right,+y up','grid_mode':'fixed_roi',
                        'max_triangle_edge_m':max_edge,'roi_points':int(inside.sum()),
                        'observed_ground_cells':int((observed!=0).sum()),'filled_ground_cells':int((filled!=0).sum()),
                        'inferred_cells':int(inferred.sum()),'blocked_cells':int(blocked.sum()),
                        'ground_class_ids':list(ground_ids),
                        'method':'per-class Delaunay triangles on observed cell centers, bounded edge length; nearest support resolves inferred overlaps',
                        'limitations':'Filled cells are inferred, not measured. No guaranteed road surface, grade/height estimation or motion reasoning. '
                                      'Disconnected distant supports remain separate; existing non-ground semantic cells veto fill. '
                                      'Triangles may still cross unobserved obstacles. No dense ground-truth metric.'}}


def _check_output(output: Path) -> None:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Output directory is not empty: {output}')


def save_ground_planes(output: Path, xyz: np.ndarray, labels: np.ndarray, resolution: float = .2,
                       roi_size: float = 100., max_edge: float = 6., assigned: np.ndarray | None = None,
                       source_info: dict | None = None, make_images: bool = True) -> dict:
    """Save bounded ground regions from coarse16 predictions without changing inputs.

    Class 0 is unassigned. ``assigned`` records original point assignments, independently
    of ROI filtering; when omitted it is ``labels != 0``. Source information must be
    JSON serializable. Validation completes before creating any output files.
    """
    output = Path(output)
    _check_output(output)
    result = ground_planes(xyz, labels, resolution, roi_size, max_edge)
    if assigned is None:
        assigned = labels != 0
    if not isinstance(assigned, np.ndarray) or assigned.shape != labels.shape or assigned.dtype != np.bool_:
        raise ValueError('assigned must be an aligned boolean array matching labels')
    if source_info is not None and not isinstance(source_info, dict):
        raise ValueError('source_info must be a JSON-serializable dictionary')
    payload = {'metadata': result['metadata'], 'contours': result['contours'],
               'triangles': result['triangles'], 'source_info': source_info or {}}
    if source_info is not None and 'source_instances' in source_info:
        payload['source_instances'] = source_info['source_instances']
    json.dumps(payload, allow_nan=False)
    output.mkdir(parents=True, exist_ok=True)
    if make_images:
        _save_ground_images(output, result, max_edge, source_info)
    np.savez_compressed(output/'ground_grid.npz', **{key: value for key, value in result.items() if isinstance(value, np.ndarray)})
    np.savez_compressed(output/'semantic_points.npz', xyz=xyz, labels=labels, assigned=assigned)
    save_json(output/'ground_regions.json', payload)
    m = result['metadata']
    print(f"Ground plane: {m['observed_ground_cells']} observed -> {m['filled_ground_cells']} filled cells ({m['inferred_cells']} inferred)", flush=True)
    return m


def _save_ground_images(output: Path, result: dict, max_edge: float, source_info: dict | None) -> None:
    palette = make_palette()
    def colored(ids):
        rgb = palette[ids].copy()
        rgb[ids==0] = 255
        return Image.fromarray(rgb)
    sparse, filled = colored(result['observed']),colored(result['filled'])
    sparse.save(output/'ground_observed.png')
    filled.save(output/'ground_plane.png')
    source = np.full((*result['filled'].shape,3),255,dtype=np.uint8)
    source[result['observed']!=0]=(35,140,85)
    source[result['inferred']]=(90,165,235)
    source[result['blocked']]=(200,65,55)
    Image.fromarray(source).save(output/'ground_sources.png')
    n = result['metadata']['width']
    canvas=Image.new('RGB',(n*3,n+140),'white')
    draw=ImageDraw.Draw(canvas)
    for i,(title,image) in enumerate((('Merged ground observations',sparse),('Continuous semantic ground regions',filled),('Green observed / blue filled / red blocked',Image.fromarray(source)))):
        canvas.paste(image,(i*n,30));draw.text((i*n+8,8),title,fill='black')
    m=result['metadata']
    fusion = 'Fused semantic point predictions' if (source_info or {}).get('kind') == 'fused_semantic_points' else 'Six-camera same-class fusion'
    for i,text in enumerate((f"{fusion} | {m['roi_size_m']:g}mx{m['roi_size_m']:g}m | {m['resolution_m_per_pixel']:g}m/cell | max triangle edge {max_edge:g}m",
                              f"Observed cells {m['observed_ground_cells']} | plane cells {m['filled_ground_cells']} | inferred {m['inferred_cells']}",
                              '11 road / 12 other flat / 13 sidewalk / 14 terrain; no-point areas outside supported triangles remain blank.',
                              'Filling is a hypothesis, not new LiDAR observations. Different ground classes stay separate.')):
        draw.text((8,n+40+i*20),text,fill='black')
    for i,cid in enumerate(GROUND_IDS):
        x=8+i*300;y=n+120
        draw.rectangle((x,y,x+15,y+15),fill=tuple(int(v) for v in palette[cid]));draw.text((x+22,y),{11:'road',12:'other flat',13:'sidewalk',14:'terrain'}[cid],fill='black')
    canvas.save(output/'ground_comparison.png')


def _semantic_roi(metadata: dict) -> tuple[float, float]:
    """Accept only the existing centered square fixed ROI; never widen an extent."""
    if metadata.get('grid_mode') != 'fixed_roi':
        raise ValueError('Ground regions require fixed_roi BEV metadata; automatic extents are unsupported')
    try:
        resolution = float(metadata['resolution_m_per_pixel'])
        bounds = np.array([metadata[key] for key in ('x_min_m', 'y_min_m', 'x_max_m', 'y_max_m')], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('BEV metadata requires resolution and finite XY bounds') from error
    if not np.isfinite(bounds).all() or not np.isfinite(resolution) or resolution <= 0:
        raise ValueError('BEV metadata requires resolution and finite XY bounds')
    spans = bounds[2:] - bounds[:2]
    if (np.any(spans <= 0) or not np.isclose(spans[0], spans[1], rtol=0, atol=1e-8)
            or not np.allclose(bounds[:2], -bounds[2:], rtol=0, atol=1e-8)):
        raise ValueError('Ground regions support only a square ROI centered at the origin')
    cells = spans[0] / resolution
    if not np.isfinite(cells) or not np.isclose(cells, round(cells), rtol=0, atol=1e-8):
        raise ValueError('BEV ROI/resolution must be integral')
    if any(key in metadata and metadata[key] != round(cells) for key in ('width', 'height')):
        raise ValueError('BEV width and height must match the fixed ROI and resolution')
    return resolution, float(spans[0])


def process_frame(frame: Path, output: Path, max_edge: float = 6., taxonomy: str = 'lidarseg16',
                  make_images: bool = True) -> dict:
    frame, output = Path(frame), Path(output)
    _check_output(output)
    if taxonomy not in ('lidarseg16', 'lidarseg32'):
        raise ValueError('taxonomy must be lidarseg16 or lidarseg32')
    if (frame/'instance_hulls.json').is_file():
        data = json.loads((frame/'instance_hulls.json').read_text(encoding='utf-8'))
        meta = data['metadata']
        with np.load(frame/'points.npz', allow_pickle=False) as archive:
            xyz = archive['xyz'].copy()
        labels, assigned = fuse_semantic_points(len(xyz), data['instances'])
        resolution, roi_size = meta['resolution_m_per_pixel'], meta['roi_size_m']
        source_info = {
            'kind': 'sam3_instance_points', 'path': str(frame), 'taxonomy': 'lidarseg16',
            'class_assignment': 'Highest instance confidence score per original point; equal scores prefer the lowest global instance ID. Instance class IDs are already coarse16.',
            'source_instances': [{'id': row['global_instance_id'], 'camera': row['camera'],
                                  'class_id': row['class_id'], 'point_indices': row['point_indices']}
                                 for row in data['instances'] if row['class_id'] in GROUND_IDS],
        }
    else:
        meta = json.loads((frame/'bev_metadata.json').read_text(encoding='utf-8'))
        resolution, roi_size = _semantic_roi(meta)
        with np.load(frame/'points.npz', allow_pickle=False) as archive:
            xyz, labels = archive['xyz'].copy(), archive['prediction'].copy()
        if labels.dtype != np.uint8 or labels.shape != (len(xyz),):
            raise ValueError('Expected aligned uint8 prediction labels')
        if taxonomy == 'lidarseg32':
            if np.any(labels >= len(RAW_TO_COARSE)):
                raise ValueError('lidarseg32 prediction labels must be in [0, 31]')
            labels = RAW_TO_COARSE[labels]
        assigned = None
        source_info = {
            'kind': 'fused_semantic_points', 'path': str(frame), 'taxonomy': 'lidarseg16',
            'source_taxonomy': taxonomy,
            'class_assignment': 'Saved per-point camera majority predictions; no instance confidence scores or ground truth are used. '
                                + ('RAW_TO_COARSE maps raw32 IDs to coarse16 before interpolation.' if taxonomy == 'lidarseg32'
                                   else 'Prediction IDs are already coarse16; no class remapping.'),
        }
    return save_ground_planes(output, xyz, labels, resolution, roi_size, max_edge,
                              assigned=assigned, source_info=source_info, make_images=make_images)


def _discover_frames(source: Path) -> list[Path]:
    if source.is_file() and source.name == 'points.npz':
        source = source.parent
    if (source/'points.npz').is_file() or (source/'instance_hulls.json').is_file():
        return [source]
    instances = {path.parent for pattern in ('*/instance_hulls.json', 'instance/*/instance_hulls.json',
                                              'instances/*/instance_hulls.json')
                 for path in source.glob(pattern)}
    if instances:
        return sorted(instances)
    return sorted({path.parent for pattern in ('*/points.npz', 'pointclouds/*/points.npz',
                                               'raw/pointcloud_results/pointclouds/*/points.npz')
                   for path in source.glob(pattern)})


def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input-dir',type=Path,default=ROOT/'outputs/nuscenes_instance_hulls_experiment')
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--samples',type=int,default=1)
    p.add_argument('--taxonomy', choices=('lidarseg16', 'lidarseg32'), default='lidarseg16',
                   help='Saved semantic prediction taxonomy; instance class IDs are already coarse16')
    p.add_argument('--max-edge',type=float,default=6.,help='Maximum triangle edge in meters; larger values bridge larger gaps')
    return p


def main(argv=None):
    args=parser().parse_args(argv);ensure_environment()
    if args.samples<1 or not np.isfinite(args.max_edge) or args.max_edge<=0:
        raise ValueError('samples must be>=1 and max-edge finite positive')
    source=args.input_dir.resolve()
    frames = _discover_frames(source)
    if not frames:raise FileNotFoundError(f'No saved instance or semantic points under {source}')
    output=(args.output_dir or ROOT/'outputs/nuscenes_ground_planes'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for frame in frames[:args.samples]:
        process_frame(frame,output/frame.name,args.max_edge,taxonomy=args.taxonomy)
        image=(output/frame.name/'ground_comparison.png').relative_to(output).as_posix()
        rows.append(f'<h2>{escape(frame.name)}</h2><a href="{escape(image,quote=True)}"><img src="{escape(image,quote=True)}" style="width:100%"></a>')
    (output/'result.html').write_text('<!doctype html><meta charset="utf-8"><title>Ground semantic planes</title><body style="font:16px system-ui;max-width:1600px;margin:24px auto"><h1>Merged road and sidewalk planes</h1><p>Blue source-map cells are inferred fill, not measured points. Ground classes remain separate. No dense GT evaluation.</p>'+''.join(rows),encoding='utf-8')
    save_json(output/'run_config.json',{'status':'completed','input_dir':str(source),'processed_frames':min(len(frames),args.samples),'max_edge_m':args.max_edge,'taxonomy':args.taxonomy})
    print(f'Ground region results: {output/"result.html"}',flush=True)
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,RuntimeError,FileNotFoundError,FileExistsError) as error:
        print(f'Error: {error}',file=sys.stderr);raise SystemExit(1)

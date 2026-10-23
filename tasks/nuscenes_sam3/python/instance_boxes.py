"""Upright oriented 3D boxes for non-surface instance support points."""
from __future__ import annotations

from pathlib import Path
import math
import numpy as np
from shared.python.runtime import save_json
from tasks.nuscenes_sam3.python.instance_hulls import convex_hull_xy


def minimum_rectangle_xy(points: np.ndarray) -> dict:
    """Find the minimum-area rectangle over convex-hull edge orientations.

    Returns float64 metric corners counterclockwise. Degenerate inputs remain
    empty/point/segment with zero area; no padding or synthetic extent is added.
    """
    points=np.asarray(points)
    if points.ndim!=2 or points.shape[1]!=2 or points.dtype.kind not in 'iuf' or not np.isfinite(points).all():
        raise ValueError('Rectangle input must be finite numeric Nx2 coordinates')
    hull=convex_hull_xy(points.astype(np.float64))
    base={'input_point_count':len(points),'unique_xy_count':len(np.unique(points,axis=0))}
    if len(hull)<3:
        center=hull.mean(axis=0) if len(hull) else None
        length=float(np.linalg.norm(hull[1]-hull[0])) if len(hull)==2 else 0.
        angle=math.atan2(*(hull[1]-hull[0])[::-1]) if len(hull)==2 else 0.
        angle=(angle+math.pi/2)%math.pi-math.pi/2
        return {**base,'geometry_type':('empty','point','segment')[len(hull)],'corners_xy':hull.tolist(),
                'center_xy':None if center is None else center.tolist(),'length_m':length,'width_m':0.,'yaw_rad':angle,'area_m2':0.}
    origin=hull.mean(axis=0);centered=hull-origin
    edges=np.roll(hull,-1,axis=0)-hull
    angles=np.unique(np.mod(np.arctan2(edges[:,1],edges[:,0]),math.pi/2))
    candidates=[]
    for angle in angles:
        cosine,sine=math.cos(angle),math.sin(angle)
        axes=np.array([[cosine,-sine],[sine,cosine]],np.float64)
        local=centered@axes
        low,high=local.min(0),local.max(0);extents=high-low
        candidates.append((float(np.prod(extents)),float(angle),axes,low,high))
    area,angle,axes,low,high=min(candidates,key=lambda row:(row[0],row[1]))
    extents=high-low
    if extents[0]<extents[1]:
        # Recompute a canonical rectangle whose first edge is its longest side.
        angle+=math.pi/2
    angle=(angle+math.pi/2)%math.pi-math.pi/2
    axes=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    local=centered@axes;low,high=local.min(0),local.max(0);length,width=high-low
    corners=np.array([[low[0],low[1]],[high[0],low[1]],[high[0],high[1]],[low[0],high[1]]])@axes.T+origin
    # Keep winding and starting vertex deterministic for consumers.
    first=np.lexsort((corners[:,1],corners[:,0]))[0];corners=np.roll(corners,-int(first),axis=0)
    return {**base,'geometry_type':'rectangle','corners_xy':corners.tolist(),
            'center_xy':(((low+high)/2)@axes.T+origin).tolist(),
            'length_m':float(length),'width_m':float(width),'yaw_rad':float(angle),'area_m2':float(length*width)}


def build_instance_boxes(xyz: np.ndarray, instances: list[dict], coordinate_frame='input XYZ', surface_class_ids=()) -> dict:
    """Enclose non-surface support with minimum XY area and exact Z extrema.

    Boxes are upright in the requested Z-up frame, with four lower corners then
    four upper corners. Degenerate support keeps zero width/height without padding.
    This minimizes volume among upright boxes, not arbitrary roll/pitch cuboids.
    """
    surface_ids=set(surface_class_ids)
    if any(isinstance(cid,bool) or not isinstance(cid,(int,np.integer)) or not 1<=cid<=254 for cid in surface_ids):
        raise ValueError('Surface class IDs must be integers in1..254')
    xyz=np.asarray(xyz)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or xyz.dtype.kind not in 'iuf' or not np.isfinite(xyz).all():
        raise ValueError('Instance XYZ must be finite numeric Nx3')
    boxes=[];identifiers=set()
    for index,record in enumerate(instances):
        if int(record['class_id']) in surface_ids:continue
        indices=np.asarray(record['point_indices'])
        if indices.ndim!=1 or (indices.size and indices.dtype.kind not in 'iu') or np.any(indices<0) or np.any(indices>=len(xyz)):
            raise ValueError('Instance point indices must refer to the original XYZ array')
        indices=np.unique(indices.astype(np.int64))
        identifier=int(record.get('global_instance_id',index+1))
        if identifier<1 or identifier in identifiers:raise ValueError('Bounding-box instance IDs must be positive and unique')
        identifiers.add(identifier)
        points=xyz[indices].astype(np.float64)
        rectangle=minimum_rectangle_xy(points[:,:2])
        z_min=float(points[:,2].min()) if len(points) else None
        z_max=float(points[:,2].max()) if len(points) else None
        height=z_max-z_min if len(points) else 0.
        corners=np.asarray(rectangle['corners_xy'],dtype=np.float64).reshape(-1,2)
        # A footprint has four slots even for zero-width or singleton support.
        if len(corners)==1:corners=np.repeat(corners,4,axis=0)
        elif len(corners)==2:corners=corners[[0,1,1,0]]
        bottom=np.column_stack((corners,np.full(len(corners),z_min if z_min is not None else 0.)))
        top=np.column_stack((corners,np.full(len(corners),z_max if z_max is not None else 0.)))
        corners_scene=np.concatenate((bottom,top),axis=0)
        center=None if rectangle['center_xy'] is None else [*rectangle['center_xy'],(z_min+z_max)/2]
        footprint_type=rectangle['geometry_type']
        boxes.append({**rectangle,'footprint_geometry_type':footprint_type,'geometry_type':'cuboid' if len(points) else 'empty','global_instance_id':identifier,'camera_instance_id':int(record.get('camera_instance_id',record.get('instance_id',identifier))),
                      'camera':record.get('camera','unknown'),'class_id':int(record['class_id']),
                      'class_name':record.get('class_name',str(record['class_id'])),'score':float(record.get('score',0.)),
                      'source_cameras':list(record.get('source_cameras',[record.get('camera','unknown')])),
                      'member_instance_ids':list(record.get('member_instance_ids',[identifier])),
                      'member_count':int(record.get('member_count',1)),'merged':bool(record.get('merged',False)),
                      'point_indices':indices.tolist(),'support_count':len(indices),'center_scene':center,
                      'corners_scene':corners_scene.tolist(),'footprint_corners_scene':bottom.tolist(),'height_m':height,
                      'volume_m3':rectangle['area_m2']*height,
                      'edges':[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]] if len(points) else [],
                      'z_min_m':z_min,'z_max_m':z_max})
    metadata={'coordinate_frame':coordinate_frame,
              'rule':'upright oriented3D bounding box: minimum-area rectangle in XY and min/max retained Z',
              'height_rule':'bottom=min support Z, top=max support Z; centerZ=(min+max)/2',
              'corner_order':'lower0..3 counterclockwise, upper4..7 aligned;12 edges',
              'excluded_surface_class_ids':sorted(int(cid) for cid in surface_ids),
              'excluded_surface_instances':sum(int(record['class_id']) in surface_ids for record in instances),
              'instance_scope':'frame-local shared-point cross-camera groups; no cross-frame tracking' if any('member_instance_ids' in record for record in instances) else 'camera-local SAM3 detection; no cross-camera association or tracking is inferred',
              'input_instances':len(instances),'eligible_instances':len(boxes),
              'cuboids':sum(box['geometry_type']=='cuboid' for box in boxes),
              'degenerate':sum(box['geometry_type']=='cuboid' and box['volume_m3']==0 for box in boxes),
              'empty':sum(box['geometry_type']=='empty' for box in boxes),
              'limitations':'Boxes enclose retained observed support, not full object extents or GT. '
                            'Surface classes are excluded. Zero extent is retained without artificial padding. '
                            'Yaw is optimized; roll and pitch stay aligned to frame Z.'}
    return {'metadata':metadata,'boxes':boxes}


def save_instance_boxes(path: Path, xyz: np.ndarray, instances: list[dict], coordinate_frame='input XYZ', surface_class_ids=()) -> dict:
    result=build_instance_boxes(xyz,instances,coordinate_frame,surface_class_ids)
    save_json(path,result)
    return result

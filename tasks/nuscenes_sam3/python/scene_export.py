#!/usr/bin/env python
"""Per-scene offline timeline: polygon ground surfaces and colored original points."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
from html import escape
import json
import shutil
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

from shared.python.runtime import ROOT, ensure_environment, save_json
from tasks.flat_ground.python.geometry import pose_matrix, ego_yaw_frame
from tasks.flat_ground.python.project_semantics import CameraFrames, load_saved_masks
from tasks.nuscenes_sam3.python.six_class import NAMES, six_ground_plane
from shared.python.sam3 import overlay
from tasks.nuscenes_sam3.python.semantic_schema import saved_schema,saved_display_names,saved_colors,resolve_planes,class_rows,palette_for
from tasks.nuscenes_sam3.python.instance_filter import add_filter_args,config_from_args,validate_filter
from tasks.nuscenes_sam3.python.filter_support import clean_saved_support,save_cleaning
from tasks.nuscenes_sam3.python.instance_merge import merge_instance_support
from tasks.nuscenes_sam3.python.instance_boxes import save_instance_boxes
from tasks.nuscenes_sam3.python.occupancy_grid import save_occupancy_grid,flatten_surface_occupancy,save_flat_occupancy

COLORS = [(170,177,188),(210,70,180),(210,155,20),(35,110,225),(200,55,45),(40,160,85),(125,65,210)]
LABELS = ['未分配 / unassigned','人行道 / sidewalk','道路 / road','行人 / pedestrian','车辆 / vehicle','两轮车 / bike','骑乘者 / rider']


def transform_points(xyz: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all() or matrix.shape!=(4,4) or not np.isfinite(matrix).all():
        raise ValueError('Finite Nx3 coordinates and 4x4 transform required')
    return xyz.astype(np.float64) @ matrix[:3,:3].T + matrix[:3,3]


def grid_runs(labels: np.ndarray, class_id: int) -> np.ndarray:
    """Exact row-wise rectangles [row, first_col, end_exclusive], retaining holes."""
    if labels.ndim!=2:raise ValueError('A two-dimensional semantic grid is required')
    runs=[]
    for row, values in enumerate(labels==class_id):
        edges=np.diff(np.pad(values.astype(np.int8),(1,1)))
        runs.extend((row,int(start),int(end)) for start,end in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))
    return np.asarray(runs,dtype=np.int32).reshape(-1,3)


def runs_to_triangles(runs: np.ndarray, resolution: float, roi_size: float, source_to_scene: np.ndarray,
                      source_height: float=0., world_height: float | None=None, global_to_scene: np.ndarray | None=None) -> np.ndarray:
    """Two triangles per row rectangle; transform cell EDGES, not centers."""
    if resolution<=0 or roi_size<=0 or not np.isfinite([resolution,roi_size,source_height]).all():
        raise ValueError('Finite positive grid dimensions and finite height required')
    if len(runs)==0:return np.empty((0,3),np.float32)
    row,start,end=runs.T
    left=-roi_size/2+start*resolution;right=-roi_size/2+end*resolution
    top=roi_size/2-row*resolution;bottom=top-resolution
    corners=np.stack((np.column_stack((left,bottom)),np.column_stack((right,bottom)),
                      np.column_stack((right,top)),np.column_stack((left,top))),axis=1)
    xyz=np.concatenate((corners,np.full((*corners.shape[:-1],1),source_height)),axis=-1).reshape(-1,3)
    transformed=transform_points(xyz,source_to_scene)
    if world_height is not None:
        if global_to_scene is None:raise ValueError('global_to_scene required with world_height')
        # source_to_scene is source_to_global in this branch; flatten at measured median world height.
        transformed[:,2]=world_height
        transformed=transform_points(transformed,global_to_scene)
    quads=transformed.reshape(-1,4,3)
    return quads[:,[0,1,2,0,2,3],:].reshape(-1,3).astype('<f4')


def packed(array: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.asarray(array,dtype=dtype).tobytes(order='C')).decode('ascii')


def project_camera_points(xyz_global: np.ndarray, camera: dict, image_size: tuple[int, int]):
    """Project global LiDAR points to unique camera pixels, retaining nearest indices."""
    xyz=np.asarray(xyz_global)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or xyz.dtype.kind not in 'fiu' or not np.isfinite(xyz).all():
        raise ValueError('xyz_global must be a finite Nx3 array')
    if not isinstance(camera,dict):raise ValueError('camera must be a camera record')
    if not isinstance(image_size,(tuple,list)) or len(image_size)!=2:
        raise ValueError('image_size must be positive integer (width, height)')
    width,height=image_size
    if any(isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<=0
           for value in (width,height)):
        raise ValueError('image_size must be positive integer (width, height)')
    try:
        camera_to_global=np.asarray(camera['camera_to_global'])
        intrinsic=np.asarray(camera['intrinsic'])
    except (KeyError,TypeError,ValueError) as error:
        raise ValueError('camera requires finite camera_to_global and intrinsic matrices') from error
    if (camera_to_global.shape!=(4,4) or intrinsic.shape!=(3,3)
            or camera_to_global.dtype.kind not in 'fiu' or intrinsic.dtype.kind not in 'fiu'
            or not np.isfinite(camera_to_global).all() or not np.isfinite(intrinsic).all()):
        raise ValueError('camera_to_global and intrinsic must be finite 4x4 and 3x3 matrices')
    camera_to_global=camera_to_global.astype(np.float64);intrinsic=intrinsic.astype(np.float64)
    rotation=camera_to_global[:3,:3]
    if (not np.allclose(camera_to_global[3],(0.,0.,0.,1.),rtol=0,atol=1e-8)
            or not np.allclose(rotation.T@rotation,np.eye(3),rtol=0,atol=1e-7)
            or not np.isclose(np.linalg.det(rotation),1.,rtol=0,atol=1e-7)):
        raise ValueError('camera_to_global must be a rigid transform')
    try:
        global_to_camera=np.linalg.inv(camera_to_global)
        intrinsic_inverse=np.linalg.inv(intrinsic)
    except np.linalg.LinAlgError as error:
        raise ValueError('camera_to_global and intrinsic must be invertible') from error
    if not np.isfinite(global_to_camera).all() or not np.isfinite(intrinsic_inverse).all():
        raise ValueError('camera matrix inverses must be finite')
    if len(xyz)==0:return np.empty(0,np.int64),np.empty((0,2),np.int32),np.empty(0,np.float64)
    camera_xyz=xyz.astype(np.float64)@global_to_camera[:3,:3].T+global_to_camera[:3,3]
    depth=camera_xyz[:,2]
    projected=camera_xyz@intrinsic.T
    denominator=projected[:,2]
    valid=np.isfinite(camera_xyz).all(axis=1)&np.isfinite(projected).all(axis=1)&(depth>1.)&(np.abs(denominator)>1e-12)
    uv=np.full((len(xyz),2),np.nan,dtype=np.float64)
    uv[valid]=projected[valid,:2]/denominator[valid,None]
    valid &= (uv[:,0]>1.)&(uv[:,0]<width-1)&(uv[:,1]>1.)&(uv[:,1]<height-1)
    indices=np.flatnonzero(valid)
    if not len(indices):return indices,np.empty((0,2),np.int32),depth[indices]
    pixels=np.floor(uv[indices]).astype(np.int32)
    pixel_id=pixels[:,1].astype(np.int64)*int(width)+pixels[:,0]
    # Pixel first, then depth, then original index: ties are deterministic.
    order=np.lexsort((indices,depth[indices],pixel_id))
    ordered_pixels=pixel_id[order]
    keep=np.r_[True,ordered_pixels[1:]!=ordered_pixels[:-1]]
    chosen=order[keep]
    return indices[chosen],pixels[chosen],depth[indices[chosen]]


def camera_images(cameras, scene_dir: Path, sample_token: str, masks, names, colors=None,
                  xyz_global=None,point_labels=None) -> list[dict]:
    """Link originals, SAM3 overlays and optional projected LiDAR products."""
    if len(cameras)!=len(masks):raise ValueError('Each camera requires a saved semantic mask')
    if (xyz_global is None)!=(point_labels is None):raise ValueError('xyz_global and point_labels must be supplied together')
    if xyz_global is not None:
        xyz_global=np.asarray(xyz_global)
        point_labels=np.asarray(point_labels)
        if (xyz_global.ndim!=2 or xyz_global.shape[1]!=3 or xyz_global.dtype.kind not in 'fiu'
                or not np.isfinite(xyz_global).all() or point_labels.shape!=(len(xyz_global),)
                or point_labels.dtype.kind not in 'ui' or not np.isin(point_labels,list(names)).all()):
            raise ValueError('Point coordinates and semantic labels must be aligned and valid')
    image_dir=scene_dir/'camera_images'/sample_token
    image_dir.mkdir(parents=True)
    images=[];palette=palette_for(names,colors)
    for camera,mask in zip(cameras,masks):
        path=Path(camera['image_path'])
        with Image.open(path) as image:
            width,height=image.size
            if mask.shape!=(height,width) or mask.dtype!=np.uint8 or not np.isin(mask,list(names)).all():
                raise ValueError('Camera semantic mask must match image dimensions and saved taxonomy')
            original=image.convert('RGB')
            semantic=overlay(original,mask,palette,.5)
        target=image_dir/f"{camera['camera']}{path.suffix.lower()}"
        semantic_target=image_dir/f"{camera['camera']}_overlay.png"
        shutil.copyfile(path,target)
        semantic.save(semantic_target)
        record={'camera':camera['camera'],'timestamp_us':int(camera['timestamp']),
                'width':width,'height':height,'src':target.relative_to(scene_dir).as_posix(),
                'overlay_src':semantic_target.relative_to(scene_dir).as_posix()}
        if xyz_global is not None:
            indices,pixels,depth=project_camera_points(xyz_global,camera,(width,height))
            points_image=original.copy();overlay_points=semantic.copy()
            # Point circles are drawn far-to-near so nearer samples remain visible.
            order=np.lexsort((indices,-depth))
            for product in (points_image,overlay_points):
                draw=ImageDraw.Draw(product)
                for index in order:
                    x,y=(int(value) for value in pixels[index])
                    label=point_labels[indices[index]]
                    color=(255,255,255) if label==0 else tuple(int(value) for value in palette[label])
                    draw.ellipse((x-2,y-2,x+2,y+2),fill=color)
            points_target=image_dir/f"{camera['camera']}_points.png"
            overlay_points_target=image_dir/f"{camera['camera']}_overlay_points.png"
            points_image.save(points_target);overlay_points.save(overlay_points_target)
            record.update(points_src=points_target.relative_to(scene_dir).as_posix(),
                          overlay_points_src=overlay_points_target.relative_to(scene_dir).as_posix(),
                          projected_point_count=len(indices))
        images.append(record)
    return images


def flat_occupancy_mesh(record,resolution,origin,plane_z_scene,plane_ids):
    """Build non-overlapping Z=0 quads from sparse XY occupancy cells."""
    indices=np.asarray(record['indices'],dtype=np.int64).reshape(-1,2)
    labels=np.asarray(record['class_ids'],dtype=np.uint8)
    surfaces={}
    for cid in plane_ids:
        keys=indices[labels==cid]
        if len(keys):
            low=keys.astype(np.float64)*resolution+np.asarray(origin)
            high=low+resolution
            corners=np.stack((low,np.column_stack((high[:,0],low[:,1])),high,np.column_stack((low[:,0],high[:,1]))),axis=1)
            vertices=np.concatenate((corners,np.full((*corners.shape[:2],1),plane_z_scene)),axis=2)[:,[0,1,2,0,2,3]].reshape(-1,3)
        else:vertices=np.empty((0,3),np.float32)
        surfaces[str(cid)]={'xyz':packed(vertices,'<f4'),'count':len(vertices)}
    return surfaces


def source_frames(input_dir: Path, dataset: CameraFrames, scenes=None):
    paths=[input_dir] if (input_dir/'points.npz').is_file() else [path.parent for path in input_dir.glob('*/points.npz')]
    if not paths:raise FileNotFoundError(f'No saved six-class point frames under {input_dir}')
    selected=[];seen=set()
    for frame in paths:
        record=json.loads((frame/'cameras/CAM_FRONT_source.json').read_text())
        token=record['sample_token']
        if token in seen:raise ValueError('Repeated sample token in input frames')
        seen.add(token)
        sample=dataset.tables['sample'][token];scene=dataset.tables['scene'][sample['scene_token']]
        if scenes and scene['name'] not in scenes:continue
        selected.append({'path':frame,'sample':sample,'scene':scene})
    if not selected:raise ValueError('No matching saved scenes')
    return sorted(selected,key=lambda row:(row['scene']['name'],row['sample']['timestamp'],row['sample']['token']))


def lidar_records(dataset: CameraFrames):
    records={}
    for record in dataset.tables['sample_data'].values():
        if not record['is_key_frame']:continue
        calibration=dataset.tables['calibrated_sensor'][record['calibrated_sensor_token']]
        sensor=dataset.tables['sensor'][calibration['sensor_token']]
        if sensor['channel']=='LIDAR_TOP':records[record['sample_token']]=record
    return records


def export_scenes(input_dir: Path, output_dir: Path, dataset_dir: Path, scenes=None,max_frames=0,plane_z=0.,max_distance=80.,max_edge=None,map_resolution=.2,plane_classes=None,point_filter=None,bev_resolution=None,bev_roi_size=None,compute_grid_projection=False,grid_projection_samples=3,grid_projection_max_distance=80.,version='v1.0-mini',occupancy_resolution=.5) -> dict:
    filter_config=validate_filter(point_filter)
    if isinstance(occupancy_resolution,(bool,np.bool_)) or not np.isfinite(occupancy_resolution) or occupancy_resolution<=0:
        raise ValueError('occupancy-resolution must be finite and positive')
    if not isinstance(compute_grid_projection,(bool,np.bool_)):raise ValueError('compute-grid-projection must be boolean')
    if isinstance(grid_projection_samples,(bool,np.bool_)) or not isinstance(grid_projection_samples,(int,np.integer)) or not 1<=grid_projection_samples<=8:
        raise ValueError('grid-projection-samples must be an integer in 1..8')
    if not np.isfinite(grid_projection_max_distance) or grid_projection_max_distance<=0:
        raise ValueError('grid-projection-max-distance must be finite and positive')
    input_dir=Path(input_dir).resolve();output=Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):raise FileExistsError(output)
    # Retain the legacy positional plane_z/max_distance arguments; polygon exports ignore them.
    if max_frames<0:raise ValueError('max-frames must be nonnegative')
    for name,value in (('bev-resolution',bev_resolution),('bev-roi-size',bev_roi_size)):
        if value is not None and (not np.isfinite(value) or value<=0):raise ValueError(f'{name} must be finite and positive')
    if max_edge is not None and (not np.isfinite(max_edge) or max_edge<=0):
        raise ValueError('max-edge must be finite and positive')
    if not np.isfinite(map_resolution) or map_resolution<=0:
        raise ValueError('map-resolution must be finite positive')
    dataset=CameraFrames(dataset_dir,version=version)
    sources=source_frames(input_dir,dataset,scenes)
    if max_frames:sources=sources[:max_frames]
    names,default_planes=saved_schema(sources[0]["path"],allow_legacy=True)
    display_names=saved_display_names(sources[0]["path"])
    colors=saved_colors(sources[0]["path"])
    plane_ids=resolve_planes(names,plane_classes,default_planes)
    for item in sources:
        other,_=saved_schema(item["path"],allow_legacy=True)
        if other!=names:raise ValueError("Selected saved frames have different taxonomies; export them separately")
        if saved_colors(item["path"])!=colors:raise ValueError("Selected saved frames have different saved colors; export them separately")
    lidars=lidar_records(dataset)
    groups={}
    for row in sources:groups.setdefault(row['scene']['name'],[]).append(row)
    template=(ROOT/'tasks/nuscenes_sam3/web/scene_viewer.html').read_text(encoding='utf-8')
    if template.count('__SCENE_DATA__')!=1:raise ValueError('Scene template must contain one data placeholder')
    output.mkdir(parents=True,exist_ok=True)
    state={'status':'running','input_dir':str(input_dir),'dataset_dir':str(dataset_dir),'version':version,'processed_frames':0,'scenes':[],
           'point_filter':filter_config,'bev_resolution':bev_resolution,'bev_roi_size':bev_roi_size,'occupancy_resolution':float(occupancy_resolution),'plane_class_ids':list(plane_ids),'plane_classes':[names[i] for i in plane_ids],'created_at':datetime.now(timezone.utc).isoformat()}
    if compute_grid_projection:
        state.update(compute_grid_projection=True,grid_projection_samples=int(grid_projection_samples),
                     grid_projection_max_distance=float(grid_projection_max_distance))
    save_json(output/'run_config.json',state)
    try:
        for scene_name,records in groups.items():
            scene_dir=output/scene_name;scene_dir.mkdir()
            first_cameras=dataset.cameras(records[0]['sample']['token'])
            scene_to_global=ego_yaw_frame(first_cameras[0]['ego_pose'],0.)
            global_to_scene=np.linalg.inv(scene_to_global)
            first_timestamp=int(records[0]['sample']['timestamp'])
            all_scene_samples=[sample for sample in dataset.tables['sample'].values() if sample['scene_token']==records[0]['scene']['token']]
            payload={'scene':scene_name,'classes':class_rows(names,display_names,colors),'plane_class_ids':list(plane_ids),
                     'frames':[],'bounds':{'min':[np.inf]*3,'max':[-np.inf]*3},
                     'metadata':{'coordinate_frame':'First displayed frame CAM_FRONT ego XY and yaw, global z up; meters',
                       'scene_to_global':scene_to_global.tolist(),'available_frames':len(records),'dataset_scene_frames':len(all_scene_samples),
                       'point_filter':filter_config,'complete_scene':len(records)==len(all_scene_samples),'plane_class_ids':list(plane_ids),'plane_classes':[names[i] for i in plane_ids],
                       'limitations':'Only exported frames are included. Polygon is locally interpolated from labeled LiDAR points. '
                         'The surface is not GT. All-frame overlay retains moving-object trails; no object tracking/motion compensation. '
                         'Point coordinates are rigidly pose-aligned without sampling or flattening. Each class plane uses median observed world height.'}}
            payload['metadata']['surface_methods']=['polygon']+(['grid_projection'] if compute_grid_projection else [])
            frame_manifest=[]
            for row in records:
                frame=row['path'];sample=row['sample'];token=sample['token'];cameras=dataset.cameras(token)
                masks=load_saved_masks(frame,cameras)
                with np.load(frame/'points.npz',allow_pickle=False) as archive:
                    xyz=archive['xyz'].copy();prediction=archive['prediction'].copy()
                    if 'camera_names' in archive:
                        if archive['camera_names'].tolist()!=[camera['camera'] for camera in cameras]:raise ValueError('Point fusion camera order mismatch')
                if xyz.ndim!=2 or xyz.shape[1]!=3 or prediction.shape!=(len(xyz),) or prediction.dtype!=np.uint8 or np.any(prediction>=len(names)) or not np.isfinite(xyz).all():
                    raise ValueError('Saved points must match the saved taxonomy and original finite Nx3 xyz')
                scan=lidars[token];lidar_cal=dataset.tables['calibrated_sensor'][scan['calibrated_sensor_token']]
                lidar_ego=dataset.tables['ego_pose'][scan['ego_pose_token']]
                lidar_to_global=pose_matrix(lidar_ego)@pose_matrix(lidar_cal)
                lidar_to_scene=global_to_scene@lidar_to_global
                xyz_scene=transform_points(xyz,lidar_to_scene)
                plane_meta=json.loads((frame/'plane_metadata.json').read_text())
                roi_size=float(bev_roi_size if bev_roi_size is not None else plane_meta['roi_size_m'])
                resolution=float(bev_resolution if bev_resolution is not None else plane_meta['resolution_m_per_pixel'])
                edge=float(max_edge if max_edge is not None else plane_meta['max_triangle_edge_m'])
                clean=clean_saved_support(frame,xyz,prediction,filter_config,surface_class_ids=plane_ids)
                filtered_prediction=clean['prediction']
                merged=merge_instance_support(clean['instances'],surface_class_ids=plane_ids)
                save_json(scene_dir/f'{token}_instance_merge.json',merged)
                boxes=save_instance_boxes(scene_dir/f'{token}_boxes.json',xyz_scene,merged['instances'],
                                           coordinate_frame=payload['metadata']['coordinate_frame'],surface_class_ids=plane_ids)
                polygon=six_ground_plane(xyz,filtered_prediction,resolution,roi_size,edge,names,plane_ids)
                save_cleaning(scene_dir/'cleaning'/token,xyz,prediction,clean,roi_size,resolution)
                archive_arrays={'xyz_lidar':xyz,'xyz_scene':xyz_scene,'labels':prediction,
                                'filtered_labels':filtered_prediction,'filter_keep_mask':clean['keep_mask'],'removed_point_indices':clean['removed_indices'],'point_indices':np.arange(len(xyz),dtype=np.int32),
                                'lidar_to_global':lidar_to_global,'scene_to_global':scene_to_global,
                                'polygon_labels':polygon['filled']}
                surfaces={'polygon':{}};surface_stats={}
                projection=None
                grids=[('polygon',polygon['filled'])]
                if compute_grid_projection:
                    from tasks.nuscenes_sam3.python.grid_projection import project_ground_cells
                    projection=project_ground_cells(cameras,masks,roi_size,resolution,plane_ids,len(names),
                                                    samples_per_axis=grid_projection_samples,max_distance=grid_projection_max_distance)
                    archive_arrays['grid_projection_labels']=projection['labels']
                    archive_arrays['grid_projection_to_global']=projection['global_from_bev']
                    for key in ('camera_count','sample_count','support_count','conflict'):
                        archive_arrays[f'grid_projection_{key}']=projection[key]
                    surfaces['grid_projection']={}
                    grids.append(('grid_projection',projection['labels']))
                world=transform_points(xyz,lidar_to_global)
                for method,grid in grids:
                    for cid in plane_ids:
                        runs=grid_runs(grid,cid)
                        ground=filtered_prediction==cid
                        height=float(np.median(world[ground,2])) if ground.any() else 0.
                        sensor_height=float(np.median(xyz[ground,2])) if ground.any() else 0.
                        if method=='grid_projection':
                            height=0.
                            triangles=runs_to_triangles(runs,resolution,roi_size,global_to_scene@projection['global_from_bev'],source_height=0.)
                        else:
                            triangles=runs_to_triangles(runs,resolution,roi_size,lidar_to_global,sensor_height,height,global_to_scene)
                        surfaces[method][str(cid)]={'xyz':packed(triangles,'<f4'),'count':len(triangles)}
                        archive_arrays[f'{method}_{cid}_runs']=runs
                        archive_arrays[f'{method}_{cid}_triangles_scene']=triangles
                        surface_stats[f'{method}_{cid}']={'cells':int((grid==cid).sum()),'rectangles':len(runs),
                                                        'triangles':len(triangles)//3,'plane_z_global_m':height}
                        if len(triangles):
                            payload['bounds']['min']=np.minimum(payload['bounds']['min'],triangles.min(0)).tolist()
                            payload['bounds']['max']=np.maximum(payload['bounds']['max'],triangles.max(0)).tolist()
                # All original points are retained, including unassigned; WebGL class filters control display only.
                if len(xyz_scene):
                    payload['bounds']['min']=np.minimum(payload['bounds']['min'],xyz_scene.min(0)).tolist()
                    payload['bounds']['max']=np.maximum(payload['bounds']['max'],xyz_scene.max(0)).tolist()
                ego=transform_points(np.array([cameras[0]['ego_pose']['translation']]),global_to_scene)[0]
                name=frame.name
                entry={'sample_token':token,'frame_name':name,'timestamp_us':int(sample['timestamp']),
                       'time_s':(int(sample['timestamp'])-first_timestamp)/1e6,'ego':ego.tolist(),
                        'camera_images':camera_images(cameras,scene_dir,token,masks,names,colors,xyz_global=world,point_labels=filtered_prediction),
                       'points':{'xyz':packed(xyz_scene,'<f4'),'labels':packed(filtered_prediction,'u1'),'count':len(xyz)},
                       'bounding_boxes':[{key:value for key,value in box.items() if key!='point_indices'} for box in boxes['boxes']],
                        'surfaces':surfaces,'stats':{
                           'polygon_cells':int(np.count_nonzero(polygon['filled'])),'points':len(xyz)}}
                if projection is not None:
                    entry['stats']['grid_projection_cells']=int(np.count_nonzero(projection['labels']))
                    entry['stats']['grid_projection']=projection['metadata']
                payload['frames'].append(entry)
                np.savez_compressed(scene_dir/f'{token}.npz',**archive_arrays)
                frame_manifest.append({'sample_token':token,'frame_name':name,'timestamp_us':int(sample['timestamp']),
                    'time_s':entry['time_s'],'source_dir':str(frame),'original_points':len(xyz),
                    'instance_merge':merged['metadata'],
                     'instance_merge_file':f'{token}_instance_merge.json',
                     'bounding_boxes':boxes['metadata'],'bounding_box_file':f'{token}_boxes.json',
                    'point_filter':clean['metadata'],'point_class_counts':{names[cid]:int((prediction==cid).sum()) for cid in names},
                    'lidar_token':scan['token'],'lidar_timestamp':scan['timestamp'],'lidar_to_global':lidar_to_global.tolist(),
                    'roi_size_m':roi_size,'resolution_m':resolution,
                    'polygon_max_edge_m':edge,'surface_stats':surface_stats})
                if projection is not None:
                    frame_manifest[-1]['grid_projection_to_global']=np.asarray(projection['global_from_bev']).tolist()
                    frame_manifest[-1]['grid_projection']=projection['metadata']
                    surface_stats['grid_projection']=projection['metadata']
                state['processed_frames']+=1
                print(f"{scene_name} {len(payload['frames'])}/{len(records)}: original points={len(xyz)}, polygon cells={entry['stats']['polygon_cells']}",flush=True)
            def aligned_frame_clouds():
                for row in records:
                    with np.load(scene_dir/f"{row['sample']['token']}.npz",allow_pickle=False) as archive:
                        xyz_scene=archive['xyz_scene']
                        filtered_labels=archive['filtered_labels']
                    yield xyz_scene,filtered_labels
            occupancy=save_occupancy_grid(scene_dir,aligned_frame_clouds(),resolution=occupancy_resolution,semantic=True)
            def unassigned_frame_clouds():
                for xyz_scene,filtered_labels in aligned_frame_clouds():
                    yield xyz_scene[filtered_labels==0]
            unassigned=save_occupancy_grid(scene_dir,unassigned_frame_clouds(),resolution=occupancy_resolution,
                basename='unassigned_occupancy_grid',metadata={
                    'semantic_filtering':True,'semantic_source':'aligned_frame_filtered_labels_class_id_0',
                    'source_class_ids':[0],
                    'selection_note':'Display class 0: originally unassigned and filter-removed semantic points. '
                                     '显示类别0：原始未分配点及过滤后移除语义的点。'})
            original_counts=[entry['points']['count'] for entry in payload['frames']]
            if (len(occupancy['frames'])!=len(original_counts) or len(unassigned['frames'])!=len(original_counts)
                or any(semantic['point_count']+unknown['point_count']!=count
                       for semantic,unknown,count in zip(occupancy['frames'],unassigned['frames'],original_counts))
                or occupancy['fused']['point_count']+unassigned['fused']['point_count']!=sum(original_counts)):
                raise RuntimeError('Semantic and unassigned occupancy must account for every original point')
            payload['unassigned_occupancy_grid']=unassigned
            payload['metadata']['unassigned_occupancy_grid']={key:unassigned[key] for key in ('resolution','origin','metadata')}
            flat=flatten_surface_occupancy(occupancy,plane_ids,plane_z_scene=float(global_to_scene[2,3]))
            save_flat_occupancy(scene_dir,flat)
            payload['occupancy_flat']=flat
            payload['fused_surfaces']={'occupancy_flat':flat_occupancy_mesh(flat['fused'],flat['resolution'],flat['origin'],flat['plane_z_scene'],plane_ids)}
            payload['metadata']['surface_methods'].append('occupancy_flat')
            for entry,flat_frame in zip(payload['frames'],flat['frames']):
                entry['surfaces']['occupancy_flat']=flat_occupancy_mesh(flat_frame,flat['resolution'],flat['origin'],flat['plane_z_scene'],plane_ids)
                entry['stats']['occupancy_flat_cells']=flat_frame['cell_count']
            payload['occupancy_grid']=occupancy
            payload['metadata']['occupancy_grid']={key:occupancy[key] for key in ('resolution','origin','metadata')}
            if not np.isfinite(np.array([payload['bounds']['min'],payload['bounds']['max']])).all():
                payload['bounds']={'min':[-50,-50,-2],'max':[50,50,3]}
            cleaning_links=''.join(f'<section><h2>{escape(record["frame_name"])}</h2><img style="max-width:100%" src="{record["sample_token"]}/point_filter_comparison.png"><p>保留 {record["point_filter"]["kept"]} / 原语义支持 {record["point_filter"]["input"]}；移除 {record["point_filter"]["removed"]}</p></section>' for record in frame_manifest)
            (scene_dir/'cleaning/index.html').write_text('<!doctype html><meta charset="utf-8"><title>Instance point cleaning</title><h1>逐实例清洗对比</h1>'+cleaning_links,encoding='utf-8')
            from tasks.nuscenes_sam3.python.scene_bev_map import build_scene_maps
            manifest={'scene':scene_name,'classes':class_rows(names,display_names,colors),'metadata':payload['metadata'],'bounds':payload['bounds'],'frames':frame_manifest,
                      'occupancy_grid':{'npz':'occupancy_grid.npz','metadata_file':'occupancy_grid.json',
                                        'resolution':occupancy['resolution'],'origin':occupancy['origin'],
                                        'fused':{key:occupancy['fused'][key] for key in ('voxel_count','point_count')}}}
            manifest['unassigned_occupancy_grid']={'npz':'unassigned_occupancy_grid.npz',
                'metadata_file':'unassigned_occupancy_grid.json','metadata':unassigned['metadata'],
                'resolution':unassigned['resolution'],'origin':unassigned['origin'],
                'fused':{key:unassigned['fused'][key] for key in ('voxel_count','point_count')}}
            manifest['occupancy_flat']={'npz':'occupancy_flat.npz','metadata_file':'occupancy_flat.json','metadata':flat['metadata'],
                                        'resolution':flat['resolution'],'plane_z_scene':flat['plane_z_scene']}
            payload['bev_map']=build_scene_maps(scene_dir,manifest,map_resolution,occupancy_flat=flat)
            manifest['bev_map']=payload['bev_map']['metadata']
            encoded=json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('<','\\u003c')
            (scene_dir/'viewer.html').write_text(template.replace('__SCENE_DATA__',encoded),encoding='utf-8')
            save_json(scene_dir/'manifest.json',manifest)
            state['scenes'].append({'name':scene_name,'frames':len(records),'dataset_frames':len(all_scene_samples),
                                    'viewer':f'{scene_name}/viewer.html','complete':payload['metadata']['complete_scene']})
        write_index(output,state)
        state['status']='completed'
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error));raise
    finally:save_json(output/'run_config.json',state)
    print(f'Scene timeline index: {output/"index.html"}',flush=True)
    return state


def write_index(output: Path,state: dict):
    links=''.join(f'<li><a href="{escape(scene["viewer"],quote=True)}">{escape(scene["name"])}</a> · <a href="{escape(scene["name"],quote=True)}/cleaning/index.html">清洗对比</a> — 已导出 {scene["frames"]} / 场景总帧 {scene["dataset_frames"]}；{("完整场景" if scene["complete"] else "部分帧")}</li>' for scene in state['scenes'])
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>nuScenes scene viewers</title><style>body{font:17px system-ui;max-width:1000px;margin:40px auto;padding:20px}li{margin:22px 0}a{color:#1764c1}</style><h1>nuScenes 场景时间轴 / Scene timelines</h1><p>每个场景独立 HTML：BEV点多边形路面 / Polygon surfaces、染色点云、时间轴、全部帧叠加和类别图层。离线打开，无需服务器。</p><ul>'+links+'</ul><p>全部帧叠加包含动态物体多时刻轨迹；多边形路面为预测/几何假设，不是真值。</p>',encoding='utf-8')


def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input-dir',type=Path,help='Existing six-class run root or one frame; omit to run six-class SAM3 first')
    p.add_argument('--dataset-dir',type=Path,default=ROOT/'datasets/nuscenes')
    p.add_argument('--version',choices=('v1.0-mini','v1.0-trainval'),default='v1.0-mini')
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--samples',type=int,default=None,help='Frame count; default all saved frames with input-dir, otherwise 1; 0 means all selected')
    p.add_argument('--start-index',type=int,default=0)
    p.add_argument('--scenes',nargs='+');p.add_argument('--split',choices=('mini_val','mini_train','all'),default='mini_val')
    p.add_argument('--bev-resolution',type=float,default=None,help='Polygon grid cell size in meters; saved metadata or 0.2 for generation')
    p.add_argument('--bev-roi-size',type=float,default=None,help='Polygon square ROI size in meters; saved metadata or 100 for generation')
    p.add_argument('--max-edge',type=float,default=None,help='Polygon max triangle edge; default reuse source configuration')
    p.add_argument('--map-resolution',type=float,default=.2,help='Common scene BEV Map cell size, meters')
    p.add_argument('--plane-classes',nargs='+',default=None,help='Names rendered as planes; default saved choice or sidewalk road')
    p.add_argument('--compute-grid-projection',action='store_true',help='Project saved SAM3 masks onto global Z=0 ground cells')
    p.add_argument('--grid-projection-samples',type=int,default=3,help='Samples per cell axis (1..8)')
    p.add_argument('--grid-projection-max-distance',type=float,default=80.,help='Maximum ground projection distance, meters')
    p.add_argument('--occupancy-resolution',type=float,default=.5,help='3D occupancy voxel size in meters')
    p.add_argument('--classes-file',type=Path,default=ROOT/'tasks/nuscenes_sam3/config/six_classes.txt')
    add_filter_args(p)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    return p


def main(argv=None):
    args=parser().parse_args(argv);ensure_environment()
    filter_config=config_from_args(args)
    if args.samples is not None and args.samples<0 or args.start_index<0:raise ValueError('samples/start-index must be nonnegative')
    output=(args.output_dir or ROOT/'outputs/nuscenes_scene_viewer'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):raise FileExistsError(output)
    if args.input_dir:
        if args.start_index:raise ValueError('Saved input exports all selected frames in timestamp order; use --samples to limit')
        return 0 if export_scenes(args.input_dir,output,args.dataset_dir,args.scenes,args.samples or 0,max_edge=args.max_edge,map_resolution=args.map_resolution,plane_classes=args.plane_classes,point_filter=filter_config,bev_resolution=args.bev_resolution,bev_roi_size=args.bev_roi_size,compute_grid_projection=args.compute_grid_projection,grid_projection_samples=args.grid_projection_samples,grid_projection_max_distance=args.grid_projection_max_distance,occupancy_resolution=args.occupancy_resolution,version=args.version) else 1
    dataset=CameraFrames(args.dataset_dir,version=args.version)
    available=dataset.frames(args.split,args.scenes)
    if args.start_index>=len(available):raise ValueError('No selected frames')
    count=args.samples if args.samples is not None else 1
    if count==0:count=len(available)-args.start_index
    from tasks.nuscenes_sam3.python.six_class import main as run_six
    segmentation=output/'segmentation'
    command=['--dataset-dir',str(args.dataset_dir),'--version',args.version,'--samples',str(count),'--start-index',str(args.start_index),
             '--split',args.split,'--device',args.device,'--precision','bf16' if args.device=='cuda' else 'fp32',
             '--no-scene-viewer','--output-dir',str(segmentation)]
    command+=['--classes-file',str(args.classes_file),
              '--bev-resolution',str(args.bev_resolution if args.bev_resolution is not None else .2),
              '--bev-roi-size',str(args.bev_roi_size if args.bev_roi_size is not None else 100.)]
    if args.plane_classes:command+=['--plane-classes',*args.plane_classes]
    command+=['--point-filter',filter_config['algorithm']]
    for key in ('sor_k','sor_std','dbscan_eps','dbscan_min_samples','cluster_policy','min_cluster_size','cluster_ratio'):
        command+=['--'+key.replace('_','-'),str(filter_config[key])]
    if args.scenes:command+=['--scenes',*args.scenes]
    if args.max_edge is not None:command+=['--max-edge',str(args.max_edge)]
    run_six(command)
    export_scenes(segmentation,output/'scenes',args.dataset_dir,args.scenes,0,max_edge=args.max_edge,map_resolution=args.map_resolution,plane_classes=args.plane_classes,point_filter=filter_config,bev_resolution=args.bev_resolution,bev_roi_size=args.bev_roi_size,compute_grid_projection=args.compute_grid_projection,grid_projection_samples=args.grid_projection_samples,grid_projection_max_distance=args.grid_projection_max_distance,occupancy_resolution=args.occupancy_resolution,version=args.version)
    print(f'Complete new run: {output/"scenes/index.html"}',flush=True)
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (RuntimeError,ValueError,FileExistsError,FileNotFoundError) as error:
        print(f'Error: {error}',file=sys.stderr);raise SystemExit(1)

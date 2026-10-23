#!/usr/bin/env python
"""Rasterize saved scene surfaces on one metric grid and fuse frame votes."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime
from html import escape
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from shared.python.runtime import ROOT,ensure_environment,save_json


# Saved scene IDs must never be reinterpreted using the current text configuration.
LEGACY_CLASSES=[
    {'id':cid,'name':name,'label':label,'color':color}
    for cid,name,label,color in zip(range(7),
        ('unassigned','sidewalk','road','pedestrian','vehicle','bike','rider'),
        ('未分配 / unassigned','人行道 / sidewalk','道路 / road','行人 / pedestrian','车辆 / vehicle','两轮车 / bike','骑乘者 / rider'),
        ((255,255,255),(210,70,180),(210,155,20),(35,110,225),(200,55,45),(40,160,85),(125,65,210)))
]


def scene_taxonomy(manifest):
    """Read the saved taxonomy and plane selection, with a fixed legacy fallback."""
    classes=manifest.get('classes',LEGACY_CLASSES)
    if not isinstance(classes,list) or not classes:raise ValueError('Scene classes must be a nonempty list')
    ids=[]
    for entry in classes:
        cid=entry.get('id')
        if isinstance(cid,bool) or not isinstance(cid,int) or not 0<=cid<=254 or cid in ids:
            raise ValueError('Scene class IDs must be unique integers in0..254')
        ids.append(cid)
        color=entry.get('color',[255,255,255])
        if len(color)!=3 or not all(isinstance(v,(int,float)) and np.isfinite(v) and 0<=v<=255 for v in color):
            raise ValueError('Scene colors must contain three channels in0..255')
    planes=manifest.get('metadata',{}).get('plane_class_ids',[1,2])
    if not isinstance(planes,(list,tuple)) or any(isinstance(cid,bool) or not isinstance(cid,int) or cid==0 or cid not in ids for cid in planes) or len(set(planes))!=len(planes):
        raise ValueError('Plane IDs must be unique nonzero saved scene class IDs')
    return classes,list(planes)


def encode_runs(labels):
    """Compact nonzero linear runs (start, length, class_id), little-endian uint32."""
    flat=labels.reshape(-1)
    changes=np.flatnonzero(np.r_[True,flat[1:]!=flat[:-1],True])
    starts,ends=changes[:-1],changes[1:]
    selected=flat[starts]!=0
    return np.column_stack((starts[selected],ends[selected]-starts[selected],flat[starts[selected]])).astype('<u4')


def grid_spec(bounds,resolution=.2):
    bounds=np.asarray(bounds,dtype=float)
    if bounds.shape!=(2,2) or not np.isfinite(bounds).all() or not np.isfinite(resolution) or resolution<=0:
        raise ValueError('Finite ordered XY bounds and positive map resolution required')
    if np.any(bounds[1]<bounds[0]):raise ValueError('Map bounds must be ordered')
    low=np.floor(bounds[0]/resolution)*resolution;high=np.ceil(bounds[1]/resolution)*resolution
    high=np.maximum(high,low+resolution)
    size=np.rint((high-low)/resolution).astype(int)
    if np.any(size<1) or int(size[0])*int(size[1])>16000000:raise ValueError('BEV map exceeds16 million cells; increase --map-resolution')
    return {'width':int(size[0]),'height':int(size[1]),'resolution':float(resolution),
            'x_min':float(low[0]),'y_min':float(low[1]),'x_max':float(high[0]),'y_max':float(high[1]),
            'orientation':'+x right,+y up','origin':'scene first ego yaw XY',
            'labels':{'0':'unassigned/no surface evidence','1':'sidewalk','2':'road'}}


def raster_surface(source_labels,source_to_scene,source_resolution,roi_size,grid,source_height=0.):
    """Inverse-sample a source plane grid at every scene-map cell center.

    The XY affine map is exact for the same flattened calibrated plane used by
    the scene mesh. No splatting, extra interpolation, dilation or hole filling.
    """
    if source_labels.ndim!=2 or source_labels.dtype!=np.uint8 or np.any(source_labels==255):raise ValueError('Surface labels must be uint8 HxW in0..254')
    matrix=np.asarray(source_to_scene,float)
    if matrix.shape!=(4,4) or not np.isfinite(matrix).all():raise ValueError('Finite4x4 plane transform required')
    if source_resolution<=0 or roi_size<=0 or not np.isfinite([source_resolution,roi_size,source_height]).all():raise ValueError('Invalid source grid size')
    if source_labels.shape[0]!=source_labels.shape[1] or not np.isclose(source_labels.shape[0]*source_resolution,roi_size):raise ValueError('Source square grid dimensions mismatch')
    affine=matrix[:2,:2];offset=matrix[:2,3]+matrix[:2,2]*source_height
    if abs(np.linalg.det(affine))<1e-8:raise ValueError('Plane XY transform is singular')
    width,height=grid['width'],grid['height'];resolution=grid['resolution']
    # Evaluate in row chunks to bound memory at large map sizes.
    output=np.zeros((height,width),np.uint8)
    x=grid['x_min']+(np.arange(width)+.5)*resolution
    inverse=np.linalg.inv(affine)
    n=source_labels.shape[0]
    for first in range(0,height,128):
        last=min(first+128,height)
        y=grid['y_max']-(np.arange(first,last)+.5)*resolution
        xx,yy=np.meshgrid(x,y)
        local=(np.stack((xx,yy),-1)-offset)@inverse.T
        col=np.floor((local[...,0]+roi_size/2)/source_resolution).astype(np.int64)
        row=np.floor((roi_size/2-local[...,1])/source_resolution).astype(np.int64)
        valid=(col>=0)&(col<n)&(row>=0)&(row<n)
        block=output[first:last];block[valid]=source_labels[row[valid],col[valid]]
    return output


def fuse_maps(frame_maps,class_ids=None):
    """Vote on saved class IDs; ties always choose the smallest numeric ID."""
    if not frame_maps:raise ValueError('At least one frame map required')
    shape=frame_maps[0].shape
    observed=set()
    for labels in frame_maps:
        if labels.ndim!=2 or labels.shape!=shape or labels.dtype!=np.uint8 or np.any(labels==255):
            raise ValueError('Aligned uint8 surface maps in0..254 required')
        observed.update(int(cid) for cid in np.unique(labels) if cid)
    if class_ids is None:class_ids=sorted(observed) or [1,2]
    class_ids=list(class_ids)
    if any(isinstance(cid,(bool,np.bool_)) or not isinstance(cid,(int,np.integer)) or not 1<=cid<=254 for cid in class_ids) or len(set(class_ids))!=len(class_ids):
        raise ValueError('Vote class IDs must be unique integers in1..254')
    if observed-set(class_ids):raise ValueError('Surface maps contain IDs outside vote classes')
    ids=np.asarray(class_ids,np.uint8)
    votes=np.zeros((len(ids),*shape),np.uint32)
    for labels in frame_maps:
        for index,cid in enumerate(ids):votes[index]+=(labels==cid)
    count=votes.sum(0,dtype=np.uint32)
    if len(ids):
        maximum=votes.max(0)
        winners=(votes==maximum)&(maximum>0)
        labels=np.where(count>0,np.where(winners,ids[:,None,None],255).min(0),0).astype(np.uint8)
        conflict=np.count_nonzero(votes,axis=0)>=2
        tie=np.count_nonzero(winners,axis=0)>=2
    else:
        labels=np.zeros(shape,np.uint8);conflict=np.zeros(shape,bool);tie=np.zeros(shape,bool)
    return {'labels':labels,'observation_count':count,'class_votes':votes,'class_ids':ids,
            'conflict':conflict,'tie':tie}


def map_payload(labels,conflict_cells=0):
    ids,counts=np.unique(labels[labels!=0],return_counts=True)
    return {'runs':base64.b64encode(encode_runs(labels).tobytes()).decode('ascii'),
            'cells':int(np.count_nonzero(labels)),'conflict_cells':int(conflict_cells),
            'class_cells':{str(int(cid)):int(count) for cid,count in zip(ids,counts)}}


def _flat_occupancy_records(payload,plane_ids,frame_count):
    """Validate sparse XY records without changing the producer's class winners."""
    if not isinstance(payload,dict):raise ValueError('occupancy_flat must be a mapping')
    try:
        resolution=float(payload['resolution']);origin=np.asarray(payload['origin'],dtype=float)
        plane_z=float(payload['plane_z_scene'])
    except (KeyError,TypeError,ValueError,OverflowError) as error:
        raise ValueError('Invalid occupancy_flat calibration') from error
    if not np.isfinite(resolution) or resolution<=0 or origin.shape!=(2,) or not np.isfinite(origin).all() or not np.isfinite(plane_z):
        raise ValueError('Finite occupancy_flat origin, plane height and positive resolution required')
    frames=payload.get('frames')
    if not isinstance(frames,(list,tuple)) or len(frames)!=frame_count:
        raise ValueError('occupancy_flat frames must align with manifest frames')
    records=[]
    for record in [*frames,payload.get('fused')]:
        if not isinstance(record,dict):raise ValueError('occupancy_flat records must be mappings')
        indices=np.asarray(record.get('indices'));ids=np.asarray(record.get('class_ids'));counts=np.asarray(record.get('point_counts'))
        # Empty records serialize as [] and therefore lose their array shape/dtype.
        if indices.size==0:indices=np.empty((0,2),np.int64)
        if ids.size==0:ids=np.empty(0,np.uint8)
        if counts.size==0:counts=np.empty(0,np.uint64)
        if indices.ndim!=2 or indices.shape[1]!=2 or not np.issubdtype(indices.dtype,np.integer):
            raise ValueError('occupancy_flat indices must have integer shape (N, 2)')
        n=len(indices)
        if ids.shape!=(n,) or not np.issubdtype(ids.dtype,np.integer) or not np.isin(ids,plane_ids).all():
            raise ValueError('occupancy_flat class_ids must use saved plane IDs')
        if counts.shape!=(n,) or not np.issubdtype(counts.dtype,np.integer) or np.any(counts<0):
            raise ValueError('occupancy_flat point_counts must be nonnegative integers aligned with indices')
        if len({tuple(key) for key in indices})!=n:raise ValueError('occupancy_flat indices must be unique')
        if record.get('cell_count')!=n or record.get('point_count')!=sum(int(value) for value in counts):
            raise ValueError('occupancy_flat record totals must match sparse arrays')
        records.append({'indices':indices,'class_ids':ids.astype(np.uint8),'point_counts':counts.astype(np.uint64)})
    return resolution,origin,plane_z,records


def raster_occupancy_flat(record,source_resolution,origin,grid):
    """Look up each common-grid center in its exact half-open sparse XY cell."""
    lookup={tuple(key):i for i,key in enumerate(record['indices'])}
    width,height=grid['width'],grid['height'];resolution=grid['resolution']
    labels=np.zeros((height,width),np.uint8);counts=np.zeros((height,width),np.uint64)
    x=grid['x_min']+(np.arange(width)+.5)*resolution
    ix=np.floor((x-origin[0])/source_resolution).astype(np.int64)
    for first in range(0,height,128):
        last=min(first+128,height)
        y=grid['y_max']-(np.arange(first,last)+.5)*resolution
        iy=np.floor((y-origin[1])/source_resolution).astype(np.int64)
        selected=np.fromiter((lookup.get((int(col),int(row)),-1) for row in iy for col in ix),
                             dtype=np.int64,count=(last-first)*width).reshape(last-first,width)
        valid=selected>=0
        labels[first:last][valid]=record['class_ids'][selected[valid]]
        counts[first:last][valid]=record['point_counts'][selected[valid]]
    return labels,counts


def build_scene_maps(scene_dir,manifest,resolution=.2,source_scene_dir=None,occupancy_flat=None):
    """Build saved surface maps and optionally rasterize authoritative flat occupancy."""
    scene_dir=Path(scene_dir);source=Path(source_scene_dir or scene_dir)
    classes,plane_ids=scene_taxonomy(manifest)
    bounds=[];available_methods=None;frame_class_heights={}
    flat_records=None
    if occupancy_flat is not None:
        flat_resolution,flat_origin,flat_z,flat_records=_flat_occupancy_records(occupancy_flat,plane_ids,len(manifest['frames']))
        for sparse in flat_records:
            if len(sparse['indices']):
                low=flat_origin+sparse['indices'].min(0)*flat_resolution
                high=flat_origin+(sparse['indices'].max(0)+1)*flat_resolution
                if not np.isfinite([low,high]).all():raise ValueError('occupancy_flat cell bounds must be finite')
                bounds.extend([low,high])
        flat_heights={str(cid):flat_z for cid in plane_ids}
    for record in manifest['frames']:
        with np.load(source/f"{record['sample_token']}.npz",allow_pickle=False) as archive:
            methods={method for method in ('polygon','grid_projection','ipm') if f'{method}_labels' in archive}
            if 'polygon' not in methods:raise ValueError('Saved scene requires polygon_labels')
            if available_methods is None:available_methods=methods
            elif methods!=available_methods:raise ValueError('Saved scene frames have different surface methods')
            for method in methods:
                for cid in plane_ids:
                    triangles=archive[f'{method}_{cid}_triangles_scene']
                    if len(triangles):
                        bounds.extend([triangles[:,:2].min(0),triangles[:,:2].max(0)])
                        if method!='grid_projection':
                            height=float(np.median(triangles[:,2]))
                            if not np.isfinite(height):raise ValueError('Surface display heights must be finite')
                            frame_class_heights.setdefault(method,{}).setdefault(cid,[]).append(height)
    bounds=np.asarray(bounds)
    if not len(bounds):bounds=np.array([[-50,-50],[50,50]],float)
    grid=grid_spec(np.array([bounds.min(0),bounds.max(0)]),resolution)
    grid['labels']={'0':'unassigned/no surface evidence',**{str(c['id']):c.get('name',str(c['id'])) for c in classes if c['id'] in plane_ids}}
    grid['plane_class_ids']=plane_ids
    scene_to_global=np.array(manifest['metadata']['scene_to_global'],float)
    global_to_scene=np.linalg.inv(scene_to_global)
    frames=[];maps={method:[] for method in ('polygon','grid_projection','ipm') if method in (available_methods or {'polygon'})}
    destination=scene_dir/'bev_map';destination.mkdir(parents=True,exist_ok=False)
    for frame_index,record in enumerate(manifest['frames']):
        token=record['sample_token'];entry={};numeric={}
        with np.load(source/f'{token}.npz',allow_pickle=False) as archive:
            for method in maps:
                labels=archive[f'{method}_labels']
                if labels.ndim!=2 or labels.dtype!=np.uint8 or np.any(labels==255) or not np.isin(labels,[0,*plane_ids]).all():
                    raise ValueError(f'{method} surface labels must use saved plane IDs')
                if method in ('ipm','grid_projection'):
                    key=f'{method}_to_global'
                    source_to_global=record[key] if key in record else archive[key]
                    transform=global_to_scene@np.asarray(source_to_global,float);height=0.
                else:
                    transform=global_to_scene@archive['lidar_to_global']
                    # Polygon XY uses its per-class native sensor height before global flattening.
                    # Classes may lie at different elevations; rasterize each source class separately.
                    combined=np.zeros((grid['height'],grid['width']),np.uint8)
                    for cid in plane_ids:
                        points=archive['xyz_lidar'];selected=(archive['filtered_labels'] if 'filtered_labels' in archive else archive['labels'])==cid
                        height=float(np.median(points[selected,2])) if selected.any() else 0.
                        class_source=np.where(labels==cid,cid,0).astype(np.uint8)
                        raster=raster_surface(class_source,transform,record['resolution_m'],record['roi_size_m'],grid,height)
                        combined[(combined==0)&(raster!=0)]=raster[(combined==0)&(raster!=0)]
                    current=combined
                if method in ('ipm','grid_projection'):current=raster_surface(labels,transform,record['resolution_m'],record['roi_size_m'],grid,height)
                maps[method].append(current);entry[method]=map_payload(current);numeric[f'{method}_labels']=current
        if flat_records is not None:
            current,counts=raster_occupancy_flat(flat_records[frame_index],flat_resolution,flat_origin,grid)
            entry['occupancy_flat']={**map_payload(current),'class_heights_scene':flat_heights}
            numeric.update(occupancy_flat_labels=current,occupancy_flat_point_counts=counts,
                           occupancy_flat_plane_z_scene=np.asarray(flat_z))
        np.savez_compressed(destination/f'{token}.npz',**numeric)
        frames.append(entry)
    fused_payload={};summary={}
    palette=np.full((256,3),255,np.uint8)
    for entry in classes:palette[entry['id']]=entry.get('color',[255,255,255])
    palette[0]=255
    for method,frame_maps in maps.items():
        fused=fuse_maps(frame_maps,plane_ids)
        np.savez_compressed(destination/f'{method}_fused.npz',**fused)
        Image.fromarray(palette[fused['labels']]).save(destination/f'{method}_map.png')
        Image.fromarray(np.where(fused['conflict'],255,0).astype(np.uint8)).save(destination/f'{method}_conflicts.png')
        fused_payload[method]=map_payload(fused['labels'],fused['conflict'].sum())
        # Display-only flattening: each nonempty frame contributes one median,
        # regardless of mesh size. Heights never affect semantic votes.
        # 仅用于显示展平：非空帧各贡献一个中位数，高度不影响语义投票。
        fused_payload[method]['class_heights_scene']=(
            {str(cid):float(global_to_scene[2,3]) for cid in plane_ids} if method=='grid_projection' else
            {str(cid):float(np.median(heights)) for cid,heights in frame_class_heights.get(method,{}).items()})
        summary[method]={'assigned_cells':int((fused['labels']!=0).sum()),'conflict_cells':int(fused['conflict'].sum()),
                         'tie_cells':int(fused['tie'].sum()),'total_frame_votes':int(fused['observation_count'].sum()),
                         'max_frame_observations':int(fused['observation_count'].max()),
                         'class_cells':fused_payload[method]['class_cells']}
    if flat_records is not None:
        # The producer already fused 3D voxel winners before collapsing XY.
        # 生产端先融合三维体素赢家再展平XY；此处保留该权威结果，不再按帧投票。
        labels,counts=raster_occupancy_flat(flat_records[-1],flat_resolution,flat_origin,grid)
        np.savez_compressed(destination/'occupancy_flat_fused.npz',labels=labels,point_counts=counts,
                            class_ids=np.asarray(plane_ids,np.uint8),plane_z_scene=np.asarray(flat_z),
                            class_heights_scene=np.asarray([flat_z]*len(plane_ids),float),
                            source_resolution=np.asarray(flat_resolution),source_origin=flat_origin)
        Image.fromarray(palette[labels]).save(destination/'occupancy_flat_map.png')
        fused_payload['occupancy_flat']={**map_payload(labels),'class_heights_scene':flat_heights}
        summary['occupancy_flat']={'assigned_cells':int(np.count_nonzero(labels)),
                                   'class_cells':fused_payload['occupancy_flat']['class_cells'],
                                   'source_cell_count':len(flat_records[-1]['indices']),
                                   'source_point_count':int(occupancy_flat['fused']['point_count']),
                                   'fusion':'Authoritative supplied fused sparse occupancy; no map frame re-vote'}
    metadata={'grid':grid,'methods':summary,'frames':len(frames),'classes':classes,'plane_class_ids':plane_ids,
              'display_heights':'Display-only flattening: observed surfaces use the median of nonempty per-frame class triangle Z medians; '
                                'projected ground uses global Z=0 transformed into scene coordinates. Semantic votes are unchanged. '
                                '仅显示展平：观测平面取各非空帧类别三角形Z中位数的中位数；投影地面为全局Z=0转换到场景坐标；语义投票不变。',
              'sampling':'Scene XY metric cell centers inverse-sampled into each exact source plane grid; nearest source cell, no added fill',
              'fusion':'One class vote per map cell per frame; largest count wins; exact ties choose the smallest class ID; zero is no evidence and abstains',
              'observation_count':'Count of frames with assigned surface label, not sensor/ground-truth observations',
              'limits':'Derived prediction map; conflict mask records cells receiving at least two different plane classes across time. Unknown is not free space. '
                       'BEV collapses height for display/map, original3D coordinates remain unchanged.'}
    if flat_records is not None:
        metadata['occupancy_flat']={'resolution':flat_resolution,'origin':flat_origin.tolist(),
                                    'plane_z_scene':flat_z,'class_heights_scene':flat_heights,
                                    'metadata':occupancy_flat.get('metadata',{}),
                                    'sampling':'Common grid centers use floor((scene XY - origin) / source resolution); sparse cells are half-open; no added fill',
                                    'fusion':'Rasterized from supplied fused record; source voxel winners are authoritative; no frame map vote',
                                    'point_counts':'Source sparse-cell point support repeated at sampled map centers; not frame observation counts',
                                    'display_heights':'All saved plane classes use fixed plane_z_scene / 所有保存的平面类别均使用固定场景平面高度'}
        metadata['fusion']+='; occupancy_flat uses the authoritative supplied fused sparse record without a frame re-vote'
        metadata['display_heights']+=' Flat occupancy uses fixed plane_z_scene for every plane class. 展平占据对所有平面类别使用固定场景平面高度。'
    save_json(destination/'metadata.json',metadata)
    return {'grid':grid,'frames':frames,'fused':fused_payload,'metadata':metadata}


def upgrade_viewers(input_dir,output_dir,resolution=.2):
    source=Path(input_dir).resolve();output=Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):raise FileExistsError(output)
    scene_paths=[source] if (source/'manifest.json').is_file() else sorted(path.parent for path in source.glob('*/manifest.json'))
    if not scene_paths:raise FileNotFoundError('No exported scene manifest found')
    template=(ROOT/'tasks/nuscenes_sam3/web/scene_viewer.html').read_text()
    output.mkdir(parents=True,exist_ok=True);scenes=[]
    for original in scene_paths:
        manifest=json.loads((original/'manifest.json').read_text())
        html=(original/'viewer.html').read_text();start=html.index('const DATA = ')+len('const DATA = ')
        payload,end=json.JSONDecoder().raw_decode(html[start:])
        target=output/manifest['scene'];target.mkdir()
        numeric_source=Path(manifest.get('numeric_scene_source',str(original)))
        maps=build_scene_maps(target,manifest,resolution,numeric_source)
        payload['bev_map']=maps
        payload['classes']=maps['metadata']['classes']
        payload['plane_class_ids']=maps['metadata']['plane_class_ids']
        payload['metadata']={**payload.get('metadata',{}),'plane_class_ids':payload['plane_class_ids'],
                             'surface_methods':list(maps['fused'])}
        manifest={**manifest,'classes':payload['classes'],'metadata':{**manifest.get('metadata',{}),'plane_class_ids':payload['plane_class_ids']}}
        encoded=json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('<','\\u003c')
        (target/'viewer.html').write_text(template.replace('__SCENE_DATA__',encoded),encoding='utf-8')
        save_json(target/'manifest.json',{**manifest,'numeric_scene_source':str(numeric_source),'bev_map':maps['metadata']})
        scenes.append({'name':manifest['scene'],'viewer':f"{manifest['scene']}/viewer.html",'frames':len(manifest['frames'])})
        print(f"{manifest['scene']}: BEV map {maps['grid']['width']}x{maps['grid']['height']} @ {resolution:g}m; {len(manifest['frames'])} frames",flush=True)
    links=''.join(f'<li><a href="{escape(row["viewer"],quote=True)}">{escape(row["name"])}</a> — {row["frames"]} frames</li>' for row in scenes)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Scene BEV Maps</title><h1>场景 BEV Map / Scene maps</h1><p>打开场景后点击BEV Map，可查看单帧或全帧融合栅格地图。</p><ul>'+links+'</ul>',encoding='utf-8')
    save_json(output/'run_config.json',{'status':'completed','input_dir':str(source),'map_resolution':resolution,'scenes':scenes})
    print(f'BEV map viewers: {output/"index.html"}',flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path)
    p.add_argument('--map-resolution',type=float,default=.2)
    args=p.parse_args(argv);ensure_environment()
    if not np.isfinite(args.map_resolution) or args.map_resolution<=0:raise ValueError('map-resolution must be finite positive')
    output=args.output_dir or ROOT/'outputs/nuscenes_bev_map'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    upgrade_viewers(args.input_dir,output,args.map_resolution)
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,RuntimeError,FileNotFoundError,FileExistsError) as error:
        print(f'Error: {error}',file=sys.stderr);raise SystemExit(1)

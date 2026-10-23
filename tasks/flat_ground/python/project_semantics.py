#!/usr/bin/env python
"""Camera-only SAM3 semantics inverse-warped onto an assumed global z=0 plane."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

from shared.python.runtime import ROOT, ensure_environment, save_json
from shared.python.sam3 import Sam3Segmenter, make_palette, overlay, draw_text
from tasks.nuscenes_sam3.python.six_class import NAMES, RiderMask, categories_from_txt
from tasks.nuscenes_sam3.python.semantic_schema import saved_schema,resolve_planes,taxonomy_from_categories,palette_for
from tasks.flat_ground.python.geometry import pose_matrix, ego_yaw_frame, bev_ground_grid, project_ground_to_camera

CAMERAS = ('CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT')


class CameraFrames:
    """Read only nuScenes camera metadata; no LiDAR data/labels/tables required."""
    def __init__(self, root: Path, version: str = 'v1.0-mini'):
        if version not in ('v1.0-mini','v1.0-trainval','v1.0-test'):
            raise ValueError(f'Unsupported nuScenes version {version!r}; use v1.0-mini, v1.0-trainval or v1.0-test')
        self.root=Path(root)
        self.version=version
        self.tables={}
        for name in ('scene','sample','sample_data','sensor','calibrated_sensor','ego_pose'):
            rows=json.loads((self.root/version/f'{name}.json').read_text())
            self.tables[name]={row['token']:row for row in rows}
            if len(rows)!=len(self.tables[name]):raise ValueError(f'Duplicate {name} tokens')
        self.channels={}
        for row in self.tables['sample_data'].values():
            if not row['is_key_frame']:continue
            sensor=self.tables['sensor'][self.tables['calibrated_sensor'][row['calibrated_sensor_token']]['sensor_token']]
            name=sensor['channel']
            if name in CAMERAS:
                channels=self.channels.setdefault(row['sample_token'],{})
                if name in channels:raise ValueError('Duplicate camera keyframe')
                channels[name]=row

    def frames(self, split='mini_val', scenes=None):
        splits={'mini_val':{'scene-0103','scene-0916'},'mini_train':{'scene-0061','scene-0553','scene-0655','scene-0757','scene-0796','scene-1077','scene-1094','scene-1100'}}
        available={row['name'] for row in self.tables['scene'].values()}
        if split not in ('all',*splits):raise ValueError('Unsupported split; use all, mini_train or mini_val')
        if split!='all' and self.version!='v1.0-mini':raise ValueError(f'{split} requires version v1.0-mini')
        chosen=available if split=='all' else available & splits[split]
        if scenes:
            if set(scenes)-available:raise ValueError('Unknown scene name')
            chosen &= set(scenes)
        result=[]
        for row in self.tables['sample'].values():
            scene=self.tables['scene'][row['scene_token']]['name']
            if scene in chosen:result.append({'sample_token':row['token'],'timestamp':row['timestamp'],'scene_name':scene})
        return sorted(result,key=lambda row:(row['scene_name'],row['timestamp'],row['sample_token']))

    def cameras(self, sample_token):
        channels=self.channels.get(sample_token,{})
        if set(CAMERAS)-channels.keys():raise ValueError(f'Sample {sample_token} lacks six camera keyframes')
        result=[]
        for name in CAMERAS:
            row=channels[name];cal=self.tables['calibrated_sensor'][row['calibrated_sensor_token']]
            ego=self.tables['ego_pose'][row['ego_pose_token']]
            path=self.root/row['filename']
            if not path.is_file():raise FileNotFoundError(path)
            result.append({'camera':name,'camera_token':row['token'],'sample_token':sample_token,
                           'timestamp':row['timestamp'],'image_path':path,'ego_pose':ego,
                           'camera_to_global':pose_matrix(ego)@pose_matrix(cal),
                           'intrinsic':np.asarray(cal['camera_intrinsic'],dtype=np.float64)})
        return result


def fuse_ground_views(label_grids, geometry_grids, project_ids=(1,2)):
    """Select a nonzero ground class by most fronto-parallel (nearest-plane) view.

    camera_count counts geometric visibility even when its semantic mask is
    unassigned. support_count counts accepted class evidence, not LiDAR density.
    Stable camera order breaks equal-quality ties; no new class is invented.
    """
    if not label_grids or len(label_grids)!=len(geometry_grids):raise ValueError('Aligned camera grids required')
    shape=label_grids[0].shape
    labels=np.zeros(shape,dtype=np.uint8);source=np.full(shape,-1,dtype=np.int16)
    quality=np.full(shape,-np.inf,dtype=np.float64)
    camera_count=np.zeros(shape,dtype=np.uint8);support_count=np.zeros(shape,dtype=np.uint8)
    for index,(semantic,geometry) in enumerate(zip(label_grids,geometry_grids)):
        if semantic.shape!=shape or semantic.dtype!=np.uint8:raise ValueError('Camera label grids must be aligned uint8')
        valid=geometry['valid']
        if valid.shape!=shape:raise ValueError('Geometry shape mismatch')
        accepted=valid & np.isin(semantic,project_ids)
        camera_count += valid.astype(np.uint8);support_count += accepted.astype(np.uint8)
        update=accepted & (geometry['incidence']>quality)
        labels[update]=semantic[update];source[update]=index;quality[update]=geometry['incidence'][update]
    return {'labels':labels,'source_camera':source,'camera_count':camera_count,'support_count':support_count,
            'selected_incidence':np.where(source>=0,quality,0).astype(np.float32)}


def color_image(labels,palette):
    pixels=palette[labels].copy();pixels[(labels==0)|(labels==255)]=255
    return Image.fromarray(pixels)


def render_comparison(images,palette,title,names=None):
    names=names or NAMES
    n=images[0][1].width
    panel=Image.new('RGB',(n*len(images),n+100+((len(names)-1+2)//3)*24),'white');draw=ImageDraw.Draw(panel)
    for i,(label,image) in enumerate(images):
        panel.paste(image,(i*n,30));draw_text(draw,(i*n+8,8),label)
    draw_text(draw,(10,n+40),title)
    draw_text(draw,(10,n+65),'Assumed plane, not LiDAR evidence or verified surface. No semantic/geometry accuracy metric.')
    for index,i in enumerate(cid for cid in names if cid):
        x=12+(index%3)*n;y=n+92+(index//3)*24
        draw.rectangle((x,y,x+15,y+15),fill=tuple(int(v) for v in palette[i]));draw_text(draw,(x+23,y),f'{i}: {names[i]}')
    return panel


def project_frame(output, cameras, masks, plane_z=0., roi_size=100., resolution=.2, min_depth=.5,max_distance=80.,project_ids=(1,2),names=None):
    names=names or NAMES
    project_ids=resolve_planes(names,default_ids=project_ids)
    output=Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):raise FileExistsError(output)
    if len(cameras)!=6 or len(masks)!=6 or tuple(camera['camera'] for camera in cameras)!=CAMERAS:
        raise ValueError('Six aligned original camera masks in canonical order are required')
    xy,grid=bev_ground_grid(roi_size,resolution)
    reference=next(camera for camera in cameras if camera['camera']=='CAM_FRONT')
    global_from_bev=ego_yaw_frame(reference['ego_pose'],plane_z)
    geometries=[];label_grids=[]
    for camera,mask in zip(cameras,masks):
        with Image.open(camera['image_path']) as image:size=image.size
        if mask.shape!=(size[1],size[0]) or mask.dtype!=np.uint8 or np.any(mask>=len(names)):raise ValueError('Full-resolution masks must match saved taxonomy')
        geometry=project_ground_to_camera(xy,global_from_bev,camera['camera_to_global'],camera['intrinsic'],size,
                                          min_depth=min_depth,max_distance=max_distance)
        semantic=np.zeros(xy.shape[:2],dtype=np.uint8)
        valid=geometry['valid'];uv=geometry['pixel_xy'][valid]
        semantic[valid]=mask[uv[:,1],uv[:,0]]
        label_grids.append(semantic);geometries.append(geometry)
    fused=fuse_ground_views(label_grids,geometries,project_ids)
    palette=palette_for(names)
    output.mkdir(parents=True,exist_ok=True)
    save_json(output/'classes.json',[{'id':cid,'name':name,'color_rgb':palette[cid].tolist(),
              'projected_by_default':cid in project_ids} for cid,name in names.items()])
    per_camera=output/'cameras';per_camera.mkdir()
    records=[]
    for camera,mask,semantic,geometry in zip(cameras,masks,label_grids,geometries):
        name=camera['camera']
        selected=np.where(np.isin(semantic,project_ids),semantic,0).astype(np.uint8)
        color_image(selected,palette).save(per_camera/f'{name}_ipm.png')
        Image.fromarray(mask).save(per_camera/f'{name}_class_ids.png')
        save_json(per_camera/f'{name}_source.json', {'camera_token':camera['camera_token'],
                  'sample_token':camera['sample_token'], 'image_path':str(camera['image_path'])})
        Image.fromarray(semantic).save(per_camera/f'{name}_sampled_bev_ids.png')
        np.savez_compressed(per_camera/f'{name}_geometry.npz',**geometry,
                            camera_to_global=camera['camera_to_global'],intrinsic=camera['intrinsic'])
        # Grid-wise plane hypothesis reprojected onto source image: selected source pixels show the sampled evidence.
        with Image.open(camera['image_path']) as source:image=source.convert('RGB')
        shaded=overlay(image,mask,palette,.5)
        panel=Image.new('RGB',(1780,420),'white');draw=ImageDraw.Draw(panel)
        panel.paste(image.resize((640,360)),(0,30));panel.paste(shaded.resize((640,360)),(640,30))
        panel.paste(color_image(selected,palette).resize((360,360),Image.Resampling.NEAREST),(1350,30))
        for x,text in ((8,'Original'),(648,'SAM3 six-class segmentation'),(1358,'z=0 ground IPM')):draw_text(draw,(x,8),text)
        draw_text(draw,(8,395),name+' | only configured ground classes are projected; horizon/behind-camera and out-of-range cells remain blank')
        panel.save(per_camera/f'{name}_comparison.jpg',quality=94)
        records.append({'camera':name,'camera_token':camera['camera_token'],'timestamp':camera['timestamp'],
                        'image_path':str(camera['image_path']),'geometric_cells':int(geometry['valid'].sum()),
                        'ground_semantic_cells':int((selected!=0).sum()),'camera_height_above_plane_m':float(camera['camera_to_global'][2,3]-plane_z)})
    Image.fromarray(fused['labels']).save(output/'bev_class_ids.png')
    final=color_image(fused['labels'],palette);final.save(output/'bev_ground_ipm.png')
    count_rgb=np.full((*fused['camera_count'].shape,3),255,dtype=np.uint8)
    count_palette=np.array([[255,255,255],[200,235,250],[130,205,245],[70,160,230],[30,115,210],[15,75,175],[5,40,120]],dtype=np.uint8)
    count_rgb=count_palette[np.minimum(fused['camera_count'],6)]
    count_image=Image.fromarray(count_rgb);count_image.save(output/'bev_camera_coverage.png')
    source_rgb=np.full((*fused['labels'].shape,3),255,dtype=np.uint8)
    for i in range(6):source_rgb[fused['source_camera']==i]=tuple(int(v) for v in make_palette()[i+1])
    source_image=Image.fromarray(source_rgb);source_image.save(output/'bev_source_camera.png')
    render_comparison([('Semantic ground plane IPM',final),('Geometric camera coverage (1..6 views)',count_image),
                       ('Selected semantic evidence camera',source_image)],palette,
                      f'{roi_size:g}mx{roi_size:g}m | {resolution:g}m/cell | global z={plane_z:g}m | +X forward (right in PNG), +Y left (up in PNG)',names).save(output/'bev_comparison.png')
    np.savez_compressed(output/'ipm_grid.npz',**fused,local_xy=xy,global_from_bev=global_from_bev,
                        camera_names=np.array([camera['camera'] for camera in cameras]),plane_z=np.array(plane_z))
    metadata={**grid,'plane_z_global_m':plane_z,'reference_camera':'CAM_FRONT','reference_timestamp':reference['timestamp'],
              'bev_frame':'horizontal yaw-aligned ego XY centered at CAM_FRONT ego position; local z=0 is global plane_z',
              'sampling':'BEV cell center -> world z=plane_z -> camera pixel nearest neighbor, no LiDAR points',
              'project_class_ids':list(project_ids),'project_class_names':[names[i] for i in project_ids],
              'fusion':'Accepted ground label from greatest plane-ray incidence (more fronto-parallel); camera order breaks ties',
              'valid_intersection':'camera depth positive/min_depth, inside original image, camera slant distance<=max_distance',
              'max_camera_distance_m':max_distance,'min_camera_depth_m':min_depth,'cameras':records,
              'semantic_cells':{names[i]:int((fused['labels']==i).sum()) for i in project_ids},
              'geometric_covered_cells':int((fused['camera_count']>0).sum()),'unassigned_cells':int((fused['labels']==0).sum()),
              'evaluation':None,'limitations':'Flat-plane hypothesis only; global nuScenes z is a local map coordinate, not geodetic sea-level altitude. '
                  'Slope, sidewalk curb height, occlusion and calibration/segmentation error distort ground labels. '
                  'Projecting elevated pedestrian/vehicle/bike/rider masks to the plane stretches their silhouettes; default excludes them. '
                  'No convex fill, triangle interpolation, LiDAR depth or ground-truth labels are used.'}
    save_json(output/'ipm_metadata.json',metadata)
    return metadata


def load_saved_masks(root,cameras):
    masks=[]
    token=cameras[0]['sample_token']
    for camera in cameras:
        name=camera['camera'];record_path=root/'cameras'/f'{name}_source.json'
        record=json.loads(record_path.read_text())
        if record.get('camera_token')!=camera['camera_token'] or record.get('sample_token')!=token:
            raise ValueError('Saved mask sample/camera token mismatch')
        with Image.open(root/'cameras'/f'{name}_class_ids.png') as image:masks.append(np.asarray(image).copy())
    return masks


def write_page(output,frames):
    cards=[]
    for frame in frames:
        candidates=[('Six-camera semantic ground plane',frame/'bev_comparison.png')]
        candidates +=[(name,frame/'cameras'/f'{name}_comparison.jpg') for name in CAMERAS]
        for title,path in candidates:
            link=path.relative_to(output).as_posix()
            cards.append(f'<section><h2>{escape(title)} — {escape(frame.name)}</h2><a href="{escape(link,quote=True)}"><img loading="lazy" src="{escape(link,quote=True)}"></a></section>')
    (output/'result.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Flat-ground semantic IPM</title><style>body{font:16px system-ui;max-width:1600px;margin:24px auto;background:#f6f7fa;padding:0 16px}img{width:100%}section{background:white;margin:22px 0;padding:12px}</style><h1>Camera-only semantic ground projection (z=0)</h1><p>Assumed horizontal ground, not LiDAR reconstruction or measured free space. White cells lack an accepted ground label. 只按平面假设投影，不代表真实海拔或安全通行。</p>'+''.join(cards),encoding='utf-8')


def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataset-dir',type=Path,default=ROOT/'datasets/nuscenes')
    p.add_argument('--version',choices=('v1.0-mini','v1.0-trainval','v1.0-test'),default='v1.0-mini')
    p.add_argument('--input-dir',type=Path,help='Reuse existing six-class run root or sample directory without SAM3 inference')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'models/sam3/sam3.pt')
    p.add_argument('--classes-file',type=Path,default=ROOT/'tasks/nuscenes_sam3/config/six_classes.txt')
    p.add_argument('--samples',type=int,default=1);p.add_argument('--start-index',type=int,default=0)
    p.add_argument('--split',choices=('mini_val','mini_train','all'),default='mini_val');p.add_argument('--scenes',nargs='+')
    p.add_argument('--plane-z',type=float,default=0.,help='Assumed horizontal plane height in nuScenes global map coordinates, meters')
    p.add_argument('--bev-roi-size',type=float,default=100.);p.add_argument('--bev-resolution',type=float,default=.2)
    p.add_argument('--max-distance',type=float,default=80.,help='Maximum camera-to-ground intersection distance, meters')
    p.add_argument('--plane-classes','--project-classes',dest='plane_classes',nargs='+',default=None,help='Class names to project as planes; default saved choice or sidewalk road')
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda');p.add_argument('--precision',choices=('bf16','fp32','fp16'),default='bf16')
    p.add_argument('--score-threshold',type=float,default=.5);p.add_argument('--mask-threshold',type=float,default=.5)
    p.add_argument('--output-dir',type=Path)
    return p


def main(argv=None):
    args=parser().parse_args(argv);ensure_environment()
    if args.samples<1 or args.start_index<0:raise ValueError('samples>=1,start-index>=0 required')
    if not np.isfinite(args.plane_z):raise ValueError('plane-z must be finite')
    if not np.isfinite(args.max_distance) or args.max_distance<=0:raise ValueError('max-distance must be positive')
    if not all(np.isfinite(v) and 0<=v<=1 for v in (args.score_threshold,args.mask_threshold)):raise ValueError('Thresholds in[0,1] required')
    bev_ground_grid(args.bev_roi_size,args.bev_resolution)
    output=(args.output_dir or ROOT/'outputs/flat_ground'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):raise FileExistsError(output)
    dataset=CameraFrames(args.dataset_dir.resolve(),version=args.version)
    saved=[]
    if args.input_dir:
        source=args.input_dir.resolve()
        saved=[source] if (source/'cameras/CAM_FRONT_source.json').is_file() else sorted(p.parent.parent for p in source.glob('*/cameras/CAM_FRONT_source.json'))
        if not saved:raise FileNotFoundError('No saved six-class camera masks found')
        saved=saved[args.start_index:args.start_index+args.samples]
        selections=[]
        for source in saved:
            info=json.loads((source/'cameras/CAM_FRONT_source.json').read_text())
            sample=dataset.tables['sample'][info['sample_token']]
            selections.append({'sample_token':sample['token'],'scene_name':dataset.tables['scene'][sample['scene_token']]['name'],'source':source})
    else:
        selections=dataset.frames(args.split,args.scenes)[args.start_index:args.start_index+args.samples]
    if not selections:raise ValueError('No selected frames')
    if args.input_dir:names,default_planes=saved_schema(selections[0]['source'],allow_legacy=True)
    else:
        categories=categories_from_txt(args.classes_file.resolve());names=taxonomy_from_categories(categories);default_planes=None
    if args.input_dir:
        for record in selections:
            frame_names,_=saved_schema(record['source'],allow_legacy=True)
            if frame_names!=names:raise ValueError('Selected saved frames have different taxonomies')
    project_ids=resolve_planes(names,args.plane_classes,default_planes)
    segmenter=None
    if not args.input_dir:
        categories=categories_from_txt(args.classes_file.resolve())
        segmenter=Sam3Segmenter(args.checkpoint.resolve(),args.device,args.precision,args.score_threshold,args.mask_threshold)
    output.mkdir(parents=True,exist_ok=True)
    config={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    config.update(status='running',processed_frames=0,lidar_used=False,taxonomy=names,plane_class_ids=list(project_ids),started_at=datetime.now(timezone.utc).isoformat())
    save_json(output/'run_config.json',config)
    completed=[]
    try:
        for i,record in enumerate(selections):
            cameras=dataset.cameras(record['sample_token'])
            if args.input_dir:masks=load_saved_masks(record['source'],cameras)
            else:
                masks=[]
                for camera in cameras:
                    with Image.open(camera['image_path']) as source:image=source.convert('RGB')
                    rider=RiderMask((image.height,image.width),names);prediction,_=segmenter.predict(image,categories,instance_callback=rider)
                    masks.append(rider.apply(prediction))
            stem=f"{record['scene_name']}_{args.start_index+i:06d}_{record['sample_token']}"
            frame=output/stem
            metadata=project_frame(frame,cameras,masks,args.plane_z,args.bev_roi_size,args.bev_resolution,
                                   max_distance=args.max_distance,project_ids=project_ids,names=names)
            print(f"{stem}: {metadata['width']}x{metadata['height']}, classes={metadata['semantic_cells']}",flush=True)
            completed.append(frame);config['processed_frames']+=1
        write_page(output,completed);config['status']='completed'
    except BaseException as error:
        config.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error));raise
    finally:
        config['finished_at']=datetime.now(timezone.utc).isoformat();save_json(output/'run_config.json',config)
    print(f'Flat-ground results: {output/"result.html"}',flush=True);return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,RuntimeError,FileNotFoundError,FileExistsError) as error:
        print(f'Error: {error}',file=sys.stderr);raise SystemExit(1)

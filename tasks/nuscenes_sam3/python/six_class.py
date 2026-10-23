#!/usr/bin/env python
"""Configured SAM3 semantic classes and user-selected plane generation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

from shared.python.nuscenes_data import CAMERAS, NuScenesMini
from shared.python.runtime import ROOT, ensure_environment, save_json
from shared.python.sam3 import Sam3Segmenter, overlay, draw_text
from tasks.nuscenes_sam3.python.pointcloud_semantics import fuse_point_labels
from tasks.nuscenes_sam3.python.pointcloud_bev import save_bev_images
from tasks.nuscenes_sam3.python.ground_regions import ground_planes
from tasks.nuscenes_sam3.python.instance_filter import add_filter_args,config_from_args,clean_point_support
from tasks.nuscenes_sam3.python.point_support import ProjectedInstances
from tasks.nuscenes_sam3.python.semantic_schema import (categories_from_txt, taxonomy_from_categories,
    palette_for, resolve_planes, LEGACY_NAMES, load_display_names, load_colors)

# Compatibility defaults for saved six-class data and existing callers.
NAMES = dict(LEGACY_NAMES)


class RiderMask:
    """Prefer accepted rider over pedestrian and crosswalk over road/unassigned."""
    def __init__(self, shape, names=None):
        self.names=names or NAMES
        ids={name:cid for cid,name in self.names.items()}
        self.rider_id=ids.get('rider');self.person_id=ids.get('pedestrian')
        self.crosswalk_id=ids.get('crosswalk');self.road_id=ids.get('road')
        self.mask = np.zeros(shape,dtype=bool)
        self.crosswalk = np.zeros(shape,dtype=bool)
    def __call__(self, mask, record):
        if record['class_id']==self.rider_id:
            self.mask |= mask
        if record['class_id']==self.crosswalk_id:
            self.crosswalk |= mask
    def apply(self, labels):
        labels=labels.copy()
        if self.rider_id is not None:
            labels[self.mask & ((labels==0)|(labels==self.person_id))]=self.rider_id
        if self.crosswalk_id is not None:
            labels[self.crosswalk & ((labels==0)|(labels==self.road_id))]=self.crosswalk_id
        return labels


def six_ground_plane(xyz, prediction, resolution=.2, roi_size=100., max_edge=6., names=None,plane_ids=None):
    names=names or NAMES
    plane_ids=resolve_planes(names,default_ids=plane_ids)
    if prediction.dtype!=np.uint8 or prediction.shape!=(len(xyz),) or np.any(prediction>=len(names)):
        raise ValueError('Point prediction IDs must match the declared taxonomy')
    result=ground_planes(xyz,prediction,resolution,roi_size,max_edge,ground_ids=plane_ids,num_classes=len(names))
    return {**result,'metadata':{**result['metadata'],'ground_class_ids':list(plane_ids),
               'ground_class_names':[names[cid] for cid in plane_ids],'output_taxonomy':names,
               'internal_mapping':'None: original configured class IDs are retained'}}


def colors(labels,palette):
    rgb=palette[labels].copy();rgb[(labels==0)|(labels==255)]=255
    return Image.fromarray(rgb)


def save_frame_bev(output, xyz, prediction, covered, palette, resolution=.2, roi_size=100., max_edge=6., names=None,plane_ids=None,plane_prediction=None):
    names=names or NAMES;plane_ids=resolve_planes(names,default_ids=plane_ids)
    half=roi_size/2
    meta=save_bev_images(output,xyz,prediction,None,covered,names,palette,{},resolution=resolution,
                         bounds=(-half,-half,half,half))
    with np.load(output/'bev_raster.npz',allow_pickle=False) as archive:
        sparse=archive['prediction'].copy()
    ground=six_ground_plane(xyz,plane_prediction if plane_prediction is not None else prediction,resolution,roi_size,max_edge,names,plane_ids)
    final=sparse.copy()
    replace=(ground['filled']!=0)&((sparse==255)|(sparse==0)|np.isin(sparse,plane_ids))
    final[replace]=ground['filled'][replace]
    inferred=replace & ground['inferred']
    np.savez_compressed(output/'six_class_plane.npz',sparse_prediction=sparse,plane_prediction=final,
                         road_sidewalk_observed=ground['observed'],road_sidewalk_filled=ground['filled'],
                         inferred=inferred,blocked=ground['blocked'])
    colors(sparse,palette).save(output/'bev_six_sparse.png')
    colors(final,palette).save(output/'bev_six_plane.png')
    colors(ground['filled'],palette).save(output/'road_sidewalk_plane.png')
    source=np.full((*final.shape,3),255,dtype=np.uint8)
    source[(sparse!=0)&(sparse!=255)]=(35,140,85)
    source[inferred]=(90,165,235)
    source[ground['blocked']]=(220,65,55)
    Image.fromarray(source).save(output/'plane_sources.png')
    legend_rows=(len(names)-1+2)//3
    footer=100+legend_rows*28
    n=final.shape[0];canvas=Image.new('RGB',(n*3,n+footer),'white');draw=ImageDraw.Draw(canvas)
    panels=[('Configured semantic BEV',colors(sparse,palette)),
            ('Selected planes; other classes observed only',colors(final,palette)),
            ('Green observed / blue inferred / red non-ground',Image.fromarray(source))]
    for i,(title,panel) in enumerate(panels):
        canvas.paste(panel,(i*n,30));draw_text(draw,(i*n+8,7),title)
    for i,cid in enumerate(cid for cid in names if cid):
        x=12+(i%3)*n;y=n+45+(i//3)*28
        draw.rectangle((x,y,x+16,y+16),fill=tuple(int(v) for v in palette[cid]))
        draw_text(draw,(x+24,y),f'{cid}: {names[cid]}')
    draw_text(draw,(12,n+45+legend_rows*28),f'{roi_size:g}m x {roi_size:g}m | {resolution:g}m/cell | {n}x{n} | max fill edge {max_edge:g}m')
    draw_text(draw,(12,n+70+legend_rows*28),'Plane classes: '+', '.join(names[cid] for cid in plane_ids)+'; blue means inferred fill, not GT.')
    canvas.save(output/'six_class_comparison.png')
    details={**ground['metadata'],'final_inferred_cells':int(inferred.sum()),
             'final_class_cells':{name:int((final==cid).sum()) for cid,name in names.items() if cid},
             'semantic_bev':meta,'evaluation':None,
             'reason':'Custom configured taxonomy; lidarseg has no independent rider label; no unchanged 16-class metric is valid.'}
    save_json(output/'plane_metadata.json',details)
    return details


def page(output,frame_dirs,names=None,plane_ids=None,plane_prediction=None):
    names=names or NAMES;plane_ids=resolve_planes(names,default_ids=plane_ids)
    cards=[]
    if (output/'scenes/index.html').is_file():
        cards.append('<section><h2>交互式场景 / Interactive scenes</h2><p><a href="scenes/index.html">打开3D场景时间轴：零水平面 / BEV多边形、分类染色点云、全部帧叠加与图层控制</a></p></section>')
    for frame in frame_dirs:
        images=[('Six-class semantic BEV and ground planes',frame/'six_class_comparison.png')]
        images += [(camera,frame/'cameras'/f'{camera}_comparison.jpg') for camera in CAMERAS]
        for title,path in images:
            target=path.relative_to(output).as_posix()
            cards.append(f'<section><h2>{escape(title)} — {escape(frame.name)}</h2><a href="{escape(target,quote=True)}"><img loading="lazy" src="{escape(target,quote=True)}"></a></section>')
    html=('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
          '<title>Configured SAM3 and ground planes</title><style>body{font:16px system-ui;max-width:1550px;margin:24px auto;background:#f5f6fa;padding:0 16px}img{width:100%}section{background:white;padding:12px;margin:22px 0}</style>'
          '<h1>'+escape(' / '.join(name for cid,name in names.items() if cid))+'</h1><p>Plane classes: '+escape(', '.join(names[cid] for cid in plane_ids))+'. '
          'Other classes remain original semantic evidence. Filled cells are inferred, not measured or guaranteed traversable. No custom-taxonomy accuracy is reported.</p>'+''.join(cards))
    (output/'result.html').write_text(html,encoding='utf-8')


def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataset-dir',type=Path,default=ROOT/'datasets/nuscenes')
    p.add_argument('--version',choices=('v1.0-mini','v1.0-trainval'),default='v1.0-mini')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'models/sam3/sam3.pt')
    p.add_argument('--classes-file',type=Path,default=ROOT/'tasks/nuscenes_sam3/config/six_classes.txt')
    p.add_argument('--samples',type=int,default=1);p.add_argument('--start-index',type=int,default=0)
    p.add_argument('--split',choices=('mini_val','mini_train','all'),default='mini_val')
    p.add_argument('--scenes',nargs='+')
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--precision',choices=('bf16','fp32','fp16'),default='bf16')
    p.add_argument('--score-threshold',type=float,default=.5);p.add_argument('--mask-threshold',type=float,default=.5)
    p.add_argument('--bev-resolution',type=float,default=.2);p.add_argument('--bev-roi-size',type=float,default=100.)
    p.add_argument('--max-edge',type=float,default=6.)
    p.add_argument('--plane-classes',nargs='+',default=None,help='Class names to render as planes; default sidewalk road')
    p.add_argument('--no-scene-viewer',action='store_true',help='Skip offline per-scene 3D timeline with both road-plane methods')
    add_filter_args(p)
    p.add_argument('--output-dir',type=Path)
    return p


def main(argv=None):
    args=parser().parse_args(argv);ensure_environment()
    filter_config=config_from_args(args)
    if args.samples<1 or args.start_index<0:raise ValueError('samples>=1 and start-index>=0 required')
    if not all(np.isfinite(x) and 0<=x<=1 for x in (args.score_threshold,args.mask_threshold)):
        raise ValueError('Thresholds must be in [0,1]')
    if not all(np.isfinite(x) and x>0 for x in (args.bev_resolution,args.bev_roi_size,args.max_edge)):
        raise ValueError('Grid dimensions and max-edge must be finite positive')
    size=args.bev_roi_size/args.bev_resolution
    if not np.isclose(size,round(size),rtol=0,atol=1e-8) or size*size>16000000:
        raise ValueError('ROI/resolution must be integral and at most16M cells')
    output=(args.output_dir or ROOT/'outputs/nuscenes_six_classes'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):raise FileExistsError(output)
    categories=categories_from_txt(args.classes_file.resolve())
    names=taxonomy_from_categories(categories);plane_ids=resolve_planes(names,args.plane_classes)
    display_names=load_display_names(args.classes_file.resolve())
    palette=palette_for(names,load_colors(args.classes_file.resolve()))
    dataset=NuScenesMini(args.dataset_dir.resolve(),version=args.version,require_lidarseg=False);views=dataset.select_views(list(CAMERAS),args.split,args.scenes)
    frames=[views[i:i+6] for i in range(0,len(views),6)]
    if args.start_index>=len(frames):raise ValueError('start-index outside selected samples')
    selected=frames[args.start_index:args.start_index+args.samples]
    segmenter=Sam3Segmenter(args.checkpoint.resolve(),args.device,args.precision,args.score_threshold,args.mask_threshold)
    output.mkdir(parents=True,exist_ok=True)
    run={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    run.update(status='running',processed_frames=0,taxonomy=names,evaluation=None,plane_class_ids=list(plane_ids),
               plane_classes=[names[cid] for cid in plane_ids],point_filter_config=filter_config,
               class_definition={'vehicle':'car/truck/bus/trailer; excludes bikes','bike':'bicycle/motorcycle without rider person',
                                 'rider':'person riding bicycle/motorcycle','pedestrian':'person on foot'},
               overlap_rule='highest instance confidence; rider overrides pedestrian/unassigned; crosswalk overrides road/unassigned when its prompt is accepted',
               started_at=datetime.now(timezone.utc).isoformat())
    save_json(output/'run_config.json',run)
    save_json(output/'classes.json',[{'id':i,'name':name,'color_rgb':palette[i].tolist(),
               **({'display_name':display_names[name]} if name in display_names else {}),
              'prompts':list(next((category.prompts for category in categories if category.id==i),()))} for i,name in names.items()])
    completed=[]
    try:
        for index,frame_views in enumerate(selected):
            first=frame_views[0];full=dataset.load_lidar(first['lidar_token'])
            stem=f"{first['scene_name']}_{args.start_index+index:06d}_{first['sample_token']}"
            frame=output/stem;cameras=frame/'cameras';cameras.mkdir(parents=True)
            observations=[];point_instances=[]
            for view in frame_views:
                started=time.perf_counter()
                with Image.open(view['image_path']) as source:image=source.convert('RGB')
                projection=dataset.projection(view);pixels=projection['pixels']
                support=ProjectedInstances(projection['point_indices'],pixels,view['camera'],len(point_instances))
                rider=RiderMask((image.height,image.width),names)
                def collect(mask,record):
                    rider(mask,record);support(mask,record)
                prediction,instances=segmenter.predict(image,categories,instance_callback=collect)
                prediction=rider.apply(prediction)
                point_instances.extend(support.finalize(prediction))
                observations.append({'point_indices':projection['point_indices'],'prediction':prediction[pixels[:,1],pixels[:,0]]})
                name=view['camera'];Image.fromarray(prediction).save(cameras/f'{name}_class_ids.png')
                colors(prediction,palette).save(cameras/f'{name}_semantics.png')
                blended=overlay(image,prediction,palette,.5);blended.save(cameras/f'{name}_overlay.jpg',quality=94)
                panel=Image.new('RGB',(1920,420),'white');draw=ImageDraw.Draw(panel)
                for j,(title,img) in enumerate((('Original',image),('SAM3 configured classes',blended),('Semantic labels',colors(prediction,palette)))):
                    panel.paste(img.resize((640,360),Image.Resampling.NEAREST if j==2 else Image.Resampling.BILINEAR),(j*640,30))
                    draw_text(draw,(j*640+8,8),title)
                draw_text(draw,(8,398),name+' | '+' / '.join(f'{cid} {label}' for cid,label in names.items() if cid))
                panel.save(cameras/f'{name}_comparison.jpg',quality=94)
                save_json(cameras/f'{name}_instances.json',instances)
                save_json(cameras/f'{name}_source.json',{'camera_token':view['camera_token'],'sample_token':view['sample_token'],'image_path':str(view['image_path'])})
                print(f'{name}: six-class instances={len(instances)}, assigned={(prediction!=0).mean():.1%}, {time.perf_counter()-started:.1f}s',flush=True)
            fused=fuse_point_labels(len(full['xyz']),observations,len(names))
            clean=clean_point_support(full['xyz'],fused['prediction'],point_instances,filter_config,surface_class_ids=plane_ids)
            from tasks.nuscenes_sam3.python.instance_merge import merge_instance_support
            merged=merge_instance_support(clean['instances'],surface_class_ids=plane_ids)
            save_json(frame/'point_instances.json',point_instances)
            save_json(frame/'instance_merge.json',merged)
            save_json(frame/'merged_instances.json',merged['instances'])
            save_json(frame/'point_filter.json',{'metadata':clean['metadata'],'instances':clean['instances']})
            from tasks.nuscenes_sam3.python.instance_boxes import save_instance_boxes
            save_instance_boxes(frame/'instance_boxes.json',full['xyz'],merged['instances'],'native LIDAR_TOP XYZ, meters',surface_class_ids=plane_ids)
            np.savez_compressed(frame/'points.npz',xyz=full['xyz'],**fused,camera_names=np.array([view['camera'] for view in frame_views]),
                                filtered_prediction=clean['prediction'],filter_keep_mask=clean['keep_mask'],removed_point_indices=clean['removed_indices'])
            save_frame_bev(frame,full['xyz'],fused['prediction'],fused['covered'],palette,args.bev_resolution,args.bev_roi_size,args.max_edge,names,plane_ids,clean["prediction"])
            completed.append(frame);run['processed_frames']+=1
        if not args.no_scene_viewer:
            from tasks.nuscenes_sam3.python.scene_export import export_scenes
            export_scenes(output,output/'scenes',args.dataset_dir.resolve(),plane_classes=[names[cid] for cid in plane_ids],point_filter=filter_config,version=args.version)
        page(output,completed,names,plane_ids);run['status']='completed'
    except BaseException as error:
        run.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error));raise
    finally:
        run['finished_at']=datetime.now(timezone.utc).isoformat();save_json(output/'run_config.json',run)
    print(f'Six-class results: {output/"result.html"}',flush=True);return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (RuntimeError,ValueError,FileNotFoundError,FileExistsError) as error:
        print(f'Error: {error}',file=sys.stderr);raise SystemExit(1)

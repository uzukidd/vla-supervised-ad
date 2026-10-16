#!/usr/bin/env python
"""Six-class SAM3 segmentation with only road and sidewalk filled into BEV planes."""
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
from shared.python.sam3 import Category, Sam3Segmenter, parse_class_file, make_palette, overlay, draw_text
from tasks.nuscenes_sam3.python.pointcloud_semantics import fuse_point_labels
from tasks.nuscenes_sam3.python.pointcloud_bev import save_bev_images
from tasks.nuscenes_sam3.python.ground_regions import ground_planes

NAMES = {0:'unassigned',1:'sidewalk',2:'road',3:'pedestrian',4:'vehicle',5:'bike',6:'rider'}
# Internal ground helper mapping only: preserve six-class labels in every public output.
TO_GROUND = np.array([0,13,11,7,4,2,7],dtype=np.uint8)


def categories_from_txt(path: Path) -> list[Category]:
    entries = parse_class_file(path)
    if [name for name,_ in entries] != list(NAMES.values())[1:]:
        raise ValueError('Six-class TXT must retain sidewalk, road, pedestrian, vehicle, bike, rider in this order')
    return [Category(i,name,prompts) for i,(name,prompts) in enumerate(entries,1)]


class RiderMask:
    """Retain rider evidence to disambiguate the overlapping pedestrian prompt."""
    def __init__(self, shape):
        self.mask = np.zeros(shape,dtype=bool)
    def __call__(self, mask, record):
        if record['class_id']==6:
            self.mask |= mask
    def apply(self, labels):
        labels=labels.copy()
        # Do not override bikes, vehicles or surface classes with a person prompt.
        labels[self.mask & ((labels==0)|(labels==3))]=6
        return labels


def six_ground_plane(xyz, prediction, resolution=.2, roi_size=100., max_edge=6.):
    if prediction.dtype!=np.uint8 or prediction.shape!=(len(xyz),) or np.any(prediction>6):
        raise ValueError('Expected six-class uint8 point prediction')
    result=ground_planes(xyz,TO_GROUND[prediction],resolution,roi_size,max_edge)
    observed=np.zeros_like(result['observed']);filled=np.zeros_like(result['filled'])
    for internal,cid in ((13,1),(11,2)):
        observed[result['observed']==internal]=cid
        filled[result['filled']==internal]=cid
    assert not np.any(np.isin(result['filled'],[12,14]))
    return {**result,'observed':observed,'filled':filled,
            'metadata':{**result['metadata'],'ground_class_ids':[1,2],
                        'ground_class_names':['sidewalk','road'],
                        'output_taxonomy':'six_classes',
                        'internal_mapping':'ground helper only: sidewalk->13,road->11,other classes->non-ground veto; public IDs remain 1..6'}}


def colors(labels,palette):
    rgb=palette[labels].copy();rgb[(labels==0)|(labels==255)]=255
    return Image.fromarray(rgb)


def save_frame_bev(output, xyz, prediction, covered, palette, resolution=.2, roi_size=100., max_edge=6.):
    half=roi_size/2
    meta=save_bev_images(output,xyz,prediction,None,covered,NAMES,palette,{},resolution=resolution,
                         bounds=(-half,-half,half,half))
    with np.load(output/'bev_raster.npz',allow_pickle=False) as archive:
        sparse=archive['prediction'].copy()
    ground=six_ground_plane(xyz,prediction,resolution,roi_size,max_edge)
    final=sparse.copy()
    replace=(ground['filled']!=0)&((sparse==255)|(sparse==0)|(sparse==1)|(sparse==2))
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
    n=final.shape[0];canvas=Image.new('RGB',(n*3,n+170),'white');draw=ImageDraw.Draw(canvas)
    panels=[('Six-class observed semantic BEV',colors(sparse,palette)),
            ('Road + sidewalk planes; others observed only',colors(final,palette)),
            ('Green observed / blue inferred / red non-ground',Image.fromarray(source))]
    for i,(title,panel) in enumerate(panels):
        canvas.paste(panel,(i*n,30));draw_text(draw,(i*n+8,7),title)
    for i,cid in enumerate(range(1,7)):
        x=12+(i%3)*n;y=n+45+(i//3)*28
        draw.rectangle((x,y,x+16,y+16),fill=tuple(int(v) for v in palette[cid]))
        draw_text(draw,(x+24,y),f'{cid}: {NAMES[cid]}')
    draw_text(draw,(12,n+110),f'{roi_size:g}m x {roi_size:g}m | {resolution:g}m/cell | {n}x{n} | max fill edge {max_edge:g}m')
    draw_text(draw,(12,n+135),'Only road/sidewalk are interpolated. White means no assigned evidence; blue is inferred fill, not GT.')
    canvas.save(output/'six_class_comparison.png')
    details={**ground['metadata'],'final_inferred_cells':int(inferred.sum()),
             'final_class_cells':{name:int((final==cid).sum()) for cid,name in NAMES.items() if cid},
             'semantic_bev':meta,'evaluation':None,
             'reason':'Custom six-class taxonomy; lidarseg has no independent rider label; no unchanged 16-class metric is valid.'}
    save_json(output/'plane_metadata.json',details)
    return details


def page(output,frame_dirs):
    cards=[]
    for frame in frame_dirs:
        images=[('Six-class semantic BEV and ground planes',frame/'six_class_comparison.png')]
        images += [(camera,frame/'cameras'/f'{camera}_comparison.jpg') for camera in CAMERAS]
        for title,path in images:
            target=path.relative_to(output).as_posix()
            cards.append(f'<section><h2>{escape(title)} — {escape(frame.name)}</h2><a href="{escape(target,quote=True)}"><img loading="lazy" src="{escape(target,quote=True)}"></a></section>')
    html=('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
          '<title>Six-class SAM3 and ground planes</title><style>body{font:16px system-ui;max-width:1550px;margin:24px auto;background:#f5f6fa;padding:0 16px}img{width:100%}section{background:white;padding:12px;margin:22px 0}</style>'
          '<h1>sidewalk / road / pedestrian / vehicle / bike / rider</h1><p>Only road and sidewalk are interpolated into surfaces. '
          'Other classes remain original semantic evidence. Filled cells are inferred, not measured or guaranteed traversable. No custom-taxonomy accuracy is reported.</p>'+''.join(cards))
    (output/'result.html').write_text(html,encoding='utf-8')


def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataset-dir',type=Path,default=ROOT/'datasets/nuscenes')
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
    p.add_argument('--output-dir',type=Path)
    return p


def main(argv=None):
    args=parser().parse_args(argv);ensure_environment()
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
    dataset=NuScenesMini(args.dataset_dir.resolve());views=dataset.select_views(list(CAMERAS),args.split,args.scenes)
    frames=[views[i:i+6] for i in range(0,len(views),6)]
    if args.start_index>=len(frames):raise ValueError('start-index outside selected samples')
    selected=frames[args.start_index:args.start_index+args.samples]
    segmenter=Sam3Segmenter(args.checkpoint.resolve(),args.device,args.precision,args.score_threshold,args.mask_threshold)
    palette=make_palette()
    # Distinct readable colors for the six requested output classes.
    palette[1]=(210,70,180);palette[2]=(210,155,20);palette[3]=(35,110,225)
    palette[4]=(200,55,45);palette[5]=(40,160,85);palette[6]=(125,65,210)
    output.mkdir(parents=True,exist_ok=True)
    run={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    run.update(status='running',processed_frames=0,taxonomy=NAMES,evaluation=None,
               class_definition={'vehicle':'car/truck/bus/trailer; excludes bikes','bike':'bicycle/motorcycle without rider person',
                                 'rider':'person riding bicycle/motorcycle','pedestrian':'person on foot'},
               overlap_rule='highest instance confidence; rider prompt overrides pedestrian/unassigned where both claim a pixel',
               started_at=datetime.now(timezone.utc).isoformat())
    save_json(output/'run_config.json',run)
    save_json(output/'classes.json',[{'id':i,'name':name,'color_rgb':palette[i].tolist(),
              'prompts':list(next((category.prompts for category in categories if category.id==i),()))} for i,name in NAMES.items()])
    completed=[]
    try:
        for index,frame_views in enumerate(selected):
            first=frame_views[0];full=dataset.load_lidar(first['lidar_token'])
            stem=f"{first['scene_name']}_{args.start_index+index:06d}_{first['sample_token']}"
            frame=output/stem;cameras=frame/'cameras';cameras.mkdir(parents=True)
            observations=[]
            for view in frame_views:
                started=time.perf_counter()
                with Image.open(view['image_path']) as source:image=source.convert('RGB')
                rider=RiderMask((image.height,image.width))
                prediction,instances=segmenter.predict(image,categories,instance_callback=rider)
                prediction=rider.apply(prediction)
                projection=dataset.projection(view);pixels=projection['pixels']
                observations.append({'point_indices':projection['point_indices'],'prediction':prediction[pixels[:,1],pixels[:,0]]})
                name=view['camera'];Image.fromarray(prediction).save(cameras/f'{name}_class_ids.png')
                colors(prediction,palette).save(cameras/f'{name}_semantics.png')
                blended=overlay(image,prediction,palette,.5);blended.save(cameras/f'{name}_overlay.jpg',quality=94)
                panel=Image.new('RGB',(1920,420),'white');draw=ImageDraw.Draw(panel)
                for j,(title,img) in enumerate((('Original',image),('SAM3 six classes',blended),('Semantic labels',colors(prediction,palette)))):
                    panel.paste(img.resize((640,360),Image.Resampling.NEAREST if j==2 else Image.Resampling.BILINEAR),(j*640,30))
                    draw_text(draw,(j*640+8,8),title)
                draw_text(draw,(8,398),name+' | 1 sidewalk / 2 road / 3 pedestrian / 4 vehicle / 5 bike / 6 rider')
                panel.save(cameras/f'{name}_comparison.jpg',quality=94)
                save_json(cameras/f'{name}_instances.json',instances)
                save_json(cameras/f'{name}_source.json',{'camera_token':view['camera_token'],'sample_token':view['sample_token'],'image_path':str(view['image_path'])})
                print(f'{name}: six-class instances={len(instances)}, assigned={(prediction!=0).mean():.1%}, {time.perf_counter()-started:.1f}s',flush=True)
            fused=fuse_point_labels(len(full['xyz']),observations,7)
            np.savez_compressed(frame/'points.npz',xyz=full['xyz'],**fused,camera_names=np.array([view['camera'] for view in frame_views]))
            save_frame_bev(frame,full['xyz'],fused['prediction'],fused['covered'],palette,args.bev_resolution,args.bev_roi_size,args.max_edge)
            completed.append(frame);run['processed_frames']+=1
        page(output,completed);run['status']='completed'
    except BaseException as error:
        run.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error));raise
    finally:
        run['finished_at']=datetime.now(timezone.utc).isoformat();save_json(output/'run_config.json',run)
    print(f'Six-class results: {output/"result.html"}',flush=True);return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (RuntimeError,ValueError,FileNotFoundError,FileExistsError) as error:
        print(f'Error: {error}',file=sys.stderr);raise SystemExit(1)

"""Optional cleaning reports preserve full scan arrays and show removed support points."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from shared.python.runtime import save_json
from tasks.nuscenes_sam3.python.point_support import load_instance_records
from tasks.nuscenes_sam3.python.instance_filter import clean_point_support,validate_filter


def clean_saved_support(frame,xyz,prediction,config=None,surface_class_ids=()):
    config=validate_filter(config)
    records=load_instance_records(frame)
    if config['algorithm']!='none' and not records and np.any((prediction!=0)&~np.isin(prediction,list(surface_class_ids))):
        raise ValueError('Saved results contain no point_instances.json; rerun segmentation once to preserve actual instance membership. Semantic class groups are not substituted for instances.')
    return clean_point_support(xyz,prediction,records,config,surface_class_ids=surface_class_ids)


def save_cleaning(output,xyz,original,result,roi_size=100.,resolution=.2):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'point_filter.npz',xyz=xyz,original_prediction=original,
                        filtered_prediction=result['prediction'],keep_mask=result['keep_mask'],removed_point_indices=result['removed_indices'])
    save_json(output/'point_filter.json',{'metadata':result['metadata'],'instances':result['instances']})
    n=round(roi_size/resolution);half=roi_size/2
    inside=((xyz[:,:2]>=-half)&(xyz[:,:2]<half)).all(1)
    xy=np.floor((xyz[:,:2].astype(float)+half)/resolution).astype(int)
    pixels=np.column_stack((np.clip(xy[:,0],0,n-1),n-1-np.clip(xy[:,1],0,n-1)))
    image=Image.new('RGB',(n*2,n+95),'white');draw=ImageDraw.Draw(image)
    for offset,label in ((0,'Original instance-assigned support'),(n,'Retained green / removed red')):
        draw.text((offset+8,8),label,fill='black')
    for i in np.flatnonzero(inside & (original!=0)):
        x,y=(int(v) for v in pixels[i]);draw.point((x,y+30),fill=(40,110,175))
        draw.point((n+x,y+30),fill=(35,155,90) if result['keep_mask'][i] else (225,50,45))
    meta=result['metadata'];draw.text((8,n+45),f"{meta['config']['algorithm']} | input {meta['input']} | retained {meta['kept']} | removed {meta['removed']}",fill='black')
    draw.text((8,n+66),'Filtering applies to LiDAR instance support only. Original arrays and image masks remain available.',fill='black')
    image.save(output/'point_filter_comparison.png')

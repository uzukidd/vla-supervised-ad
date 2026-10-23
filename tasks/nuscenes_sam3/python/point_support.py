"""Sparse instance membership capture aligned to final semantic masks."""
import numpy as np


class ProjectedInstances:
    """Retain original LiDAR indices per camera-local text-prompt detection."""
    def __init__(self,point_indices,pixels,camera,offset=0):
        self.indices=np.asarray(point_indices,dtype=np.int64)
        self.pixels=np.asarray(pixels,dtype=np.int64)
        self.camera=camera;self.offset=offset;self.records=[]
    def __call__(self,mask,record):
        selected=mask[self.pixels[:,1],self.pixels[:,0]]
        local=int(record['instance_id'])
        self.records.append({**record,'camera':self.camera,'camera_instance_id':local,
                             'global_instance_id':self.offset+local,'point_indices':self.indices[selected].tolist()})
    def finalize(self,semantic):
        labels=semantic[self.pixels[:,1],self.pixels[:,0]]
        lookup={int(index):i for i,index in enumerate(self.indices)}
        # Semantic masks may prefer rider/crosswalk over broad person/road; resolve
        # instance ownership only among detections agreeing with the final class.
        owner=np.zeros(len(self.indices),np.int32);best=np.full(len(self.indices),-np.inf,np.float64)
        for row in self.records:
            indices=np.array([lookup[index] for index in row['point_indices']],np.int64)
            valid=(labels[indices]==row['class_id']) & (row['score']>best[indices])
            winners=indices[valid];owner[winners]=row['global_instance_id'];best[winners]=row['score']
        return [{**row,'raw_projected_point_count':len(row['point_indices']),
                 'point_indices':self.indices[owner==row['global_instance_id']].tolist()} for row in self.records]


def load_instance_records(frame):
    import json
    direct=frame/'point_instances.json'
    if direct.is_file():return json.loads(direct.read_text())
    records=[]
    for path in sorted((frame/'cameras').glob('*_point_instances.json')):records+=json.loads(path.read_text())
    return records

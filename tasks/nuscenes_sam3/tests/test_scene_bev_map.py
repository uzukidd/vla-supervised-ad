"""Metric map sampling preserves classes/holes and counts one frame vote per cell."""
import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from tasks.nuscenes_sam3.python.scene_bev_map import encode_runs,grid_spec,raster_surface,raster_occupancy_flat,fuse_maps,map_payload,build_scene_maps,scene_taxonomy


class SceneBevMapTests(unittest.TestCase):
    def test_exact_identity_cell_center_sampling_and_holes(self):
        source=np.array([[1,1,0,2],[1,0,2,2],[0,2,2,0],[0,0,0,0]],np.uint8)
        grid=grid_spec(np.array([[-2,-2],[2,2]]),1.)
        actual=raster_surface(source,np.eye(4),1.,4.,grid)
        np.testing.assert_array_equal(actual,source)
        self.assertEqual(int(actual[1,1]),0)
        self.assertEqual(grid['orientation'],'+x right,+y up')

    def test_translation_rotation_and_source_height_affine(self):
        source=np.zeros((4,4),np.uint8);source[1,2]=1
        grid=grid_spec(np.array([[-4,-4],[4,4]]),1.)
        transform=np.eye(4);transform[:2,3]=[1,0]
        actual=raster_surface(source,transform,1.,4.,grid)
        self.assertEqual(int(actual.sum()),1)
        self.assertEqual(int(actual[3,5]),1)
        transform[:2,:2]=[[0,-1],[1,0]];transform[:2,3]=0
        actual=raster_surface(source,transform,1.,4.,grid)
        self.assertEqual(int(actual[3,3]),1)
        # Tilt affects XY position at the sensor's selected source height.
        theta=.2;transform=np.eye(4)
        transform[:3,:3]=[[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]]
        ground=raster_surface(source,transform,1.,4.,grid,-2.)
        self.assertTrue(np.isin(ground,[0,1]).all())

    def test_majority_counts_ties_and_conflict_are_explicit(self):
        a=np.array([[1,2,0],[0,2,1]],np.uint8)
        b=np.array([[2,2,0],[1,0,1]],np.uint8)
        result=fuse_maps([a,b])
        np.testing.assert_array_equal(result['labels'],[[1,2,0],[1,2,1]])
        np.testing.assert_array_equal(result['observation_count'],[[2,2,0],[1,1,2]])
        self.assertEqual(int(result['conflict'].sum()),1)
        self.assertEqual(int(result['tie'].sum()),1)
        np.testing.assert_array_equal(a,[[1,2,0],[0,2,1]])

    def test_compact_runs_decode_exactly_and_include_no_zero(self):
        labels=np.array([[1,1,0,2],[0,0,2,2]],np.uint8)
        runs=encode_runs(labels);actual=np.zeros(labels.size,np.uint8)
        for start,length,cid in runs:
            self.assertIn(int(cid),(1,2));actual[start:start+length]=cid
        np.testing.assert_array_equal(actual.reshape(labels.shape),labels)
        payload=map_payload(labels,2)
        np.testing.assert_array_equal(np.frombuffer(base64.b64decode(payload['runs']),'<u4').reshape(-1,3),runs)
        self.assertEqual(payload['cells'],5);self.assertEqual(payload['conflict_cells'],2)
        self.assertEqual(encode_runs(np.zeros((3,3),np.uint8)).shape,(0,3))

    def test_dynamic_vote_ids_counts_and_lowest_id_ties(self):
        a=np.array([[3,7,0,3],[7,3,7,0]],np.uint8)
        b=np.array([[7,7,0,3],[3,7,3,0]],np.uint8)
        result=fuse_maps([a,b])
        np.testing.assert_array_equal(result['class_ids'],[3,7])
        np.testing.assert_array_equal(result['labels'],[[3,7,0,3],[3,3,3,0]])
        np.testing.assert_array_equal(result['observation_count'],[[2,2,0,2],[2,2,2,0]])
        self.assertEqual(result['class_votes'].shape,(2,2,4))
        self.assertEqual(int(result['conflict'].sum()),4)
        self.assertEqual(int(result['tie'].sum()),4)
        # Vote channels follow requested IDs; output labels never use channel indices.
        result=fuse_maps([np.array([[8,8,0]],np.uint8),np.array([[2,8,0]],np.uint8)],[8,2])
        np.testing.assert_array_equal(result['class_ids'],[8,2])
        np.testing.assert_array_equal(result['class_votes'],[[[1,2,0]],[[1,0,0]]])
        np.testing.assert_array_equal(result['labels'],[[2,8,0]])
        np.testing.assert_array_equal(result['tie'],[[True,False,False]])
        empty=fuse_maps([np.zeros((1,2),np.uint8)])
        np.testing.assert_array_equal(empty['class_ids'],[1,2])
        self.assertFalse(empty['tie'].any())
        three=fuse_maps([np.array([[8,8,0]],np.uint8),np.array([[2,8,0]],np.uint8),np.array([[3,2,0]],np.uint8)],[8,3,2])
        np.testing.assert_array_equal(three['labels'],[[2,8,0]])
        np.testing.assert_array_equal(three['conflict'],[[True,True,False]])
        np.testing.assert_array_equal(three['tie'],[[True,False,False]])
        np.testing.assert_array_equal(three['observation_count'],[[3,3,0]])
        with self.assertRaises(ValueError):fuse_maps([a],[3])
        with self.assertRaises(ValueError):fuse_maps([a],[3,3,7])

    def test_surface_raster_and_payload_keep_high_dynamic_ids(self):
        source=np.array([[7,254],[0,3]],np.uint8)
        actual=raster_surface(source,np.eye(4),1.,2.,grid_spec([[-1,-1],[1,1]],1.))
        np.testing.assert_array_equal(actual,source)
        self.assertEqual(map_payload(actual)['class_cells'],{'3':1,'7':1,'254':1})
        with self.assertRaises(ValueError):raster_surface(np.array([[255]],np.uint8),np.eye(4),1.,1.,grid_spec([[0,0],[1,1]],1.))

    def test_scene_maps_use_saved_classes_planes_and_palette(self):
        classes=[{'id':cid,'name':name,'label':name,'color':color} for cid,name,color in (
            (0,'unassigned',[255,255,255]),(1,'sidewalk',[210,70,180]),(2,'road',[210,155,20]),
            (3,'crosswalk',[12,34,56]),(7,'rider',[125,65,210]))]
        frame={'sample_token':'frame','ipm_to_global':np.eye(4).tolist(),'resolution_m':1.,'roi_size_m':2.}
        manifest={'classes':classes,'metadata':{'scene_to_global':np.eye(4).tolist(),'plane_class_ids':[3]},'frames':[frame]}
        triangles=np.array([[-1,-1,0],[1,-1,0],[1,1,0]],np.float32)
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source';source.mkdir()
            np.savez(source/'frame.npz',ipm_labels=np.full((2,2),3,np.uint8),polygon_labels=np.full((2,2),3,np.uint8),
                     ipm_3_triangles_scene=triangles,polygon_3_triangles_scene=triangles,
                     xyz_lidar=np.array([[0,0,-1],[0,0,2]],np.float32),labels=np.array([3,7],np.uint8),lidar_to_global=np.eye(4))
            polygon_source=Path(directory)/'polygon-source';polygon_source.mkdir()
            with np.load(source/'frame.npz') as archive:
                np.savez(polygon_source/'frame.npz',**{key:archive[key] for key in archive.files if not key.startswith('ipm')})
            polygon_output=Path(directory)/'polygon-output';polygon_output.mkdir()
            polygon_manifest={**manifest,'frames':[{key:value for key,value in frame.items() if key!='ipm_to_global'}]}
            polygon_maps=build_scene_maps(polygon_output,polygon_manifest,1.,polygon_source)
            self.assertEqual(set(polygon_maps['fused']),{'polygon'})
            self.assertEqual(set(polygon_maps['frames'][0]),{'polygon'})
            self.assertFalse(list((polygon_output/'bev_map').glob('ipm*')))
            output=Path(directory)/'output';output.mkdir()
            maps=build_scene_maps(output,manifest,1.,source)
            self.assertEqual(maps['grid']['labels'],{'0':'unassigned/no surface evidence','3':'crosswalk'})
            self.assertEqual(maps['metadata']['plane_class_ids'],[3])
            self.assertEqual(maps['fused']['polygon']['class_cells'],{'3':4})
            with np.load(output/'bev_map/polygon_fused.npz') as archive:
                np.testing.assert_array_equal(archive['class_ids'],[3])
                self.assertEqual(archive['class_votes'].shape,(1,2,2))
                np.testing.assert_array_equal(archive['labels'],np.full((2,2),3,np.uint8))
            with Image.open(output/'bev_map/polygon_map.png') as image:
                np.testing.assert_array_equal(np.asarray(image)[0,0],[12,34,56])
        legacy,planes=scene_taxonomy({})
        self.assertEqual([entry['id'] for entry in legacy],list(range(7)))
        self.assertEqual(legacy[3]['name'],'pedestrian');self.assertEqual(planes,[1,2])

    def test_projected_maps_use_manifest_or_archive_transform_and_saved_palette(self):
        classes=[{'id':0,'name':'unassigned','color':[255,255,255]},
                 {'id':7,'name':'custom_ground','color':[12,34,56]}]
        scene_to_global=np.eye(4);scene_to_global[:3,3]=[10,20,3]
        ground_to_global=np.eye(4);ground_to_global[:2,3]=[11,20]
        triangles=np.array([[0,-1,-3],[2,-1,-3],[2,1,-3],[0,1,-3]],np.float32)
        record={'sample_token':'frame','resolution_m':1.,'roi_size_m':2.}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.mkdir()
            np.savez(source/'frame.npz',polygon_labels=np.zeros((2,2),np.uint8),
                polygon_7_triangles_scene=np.empty((0,3),np.float32),
                grid_projection_labels=np.array([[7,0],[7,7]],np.uint8),
                grid_projection_7_triangles_scene=triangles,grid_projection_to_global=ground_to_global,
                xyz_lidar=np.empty((0,3),np.float32),labels=np.empty(0,np.uint8),lidar_to_global=scene_to_global)
            for use_manifest in (False,True):
                frame=dict(record)
                if use_manifest:frame['grid_projection_to_global']=ground_to_global.tolist()
                manifest={'classes':classes,'metadata':{'scene_to_global':scene_to_global.tolist(),
                    'plane_class_ids':[7]},'frames':[frame]}
                output=root/str(use_manifest);output.mkdir()
                maps=build_scene_maps(output,manifest,1.,source)
                self.assertEqual(set(maps['fused']),{'polygon','grid_projection'})
                self.assertEqual(maps['grid']['x_min'],0.)
                self.assertEqual(maps['fused']['grid_projection']['class_cells'],{'7':3})
                self.assertEqual(maps['fused']['grid_projection']['class_heights_scene'],{'7':-3.})
                self.assertEqual(maps['fused']['polygon']['class_heights_scene'],{})
                with np.load(output/'bev_map/grid_projection_fused.npz') as archive:
                    np.testing.assert_array_equal(archive['labels'],[[7,0],[7,7]])
                    np.testing.assert_array_equal(archive['class_ids'],[7])
                with Image.open(output/'bev_map/grid_projection_map.png') as image:
                    np.testing.assert_array_equal(np.asarray(image)[0,0],[12,34,56])
                    np.testing.assert_array_equal(np.asarray(image)[0,1],[255,255,255])

    def test_fused_display_heights_use_equal_frame_weight_and_skip_empty_geometry(self):
        classes=[{'id':cid,'name':str(cid),'color':[12,34,56]} for cid in (0,3,7)]
        manifest={'classes':classes,'metadata':{'scene_to_global':np.eye(4).tolist(),
                  'plane_class_ids':[3,7]},'frames':[]}
        triangle=np.array([[-1,-1,0],[1,-1,0],[1,1,0]],np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.mkdir()
            for index,(height,copies) in enumerate(((2.,20),(6.,1),(None,0))):
                token=f'frame{index}'
                manifest['frames'].append({'sample_token':token,'resolution_m':1.,'roi_size_m':2.,
                                          'ipm_to_global':np.eye(4).tolist(),
                                          'grid_projection_to_global':np.eye(4).tolist()})
                labels=np.array([[3,7],[3,7]],np.uint8)
                data={'xyz_lidar':np.array([[0,0,99],[0,0,100]],np.float32),
                      'labels':np.array([3,7],np.uint8),'lidar_to_global':np.eye(4)}
                for method in ('polygon','ipm','grid_projection'):
                    data[f'{method}_labels']=labels if height is not None else np.zeros((2,2),np.uint8)
                    for cid in (3,7):
                        geometry=np.tile(triangle,(copies,1))
                        if len(geometry):
                            geometry[:,2]=height+(cid==7)*10+(method=='ipm')
                            # Preserve median while ensuring actual vertices determine height.
                            geometry[::3,2]-=1;geometry[2::3,2]+=1
                        data[f'{method}_{cid}_triangles_scene']=geometry
                np.savez(source/f'{token}.npz',**data)
            output=root/'output';output.mkdir()
            maps=build_scene_maps(output,manifest,1.,source)
            self.assertEqual(maps['fused']['polygon']['class_heights_scene'],{'3':4.,'7':14.})
            self.assertEqual(maps['fused']['ipm']['class_heights_scene'],{'3':5.,'7':15.})
            self.assertEqual(maps['fused']['grid_projection']['class_heights_scene'],{'3':0.,'7':0.})
            for method in maps['fused']:
                with np.load(output/f'bev_map/{method}_fused.npz') as archive:
                    np.testing.assert_array_equal(archive['labels'],[[3,7],[3,7]])
                    np.testing.assert_array_equal(archive['observation_count'],np.full((2,2),2))
                    np.testing.assert_array_equal(archive['class_votes'],[[[2,0],[2,0]],[[0,2],[0,2]]])
                    self.assertFalse(archive['conflict'].any());self.assertFalse(archive['tie'].any())

    def test_flat_occupancy_common_bounds_centers_fused_authority_heights_and_colors(self):
        classes=[{'id':cid,'name':str(cid),'color':color} for cid,color in
                 ((0,[255,255,255]),(3,[12,34,56]),(7,[78,90,123]))]
        manifest={'classes':classes,'metadata':{'scene_to_global':np.eye(4).tolist(),
                  'plane_class_ids':[7,3]},'frames':[]}
        def sparse(indices,ids,counts):
            return {'indices':indices,'class_ids':ids,'point_counts':counts,
                    'cell_count':len(indices),'point_count':sum(counts)}
        # Both frame maps select 3 in their overlapping cell. Fused voxel winners
        # instead select 7 there and also contain evidence absent from every frame.
        flat={'resolution':1.,'origin':[-.25,.25],'plane_z_scene':-2.75,
              'frames':[sparse([[-1,-1],[1,1]],[3,7],[2,4]),
                        sparse([[-1,-1]],[3],[5])],
              'fused':sparse([[-1,-1],[2,-2]],[7,3],[11,13]),
              'metadata':{'source':'authoritative 3D voxel winners'}}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.mkdir()
            triangle=np.array([[0,0,5],[1,0,5],[1,1,5]],np.float32)
            for index in range(2):
                token=f'frame{index}';manifest['frames'].append(
                    {'sample_token':token,'resolution_m':1.,'roi_size_m':2.})
                np.savez(source/f'{token}.npz',polygon_labels=np.full((2,2),3,np.uint8),
                         polygon_3_triangles_scene=triangle,polygon_7_triangles_scene=np.empty((0,3),np.float32),
                         xyz_lidar=np.array([[0,0,5]],np.float32),labels=np.array([3],np.uint8),lidar_to_global=np.eye(4))
            for resolution in (.5,1.,1.5):
                output=root/str(resolution);output.mkdir()
                with patch('tasks.nuscenes_sam3.python.scene_bev_map.fuse_maps',wraps=fuse_maps) as fuse:
                    maps=build_scene_maps(output,manifest,resolution,source,occupancy_flat=flat)
                self.assertEqual(fuse.call_count,1)  # Polygon alone uses map voting.
                self.assertEqual(set(maps['fused']),{'polygon','occupancy_flat'})
                grid=maps['grid'];self.assertEqual(grid['resolution'],resolution)
                self.assertLessEqual(grid['x_min'],-1.25);self.assertGreaterEqual(grid['x_max'],2.75)
                self.assertLessEqual(grid['y_min'],-1.75);self.assertGreaterEqual(grid['y_max'],2.25)
                def expected(record,key,dtype):
                    lookup={tuple(index):value for index,value in zip(record['indices'],record[key])}
                    return np.asarray([[lookup.get((int(np.floor((grid['x_min']+(col+.5)*resolution+.25))),
                                                    int(np.floor((grid['y_max']-(row+.5)*resolution-.25)))),0)
                                        for col in range(grid['width'])] for row in range(grid['height'])],dtype=dtype)
                wanted=expected(flat['fused'],'class_ids',np.uint8)
                with np.load(output/'bev_map/occupancy_flat_fused.npz') as archive:
                    np.testing.assert_array_equal(archive['labels'],wanted)
                    np.testing.assert_array_equal(archive['point_counts'],expected(flat['fused'],'point_counts',np.uint64))
                    np.testing.assert_array_equal(archive['class_ids'],[7,3])
                    np.testing.assert_array_equal(archive['class_heights_scene'],[-2.75,-2.75])
                    self.assertEqual(float(archive['plane_z_scene']),-2.75)
                    self.assertEqual(float(archive['source_resolution']),1.)
                    np.testing.assert_array_equal(archive['source_origin'],[-.25,.25])
                    self.assertNotIn('class_votes',archive.files)
                frame_labels=[]
                for index,record in enumerate(flat['frames']):
                    with np.load(output/f'bev_map/frame{index}.npz') as archive:
                        frame_labels.append(archive['occupancy_flat_labels'])
                        np.testing.assert_array_equal(frame_labels[-1],expected(record,'class_ids',np.uint8))
                        np.testing.assert_array_equal(archive['occupancy_flat_point_counts'],expected(record,'point_counts',np.uint64))
                    self.assertEqual(maps['frames'][index]['occupancy_flat']['class_heights_scene'],{'7':-2.75,'3':-2.75})
                if resolution==.5:
                    overlap=(frame_labels[0]==3)&(frame_labels[1]==3)
                    self.assertTrue(overlap.any());self.assertTrue(np.all(wanted[overlap]==7))
                    self.assertFalse(np.array_equal(fuse_maps(frame_labels,[7,3])['labels'],wanted))
                payload=maps['fused']['occupancy_flat']
                self.assertEqual(payload['class_heights_scene'],{'7':-2.75,'3':-2.75})
                decoded=np.zeros(wanted.size,np.uint8)
                for start,length,cid in np.frombuffer(base64.b64decode(payload['runs']),'<u4').reshape(-1,3):
                    decoded[start:start+length]=cid
                np.testing.assert_array_equal(decoded.reshape(wanted.shape),wanted)
                palette=np.full((256,3),255,np.uint8);palette[3]=[12,34,56];palette[7]=[78,90,123]
                with Image.open(output/'bev_map/occupancy_flat_map.png') as image:
                    np.testing.assert_array_equal(np.asarray(image),palette[wanted])
                metadata=json.loads((output/'bev_map/metadata.json').read_text())
                self.assertEqual(metadata['occupancy_flat']['plane_z_scene'],-2.75)
                self.assertEqual(metadata['occupancy_flat']['metadata'],flat['metadata'])
                self.assertEqual(metadata['methods']['occupancy_flat']['source_point_count'],24)
                self.assertEqual(metadata['plane_class_ids'],[7,3])

    def test_sparse_nonunit_resolution_half_open_boundaries_and_topdown_chunks(self):
        # Centers at exact source boundaries belong to the cell above/right.
        grid=grid_spec([[-1,-1],[1,1]],.5)
        sparse={'indices':np.array([[-1,-1],[0,0]],np.int64),
                'class_ids':np.array([3,7],np.uint8),'point_counts':np.array([11,13],np.uint64)}
        labels,counts=raster_occupancy_flat(sparse,.5,np.array([-.25,-.25]),grid)
        np.testing.assert_array_equal(labels,[[0,0,0,0],[0,0,0,0],[0,7,0,0],[3,0,0,0]])
        np.testing.assert_array_equal(counts,[[0,0,0,0],[0,0,0,0],[0,13,0,0],[11,0,0,0]])
        # Repeated source sampling crosses the rasterizer's 128-row chunk boundary.
        grid=grid_spec([[-.5,-.5],[.5,.5]],.005)
        sparse={'indices':np.array([[0,0]],np.int64),'class_ids':np.array([7],np.uint8),
                'point_counts':np.array([19],np.uint64)}
        labels,counts=raster_occupancy_flat(sparse,.5,np.array([-.25,-.25]),grid)
        expected=np.zeros((200,200),np.uint8);expected[50:150,50:150]=7
        np.testing.assert_array_equal(labels,expected)
        np.testing.assert_array_equal(counts,np.where(expected!=0,19,0))

    def test_empty_flat_occupancy_and_invalid_saved_plane_ids(self):
        manifest={'classes':[{'id':0},{'id':7}],
                  'metadata':{'scene_to_global':np.eye(4).tolist(),'plane_class_ids':[7]},
                  'frames':[{'sample_token':'frame','resolution_m':1.,'roi_size_m':2.}]}
        empty={'indices':[],'class_ids':[],'point_counts':[],'cell_count':0,'point_count':0}
        flat={'resolution':.3,'origin':[0,0],'plane_z_scene':4.,'frames':[empty],'fused':empty,'metadata':{}}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.mkdir()
            np.savez(source/'frame.npz',polygon_labels=np.zeros((2,2),np.uint8),
                     polygon_7_triangles_scene=np.empty((0,3),np.float32),
                     xyz_lidar=np.empty((0,3),np.float32),labels=np.empty(0,np.uint8),lidar_to_global=np.eye(4))
            output=root/'output';output.mkdir()
            maps=build_scene_maps(output,manifest,1.,source,flat)
            self.assertEqual(maps['fused']['occupancy_flat']['cells'],0)
            self.assertEqual(maps['fused']['occupancy_flat']['class_heights_scene'],{'7':4.})
            invalid={'indices':[[0,0]],'class_ids':[3],'point_counts':[1],'cell_count':1,'point_count':1}
            with self.assertRaisesRegex(ValueError,'saved plane IDs'):
                build_scene_maps(root/'invalid',manifest,1.,source,{**flat,'fused':invalid})
            with self.assertRaisesRegex(ValueError,'align with manifest'):
                build_scene_maps(root/'invalid',manifest,1.,source,{**flat,'frames':[]})

    def test_grid_rounds_outwards_and_limits_memory(self):
        grid=grid_spec(np.array([[-1.01,-2.01],[2.01,1.01]]),.2)
        self.assertAlmostEqual(grid['x_min'],-1.2);self.assertAlmostEqual(grid['y_max'],1.2)
        self.assertEqual(grid['width'],17);self.assertEqual(grid['height'],17)
        for bounds,resolution in (([[-1,-1],[1,1]],0),([[-1,-1],[10000,10000]],.01),([[1,0],[0,1]],.2)):
            with self.assertRaises(ValueError):grid_spec(np.array(bounds),resolution)


if __name__=='__main__':unittest.main()

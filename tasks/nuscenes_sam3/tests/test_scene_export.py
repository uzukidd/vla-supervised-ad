"""Scene coordinate alignment, exact polygon meshing and saved multi-scene exports."""
from pathlib import Path
import base64
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from tasks.nuscenes_sam3.python.scene_export import grid_runs,runs_to_triangles,transform_points,project_camera_points,export_scenes,camera_images,parser,main
from tasks.nuscenes_sam3.python.semantic_schema import palette_for,LEGACY_NAMES
from tasks.flat_ground.python.project_semantics import CAMERAS


class SceneExportTests(unittest.TestCase):
    def assert_occupancy_partition(self, scene_dir):
        html=(scene_dir/'viewer.html').read_text(encoding='utf-8')
        payload=json.loads(html.split('const DATA = ',1)[1].split(';\n',1)[0])
        semantic=payload['occupancy_grid'];unassigned=payload['unassigned_occupancy_grid']
        self.assertEqual(unassigned['resolution'],semantic['resolution'])
        self.assertEqual(unassigned['origin'],semantic['origin'])
        self.assertEqual(len(unassigned['frames']),len(payload['frames']))
        self.assertEqual(unassigned['metadata']['semantic_source'],'aligned_frame_filtered_labels_class_id_0')
        self.assertEqual(unassigned['metadata']['source_class_ids'],[0])
        self.assertIn('filter-removed',unassigned['metadata']['selection_note'])
        total=0
        with np.load(scene_dir/'unassigned_occupancy_grid.npz',allow_pickle=False) as saved:
            self.assertFalse(any('class_ids' in key or 'vote_counts' in key for key in saved.files))
            for i,(entry,known,unknown) in enumerate(zip(payload['frames'],semantic['frames'],unassigned['frames'])):
                with np.load(scene_dir/f"{entry['sample_token']}.npz",allow_pickle=False) as archive:
                    xyz=archive['xyz_scene'];labels=archive['filtered_labels']
                expected,counts=np.unique(np.floor(xyz[labels==0]/unassigned['resolution']).astype(np.int64),axis=0,return_counts=True)
                np.testing.assert_array_equal(saved[f'frame_{i}_indices'],expected)
                np.testing.assert_array_equal(saved[f'frame_{i}_counts'],counts)
                self.assertEqual(unknown['point_count'],int((labels==0).sum()))
                self.assertEqual(known['point_count']+unknown['point_count'],len(xyz))
                self.assertNotIn('class_ids',unknown)
                self.assertNotIn('vote_counts',unknown)
                total+=len(xyz)
            self.assertEqual(int(saved['fused_point_counts'].sum()),unassigned['fused']['point_count'])
        self.assertEqual(semantic['fused']['point_count']+unassigned['fused']['point_count'],total)
        self.assertNotIn('class_ids',unassigned['fused'])
        self.assertEqual(unassigned['metadata']['point_count'],unassigned['fused']['point_count'])
        metadata=json.loads((scene_dir/'unassigned_occupancy_grid.json').read_text(encoding='utf-8'))
        self.assertEqual(metadata['metadata'],unassigned['metadata'])
        self.assertEqual(metadata['fused']['point_count'],unassigned['fused']['point_count'])
        manifest=json.loads((scene_dir/'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['unassigned_occupancy_grid']['npz'],'unassigned_occupancy_grid.npz')
        self.assertEqual(manifest['unassigned_occupancy_grid']['metadata_file'],'unassigned_occupancy_grid.json')
        self.assertEqual(manifest['metadata']['unassigned_occupancy_grid']['metadata'],unassigned['metadata'])
        return payload

    def test_projection_cli_defaults_and_saved_input_forwarding(self):
        defaults=parser().parse_args([])
        self.assertFalse(defaults.compute_grid_projection)
        self.assertEqual(defaults.grid_projection_samples,3)
        self.assertEqual(defaults.grid_projection_max_distance,80.)
        self.assertEqual(defaults.occupancy_resolution,.5)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch('tasks.nuscenes_sam3.python.scene_export.ensure_environment'), \
                 patch('tasks.nuscenes_sam3.python.scene_export.export_scenes',return_value={'status':'completed'}) as export:
                self.assertEqual(main(['--input-dir',str(root/'input'),'--output-dir',str(root/'output'),
                    '--version','v1.0-trainval','--compute-grid-projection','--grid-projection-samples','8','--grid-projection-max-distance','25','--occupancy-resolution','.25']),0)
            self.assertEqual(export.call_args.kwargs['version'],'v1.0-trainval')
            self.assertTrue(export.call_args.kwargs['compute_grid_projection'])
            self.assertEqual(export.call_args.kwargs['grid_projection_samples'],8)
            self.assertEqual(export.call_args.kwargs['grid_projection_max_distance'],25.)
            self.assertEqual(export.call_args.kwargs['occupancy_resolution'],.25)
        for resolution in (0,-1,float('nan'),float('inf'),True):
            with self.assertRaisesRegex(ValueError,'occupancy-resolution'):
                export_scenes(Path('missing'),Path('unused'),Path('unused'),occupancy_resolution=resolution)
        for samples in (0,9,True,1.5):
            with self.assertRaisesRegex(ValueError,'grid-projection-samples'):
                export_scenes(Path('missing'),Path('unused'),Path('unused'),grid_projection_samples=samples)
        for distance in (0,-1,float('nan'),float('inf')):
            with self.assertRaisesRegex(ValueError,'grid-projection-max-distance'):
                export_scenes(Path('missing'),Path('unused'),Path('unused'),grid_projection_max_distance=distance)

    def test_run_mesh_preserves_every_cell_and_holes(self):
        labels=np.array([[1,1,0,2],[1,0,1,2],[0,1,1,0]],np.uint8)
        for cid in (1,2):
            runs=grid_runs(labels,cid)
            restored=np.zeros_like(labels,dtype=bool)
            for row,start,end in runs:restored[row,start:end]=True
            np.testing.assert_array_equal(restored,labels==cid)
            triangles=runs_to_triangles(runs,1.,4.,np.eye(4)).reshape(-1,3,3)
            a=triangles[:,1,:2]-triangles[:,0,:2];b=triangles[:,2,:2]-triangles[:,0,:2]
            areas=np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0])/2
            self.assertAlmostEqual(float(areas.sum()),float((labels==cid).sum()))
        self.assertEqual(grid_runs(np.zeros((2,2),np.uint8),1).shape,(0,3))

    def test_pose_alignment_and_measured_height_preserve_points(self):
        xyz=np.array([[1,2,3],[2,3,5]],np.float32);matrix=np.eye(4);matrix[:3,3]=[10,20,1]
        np.testing.assert_array_equal(transform_points(xyz,matrix),[[11,22,4],[12,23,6]])
        inverse=np.linalg.inv(matrix)
        np.testing.assert_allclose(transform_points(transform_points(xyz,matrix),inverse),xyz)
        points=runs_to_triangles(np.array([[0,0,2]],np.int32),.2,2.,matrix,source_height=-2.,world_height=.15,global_to_scene=inverse)
        self.assertTrue(np.allclose(points[:,2],-.85))
        np.testing.assert_array_equal(xyz,[[1,2,3],[2,3,5]])

    def test_camera_projection_filters_bounds_and_keeps_nearest_pixel(self):
        camera={'camera_to_global':np.eye(4),'intrinsic':np.array([[10.,0.,10.],[0.,10.,10.],[0.,0.,1.]])}
        xyz=np.array([[0.,0.,3.],[.01,0.,2.],[0.,0.,-2.],[100.,0.,3.],[.01,0.,2.],
                      [0.,0.,1.],[-1.8,0.,2.],[2.6,0.,2.],[0.,-1.8,2.],[0.,2.6,2.],
                      [.55,.75,2.]],np.float64)
        indices,pixels,depth=project_camera_points(xyz,camera,(24,24))
        np.testing.assert_array_equal(indices,[1,10])
        np.testing.assert_array_equal(pixels,[[10,10],[12,13]])
        np.testing.assert_array_equal(depth,[2.,2.])
        translated=dict(camera,camera_to_global=np.eye(4),timestamp=2000,
                        ego_pose={'translation':[999.,999.,999.],'rotation':[1.,0.,0.,0.]})
        translated['camera_to_global'][:3,3]=[1.,0.,0.]
        translated_indices,translated_pixels,_=project_camera_points(np.array([[1.,0.,3.]]),translated,(24,24))
        np.testing.assert_array_equal(translated_indices,[0])
        np.testing.assert_array_equal(translated_pixels,[[10,10]])
        for bad_xyz in (np.zeros((2,2)),np.array([[np.nan,0.,1.]])):
            with self.assertRaises(ValueError):project_camera_points(bad_xyz,camera,(24,24))
        with self.assertRaises(ValueError):project_camera_points(np.zeros((1,3)),camera,(0,24))
        for key,invalid in (('camera_to_global',np.zeros((4,4))),('camera_to_global',np.full((4,4),np.inf)),
                            ('camera_to_global',np.eye(3)),('intrinsic',np.zeros((3,3))),
                            ('intrinsic',np.full((3,3),np.nan)),('intrinsic',np.eye(4))):
            with self.subTest(key=key,invalid=invalid),self.assertRaises(ValueError):
                project_camera_points(np.empty((0,3)),dict(camera,**{key:invalid}),(24,24))
        indices,pixels,depth=project_camera_points(np.empty((0,3)),camera,(24,24))
        self.assertEqual(indices.shape,(0,));self.assertEqual(pixels.shape,(0,2));self.assertEqual(depth.shape,(0,))

    def test_camera_projection_uses_camera_timestamp_ego_and_calibration(self):
        from tasks.flat_ground.python.geometry import pose_matrix
        lidar_ego={'translation':[10.,20.,0.],'rotation':[1.,0.,0.,0.]}
        camera_ego={'translation':[11.,20.,0.],'rotation':[1.,0.,0.,0.]}
        calibration={'translation':[.5,0.,0.],'rotation':[1.,0.,0.,0.]}
        camera={'camera_to_global':pose_matrix(camera_ego)@pose_matrix(calibration),
                'ego_pose':camera_ego,'timestamp':2000,'intrinsic':np.array([[10.,0.,10.],[0.,10.,10.],[0.,0.,1.]])}
        world=transform_points(np.array([[1.5,0.,3.]]),pose_matrix(lidar_ego))
        indices,pixels,depth=project_camera_points(world,camera,(24,24))
        np.testing.assert_array_equal(indices,[0]);np.testing.assert_array_equal(pixels,[[10,10]])
        np.testing.assert_array_equal(depth,[3.])

    def test_camera_overlay_uses_saved_class_colors_and_preserves_background(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'original.png'
            pixels=np.full((2,3,3),80,np.uint8);Image.fromarray(pixels).save(path)
            mask=np.array([[0,1,2],[2,0,1]],np.uint8)
            names={0:'unassigned',1:'vehicle',2:'custom_class'}
            cameras=[{'camera':'CAM_FRONT','timestamp':123,'image_path':path}]
            colors={'vehicle':'#123456','custom_class':'#abcdef'}
            images=camera_images(cameras,root,'frame',[mask],names,colors)
            self.assertEqual(images[0]['overlay_src'],'camera_images/frame/CAM_FRONT_overlay.png')
            self.assertNotIn('points_src',images[0]);self.assertNotIn('overlay_points_src',images[0])
            self.assertNotIn('projected_point_count',images[0])
            with Image.open(root/images[0]['overlay_src']) as image:blended=np.asarray(image)
            expected=pixels.copy();active=mask!=0
            expected[active]=(pixels[active]*.5+palette_for(names,colors)[mask[active]]*.5).astype(np.uint8)
            np.testing.assert_array_equal(blended,expected)
            self.assertEqual((root/images[0]['src']).read_bytes(),path.read_bytes())
            with self.assertRaisesRegex(ValueError,'dimensions'):
                camera_images(cameras,root,'bad-frame',[mask[:1]],names)

    def test_camera_images_include_projected_point_products_with_custom_and_unknown_colors(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'camera.png';Image.new('RGB',(24,24),(60,70,80)).save(path)
            camera={'camera':'CAM_FRONT','timestamp':456,'image_path':path,'camera_to_global':np.eye(4),
                    'intrinsic':np.array([[10.,0.,12.],[0.,10.,12.],[0.,0.,1.]])}
            names={0:'unassigned',1:'custom'};colors={'custom':'#123456'}
            points=np.array([[0.,0.,2.],[.4,0.,4.],[0.,0.,-2.]])
            images=camera_images([camera],root,'points',[np.ones((24,24),np.uint8)],names,colors,
                                 xyz_global=points,point_labels=np.array([1,0,1],np.uint8))
            record=images[0]
            self.assertEqual(record['projected_point_count'],2)
            self.assertEqual(record['points_src'],'camera_images/points/CAM_FRONT_points.png')
            self.assertEqual(record['overlay_points_src'],'camera_images/points/CAM_FRONT_overlay_points.png')
            self.assertEqual(record['src'],'camera_images/points/CAM_FRONT.png')
            self.assertEqual(record['overlay_src'],'camera_images/points/CAM_FRONT_overlay.png')
            for key in ('points_src','overlay_points_src'):
                with Image.open(root/record[key]) as image:self.assertEqual(image.size,(24,24))
            with Image.open(root/record['points_src']) as image:
                rendered=np.asarray(image)
                np.testing.assert_array_equal(rendered[0,0],[60,70,80])
                np.testing.assert_array_equal(rendered[12,13],[18,52,86])
                np.testing.assert_array_equal(rendered[12,12],[18,52,86])
                np.testing.assert_array_equal(rendered[12,15],[255,255,255])
            with Image.open(root/record['overlay_points_src']) as image:
                rendered=np.asarray(image)
                np.testing.assert_array_equal(rendered[0,0],[39,61,83])
                np.testing.assert_array_equal(rendered[12,13],[18,52,86])
                np.testing.assert_array_equal(rendered[12,12],[18,52,86])
                np.testing.assert_array_equal(rendered[12,15],[255,255,255])
            for kwargs in ({'xyz_global':points},{'point_labels':np.array([1,0,1])},
                           {'xyz_global':points,'point_labels':np.array([-1,0,1])}):
                with self.assertRaises(ValueError):
                    camera_images([camera],root,'invalid',[np.zeros((24,24),np.uint8)],names,colors,**kwargs)

    def test_multiframe_multiscene_export_builds_semantic_occupancy_from_filtered_support(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.mkdir()
            tables={'sample':{},'scene':{},'sample_data':{},'calibrated_sensor':{},'sensor':{},'ego_pose':{}}
            tables['sensor']['lidar']={'channel':'LIDAR_TOP'}
            tables['calibrated_sensor']['lidar']={'sensor_token':'lidar','rotation':[1,0,0,0],'translation':[0,0,2]}
            camera_sets={}
            xyz=np.array([[0,0,-2],[1,0,-2],[0,1,-2],[1,1,-2],[2,1,-.4],[3,2,1.6],[2,2,.6],[3,1,.6],[-1,-1,-3],[.1,.1,-2]],np.float32)
            labels=np.array([2,2,2,2,4,4,4,4,0,0],np.uint8)
            names=[('scene-a','a0',0),('scene-a','a1',1),('scene-b','b0',0)]
            for scene,token,index in names:
                tables['scene'][scene]={'token':scene,'name':scene}
                tables['sample'][token]={'token':token,'scene_token':scene,'timestamp':1000000+index*500000}
                pose={'rotation':[1,0,0,0],'translation':[index,0,0]}
                tables['ego_pose'][token]=pose
                tables['sample_data'][token]={'token':token,'sample_token':token,'is_key_frame':True,'timestamp':1000000+index*500000,
                    'calibrated_sensor_token':'lidar','ego_pose_token':token}
                frame=source/token;(frame/'cameras').mkdir(parents=True)
                np.savez_compressed(frame/'points.npz',xyz=xyz,prediction=labels,camera_names=np.array(CAMERAS))
                (frame/'point_instances.json').write_text(json.dumps([
                    {'global_instance_id':1,'camera':'CAM_FRONT','class_id':2,'class_name':'road','score':.9,'point_indices':[0,1,2,3]},
                    {'global_instance_id':2,'camera':'CAM_BACK','class_id':4,'class_name':'vehicle','score':.8,'point_indices':[4,5,6,7]},
                    {'global_instance_id':3,'camera':'CAM_FRONT_LEFT','class_id':4,'class_name':'vehicle','score':.9,'point_indices':[4,5,6]}]))
                (frame/'plane_metadata.json').write_text(json.dumps({'roi_size_m':10.,'resolution_m_per_pixel':.2,'max_triangle_edge_m':3.}))
                camera_sets[token]=[]
                for camera in CAMERAS:
                    camera_token=token+camera
                    (frame/'cameras'/f'{camera}_source.json').write_text(json.dumps({'camera_token':camera_token,'sample_token':token}))
                    Image.fromarray(np.full((32,32),2,np.uint8)).save(frame/'cameras'/f'{camera}_class_ids.png')
                    path=root/f'{camera_token}.jpg';Image.new('RGB',(32,32)).save(path)
                    transform=np.eye(4);transform[:3,:3]=np.diag([1,-1,-1]);transform[:3,3]=[index,0,2]
                    camera_sets[token].append({'camera':camera,'camera_token':camera_token,'sample_token':token,'timestamp':1000000+index*500000,
                        'image_path':path,'camera_to_global':transform,'ego_pose':pose,'intrinsic':np.array([[16,0,16],[0,16,16],[0,0,1]],float)})
            class Dataset:
                def __init__(self,*args,**kwargs):self.tables=tables
                def cameras(self,token):return camera_sets[token]
            output=root/'viewer'
            with patch('tasks.nuscenes_sam3.python.scene_export.CameraFrames',Dataset), \
                 patch('tasks.nuscenes_sam3.python.grid_projection.project_ground_cells',side_effect=AssertionError('disabled projection called')) as project:
                state=export_scenes(source,output,root)
            project.assert_not_called()
            self.assertEqual(state['processed_frames'],3)
            self.assertEqual(len(state['scenes']),2)
            self.assertTrue((output/'index.html').exists())
            self.assertFalse(any('ipm' in path.name.lower() for path in output.rglob('*')))
            self.assertNotIn('ipm',json.dumps(state).lower())
            manifest=json.loads((output/'scene-a/manifest.json').read_text())
            self.assertEqual(manifest['metadata']['occupancy_grid']['resolution'],.5)
            with np.load(output/'scene-a/occupancy_grid.npz',allow_pickle=False) as occupancy:
                self.assertEqual(int(occupancy['fused_point_counts'].sum()),2*int(np.count_nonzero(labels)))
                self.assertEqual(int(occupancy['frame_0_counts'].sum()),int(np.count_nonzero(labels)))
                self.assertEqual(set(occupancy['frame_0_class_ids'].tolist()),{2,4})
                self.assertEqual(occupancy['frame_0_class_ids'].dtype,np.uint8)
                self.assertIn('fused_class_ids',occupancy.files)
            occ_json=json.loads((output/'scene-a/occupancy_grid.json').read_text())
            self.assertEqual(occ_json['fused']['point_count'],2*int(np.count_nonzero(labels)))
            self.assertTrue(occ_json['metadata']['semantic_filtering'])
            self.assertNotIn('ipm',json.dumps(manifest).lower())
            self.assertEqual(len(manifest['frames']),2)
            self.assertTrue(manifest['metadata']['complete_scene'])
            self.assertEqual(manifest['frames'][1]['time_s'],.5)
            for token,index in (('a0',0),('a1',1)):
                boxes=json.loads((output/f'scene-a/{token}_boxes.json').read_text())
                self.assertEqual(boxes['metadata']['input_instances'],2)
                self.assertEqual(boxes['metadata']['excluded_surface_class_ids'],[1,2])
                self.assertEqual(boxes['metadata']['excluded_surface_instances'],1)
                self.assertEqual([box['class_id'] for box in boxes['boxes']],[4])
                box=boxes['boxes'][0]
                self.assertEqual(box['member_instance_ids'],[2,3])
                self.assertEqual(box['source_cameras'],['CAM_BACK','CAM_FRONT_LEFT'])
                self.assertEqual(box['point_indices'],[4,5,6,7])
                self.assertTrue(box['merged']);self.assertEqual(box['member_count'],2)
                merge=json.loads((output/f'scene-a/{token}_instance_merge.json').read_text())
                self.assertEqual(merge['metadata']['input_instances'],3)
                self.assertEqual(merge['metadata']['merged_groups'],1)
                self.assertEqual(merge['metadata']['output_groups'],2)
                self.assertEqual(merge['instances'][1]['source_instances'][1]['camera'],'CAM_FRONT_LEFT')
                self.assertEqual(box['geometry_type'],'cuboid')
                self.assertEqual(len(box['corners_scene']),8);self.assertEqual(len(box['edges']),12)
                self.assertAlmostEqual(box['height_m'],2.,places=6)
                np.testing.assert_allclose(box['center_scene'],[index+2.5,1.5,2.6])
                np.testing.assert_allclose(np.asarray(box['corners_scene'])[:4,2],1.6)
                np.testing.assert_allclose(np.asarray(box['corners_scene'])[4:,2],3.6)
                with np.load(output/f'scene-a/{token}.npz',allow_pickle=False) as data:
                    np.testing.assert_array_equal(data['xyz_lidar'],xyz)
                    np.testing.assert_array_equal(data['labels'],labels)
                    np.testing.assert_allclose(data['xyz_scene'],xyz+[index,0,2])
                    self.assertFalse(any(key.startswith('ipm') for key in data.files))
                    self.assertFalse(any(key.startswith('grid_projection') for key in data.files))
                    self.assertTrue(np.isin(data['polygon_labels'],[0,1,2]).all())
                    self.assertTrue(np.isfinite(data['polygon_2_triangles_scene']).all())
            html=(output/'scene-a/viewer.html').read_text()
            self.assertNotIn('__SCENE_DATA__',html)
            self.assertIn('"scene":"scene-a"',html)
            self.assertIn('id="allFrames"',html)
            self.assertIn('id="timeline"',html)
            self.assertNotIn('https://',html)
            payload=self.assert_occupancy_partition(output/'scene-a')
            self.assert_occupancy_partition(output/'scene-b')
            semantic_keys={tuple(key) for key in payload['occupancy_grid']['frames'][0]['indices']}
            unknown_keys={tuple(key) for key in payload['unassigned_occupancy_grid']['frames'][0]['indices']}
            self.assertIn((0,0,0),semantic_keys & unknown_keys)
            self.assertIn((-2,-2,-2),unknown_keys-semantic_keys)
            self.assertNotIn('IPM',html)
            self.assertNotIn('ipmLayer',html)
            self.assertEqual(set(payload['bev_map']['fused']),{'polygon','occupancy_flat'})
            self.assertEqual(set(payload['occupancy_flat']['fused']['class_ids']),{2})
            self.assertEqual(payload['occupancy_flat']['plane_z_scene'],0.)
            self.assertEqual(set(payload['fused_surfaces']['occupancy_flat']),{'1','2'})
            self.assertTrue((output/'scene-a/occupancy_flat.npz').is_file())
            self.assertTrue((output/'scene-a/bev_map/occupancy_flat_map.png').is_file())
            for frame in payload['frames']:
                flat_mesh=frame['surfaces']['occupancy_flat']
                self.assertEqual(flat_mesh['1']['count'],0)
                xyz_flat=np.frombuffer(base64.b64decode(flat_mesh['2']['xyz']),'<f4').reshape(-1,3)
                self.assertTrue(np.all(xyz_flat[:,2]==0.))
                self.assertGreater(len(xyz_flat),0)
                self.assertEqual(set(frame['surfaces']),{'polygon','occupancy_flat'})
                self.assertNotIn('ipm_cells',frame['stats'])
                self.assertEqual({image['camera'] for image in frame['camera_images']},set(CAMERAS))
                for image in frame['camera_images']:
                    original=root/f"{frame['sample_token']}{image['camera']}.jpg"
                    self.assertEqual(image['src'],f"camera_images/{frame['sample_token']}/{image['camera']}.jpg")
                    self.assertEqual((output/'scene-a'/image['src']).read_bytes(),original.read_bytes())
                    self.assertEqual(image['overlay_src'],f"camera_images/{frame['sample_token']}/{image['camera']}_overlay.png")
                    with Image.open(output/'scene-a'/image['overlay_src']) as semantic:
                        self.assertEqual(semantic.size,(32,32))
                        np.testing.assert_array_equal(np.asarray(semantic)[0,0],(palette_for({0:'unassigned',2:'road'})[2]*.5).astype(np.uint8))
                    self.assertEqual((image['width'],image['height']),(32,32))
                    self.assertEqual(image['timestamp_us'],frame['timestamp_us'])
                    self.assertEqual(image['points_src'],f"camera_images/{frame['sample_token']}/{image['camera']}_points.png")
                    self.assertEqual(image['overlay_points_src'],f"camera_images/{frame['sample_token']}/{image['camera']}_overlay_points.png")
                    self.assertGreaterEqual(image['projected_point_count'],0)
                    for product in ('points_src','overlay_points_src'):
                        with Image.open(output/'scene-a'/image[product]) as points_image:self.assertEqual(points_image.size,(32,32))
                self.assertEqual([box['class_id'] for box in frame['bounding_boxes']],[4])
                self.assertEqual(frame['bounding_boxes'][0]['geometry_type'],'cuboid')
            self.assertEqual(manifest['classes'][2]['label'],'道路 / road')
            self.assertEqual(payload['classes'],manifest['classes'])
            projected_output=root/'projected-viewer'
            from tasks.nuscenes_sam3.python.grid_projection import project_ground_cells
            with patch('tasks.nuscenes_sam3.python.scene_export.CameraFrames',Dataset), \
                 patch('tasks.nuscenes_sam3.python.grid_projection.project_ground_cells',wraps=project_ground_cells) as project:
                projected_state=export_scenes(source,projected_output,root,scenes=['scene-a'],
                    bev_resolution=1.,bev_roi_size=4.,compute_grid_projection=True,
                    grid_projection_samples=2,grid_projection_max_distance=20.,map_resolution=1.)
            self.assertEqual(project.call_count,2)
            self.assertEqual(project.call_args.kwargs,{'samples_per_axis':2,'max_distance':20.})
            self.assertEqual(projected_state['grid_projection_samples'],2)
            projected_manifest=json.loads((projected_output/'scene-a/manifest.json').read_text())
            projected_html=(projected_output/'scene-a/viewer.html').read_text()
            projected_payload=json.loads(projected_html.split('const DATA = ',1)[1].split(';\n',1)[0])
            self.assertEqual(projected_payload['metadata']['surface_methods'],['polygon','grid_projection','occupancy_flat'])
            self.assertEqual(set(projected_payload['bev_map']['fused']),{'polygon','grid_projection','occupancy_flat'})
            for index,record in enumerate(projected_manifest['frames']):
                self.assertEqual(record['grid_projection']['plane_z_global_m'],0.)
                self.assertEqual(record['surface_stats']['grid_projection']['samples_per_axis'],2)
                expected=np.eye(4);expected[0,3]=index
                np.testing.assert_allclose(record['grid_projection_to_global'],expected)
                with np.load(projected_output/f"scene-a/{record['sample_token']}.npz") as data:
                    np.testing.assert_array_equal(data['grid_projection_labels'],np.full((4,4),2,np.uint8))
                    np.testing.assert_allclose(data['grid_projection_to_global'],expected)
                    np.testing.assert_array_equal(data['grid_projection_camera_count'],np.full((4,4),6,np.uint8))
                    self.assertTrue((data['grid_projection_sample_count']>=data['grid_projection_support_count']).all())
                    self.assertFalse(data['grid_projection_conflict'].any())
                    triangles=data['grid_projection_2_triangles_scene']
                    self.assertTrue(np.all(triangles[:,2]==0.))
                    np.testing.assert_allclose(triangles[:,:2].min(0),[-2+index,-2])
                    np.testing.assert_allclose(triangles[:,:2].max(0),[2+index,2])
                with np.load(projected_output/f"scene-a/bev_map/{record['sample_token']}.npz") as data:
                    self.assertEqual(int(np.count_nonzero(data['grid_projection_labels'])),16)
            with np.load(projected_output/'scene-a/bev_map/grid_projection_fused.npz') as data:
                self.assertEqual(int(np.count_nonzero(data['labels'])),20)
                np.testing.assert_array_equal(data['class_ids'],[1,2])
            self.assertTrue((projected_output/'scene-a/bev_map/grid_projection_map.png').exists())
            # Saved exports recover labels from classes.json without a prompt file or sidecar.
            (source/'classes.json').write_text(json.dumps([
                {'id':cid,'name':name,'color_rgb':([12,34,56] if name=='road' else palette_for(LEGACY_NAMES)[cid].tolist()),**({'display_name':{'road':'定制道路','vehicle':'机动车辆'}[name]}
                                        if name in ('road','vehicle') else {})}
                for cid,name in LEGACY_NAMES.items()],ensure_ascii=False),encoding='utf-8')
            custom_output=root/'custom-viewer'
            with patch('tasks.nuscenes_sam3.python.scene_export.CameraFrames',Dataset):
                export_scenes(source,custom_output,root,plane_z=float('nan'),max_distance=-1,plane_classes=['vehicle'],bev_resolution=.5,bev_roi_size=12.,occupancy_resolution=.25)
            custom_manifest=json.loads((custom_output/'scene-a/manifest.json').read_text(encoding='utf-8'))
            custom_html=(custom_output/'scene-a/viewer.html').read_text(encoding='utf-8')
            custom_payload=self.assert_occupancy_partition(custom_output/'scene-a')
            self.assertEqual(custom_payload['unassigned_occupancy_grid']['resolution'],.25)
            self.assertEqual(custom_manifest['classes'][2]['label'],'定制道路')
            self.assertEqual(custom_manifest['classes'][2]['color'],[12,34,56])
            self.assertEqual(custom_manifest['frames'][0]['resolution_m'],.5)
            self.assertEqual(custom_manifest['frames'][0]['roi_size_m'],12.)
            with Image.open(custom_output/'scene-a'/custom_payload['frames'][0]['camera_images'][0]['overlay_src']) as image:
                np.testing.assert_array_equal(np.asarray(image)[0,0],[6,17,28])
            self.assertEqual(custom_manifest['classes'][4]['label'],'机动车辆')
            self.assertEqual(custom_manifest['classes'][4]['display_name'],'机动车辆')
            self.assertEqual(custom_manifest['classes'][4]['name'],'vehicle')
            self.assertEqual(custom_manifest['classes'][1]['label'],'人行道 / sidewalk')
            self.assertEqual(custom_payload['classes'],custom_manifest['classes'])
            self.assertEqual(set(custom_payload['occupancy_flat']['fused']['class_ids']),{4})
            self.assertEqual(custom_payload['fused_surfaces']['occupancy_flat'].keys(),{'4'})
            self.assertEqual(custom_payload['plane_class_ids'],[4])
            custom_boxes=json.loads((custom_output/'scene-a/a0_boxes.json').read_text())
            self.assertEqual(custom_boxes['metadata']['excluded_surface_class_ids'],[4])
            self.assertEqual([box['class_id'] for box in custom_boxes['boxes']],[2])
            removed_output=root/'removed-filter-viewer'
            with patch('tasks.nuscenes_sam3.python.scene_export.CameraFrames',Dataset), \
                 patch('tasks.nuscenes_sam3.python.scene_export.camera_images',wraps=camera_images) as export_images:
                export_scenes(source,removed_output,root,scenes=['scene-a'],
                    point_filter={'algorithm':'dbscan','dbscan_min_samples':5})
            self.assertEqual(export_images.call_count,2)
            np.testing.assert_array_equal(export_images.call_args_list[0].kwargs['xyz_global'],xyz+[0,0,2])
            removed_manifest=json.loads((removed_output/'scene-a/manifest.json').read_text())
            removed_payload=self.assert_occupancy_partition(removed_output/'scene-a')
            self.assertGreater(removed_payload['unassigned_occupancy_grid']['frames'][0]['point_count'],int((labels==0).sum()))
            self.assertGreater(removed_manifest['frames'][0]['point_filter']['removed'],0)
            with np.load(removed_output/'scene-a/a0.npz',allow_pickle=False) as frame_data:
                surface=labels==2
                np.testing.assert_array_equal(frame_data['filtered_labels'][surface],labels[surface])
                self.assertTrue((frame_data['filtered_labels'][labels==4]==0).all())
                self.assertEqual(removed_manifest['frames'][0]['point_filter']['surface_class_ids'],[1,2])
                self.assertEqual(removed_manifest['frames'][0]['point_filter']['exempt_surface_points'],int(surface.sum()))
                self.assertTrue(all(record['filter']['status']=='surface_exempt' for record in json.loads((removed_output/'scene-a/cleaning/a0/point_filter.json').read_text())['instances'] if record['class_id']==2))
                expected_labeled_retained=int(np.count_nonzero(frame_data['filtered_labels']))
                np.testing.assert_array_equal(export_images.call_args_list[0].kwargs['point_labels'],frame_data['filtered_labels'])
            with np.load(removed_output/'scene-a/occupancy_grid.npz',allow_pickle=False) as occupancy:
                self.assertEqual(int(occupancy['frame_0_counts'].sum()),expected_labeled_retained)
            with self.assertRaises(FileExistsError):export_scenes(source,output,root)


if __name__=='__main__':unittest.main()

"""Camera-only indexing and same-plane multi-camera semantic evidence fusion."""
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
from PIL import Image

from tasks.flat_ground.python.project_semantics import CameraFrames,fuse_ground_views,load_saved_masks,CAMERAS,project_frame


class FlatGroundTaskTests(unittest.TestCase):
    def test_fusion_uses_only_requested_ground_classes_and_stable_incidence(self):
        a=np.array([[1,2,4],[0,1,0]],np.uint8)
        b=np.array([[2,1,2],[2,6,0]],np.uint8)
        valid=np.array([[1,1,1],[1,1,0]],bool)
        geometry=[{'valid':valid,'incidence':np.array([[.2,.5,.1],[.2,.3,0]],np.float32)},
                  {'valid':valid,'incidence':np.array([[.4,.5,.3],[.2,.9,0]],np.float32)}]
        result=fuse_ground_views([a,b],geometry)
        np.testing.assert_array_equal(result['labels'],[[2,2,2],[2,1,0]])
        np.testing.assert_array_equal(result['source_camera'],[[1,0,1],[1,0,-1]])
        np.testing.assert_array_equal(result['camera_count'],[[2,2,2],[2,2,0]])
        np.testing.assert_array_equal(result['support_count'],[[2,2,1],[1,1,0]])
        np.testing.assert_array_equal(a,[[1,2,4],[0,1,0]])
        all_classes=fuse_ground_views([a,b],geometry,(1,2,3,4,5,6))
        self.assertEqual(int(all_classes['labels'][1,1]),6)

    def test_camera_metadata_does_not_require_lidar_or_lidarseg(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);meta=root/'v1.0-mini';meta.mkdir()
            q=[1,0,0,0]
            tables={'scene':[{'token':'scene','name':'scene-0103'}],
                    'sample':[{'token':'sample','scene_token':'scene','timestamp':10}],
                    'sample_data':[],'sensor':[],'calibrated_sensor':[],
                    'ego_pose':[{'token':'pose','rotation':q,'translation':[100,200,0]}]}
            for i,name in enumerate(CAMERAS):
                filename=f'{name}.jpg';Image.new('RGB',(4,3)).save(root/filename)
                tables['sensor'].append({'token':name,'channel':name,'modality':'camera'})
                tables['calibrated_sensor'].append({'token':name,'sensor_token':name,'rotation':q,'translation':[0,0,1.5],'camera_intrinsic':np.eye(3).tolist()})
                tables['sample_data'].append({'token':name,'sample_token':'sample','is_key_frame':True,'timestamp':10+i,
                    'calibrated_sensor_token':name,'ego_pose_token':'pose','filename':filename})
            for name,rows in tables.items():(meta/f'{name}.json').write_text(json.dumps(rows))
            dataset=CameraFrames(root)
            self.assertEqual(len(dataset.frames()),1)
            cameras=dataset.cameras('sample')
            self.assertEqual(len(cameras),6)
            np.testing.assert_allclose(cameras[0]['camera_to_global'][:3,3],[100,200,1.5])
            self.assertFalse((meta/'lidarseg.json').exists())
            self.assertFalse((root/'samples/LIDAR_TOP').exists())
            self.assertEqual(dataset.version,'v1.0-mini')
            for version in ('v1.0-trainval','v1.0-test'):
                with self.subTest(version=version):
                    version_meta=root/version;version_meta.mkdir()
                    tables['scene'][0]['name']='scene-0001'
                    for name,rows in tables.items():(version_meta/f'{name}.json').write_text(json.dumps(rows))
                    dataset=CameraFrames(root,version=version)
                    self.assertEqual(dataset.version,version)
                    self.assertEqual(dataset.frames('all'),[{'sample_token':'sample','timestamp':10,'scene_name':'scene-0001'}])
                    self.assertEqual(len(dataset.cameras('sample')),6)
                    self.assertFalse((version_meta/'lidarseg.json').exists())
                    for split in ('mini_train','mini_val'):
                        with self.assertRaisesRegex(ValueError,'requires version v1.0-mini'):dataset.frames(split)
                    with self.assertRaisesRegex(ValueError,'Unsupported split'):dataset.frames('train')
                    with self.assertRaisesRegex(ValueError,'Unknown scene name'):dataset.frames('all',['scene-9999'])
            for version in ('v1.0','trainval','../v1.0-mini',''):
                with self.subTest(version=version),self.assertRaisesRegex(ValueError,'Unsupported nuScenes version'):
                    CameraFrames(root,version=version)

    def test_six_camera_projection_saves_numeric_evidence_and_reusable_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cameras=[]
            pose=np.eye(4);pose[:3,:3]=np.diag([1,-1,-1]);pose[:3,3]=[0,0,2]
            for name in CAMERAS:
                image_path=root/f'{name}.png';Image.new('RGB',(32,32),(20,30,40)).save(image_path)
                cameras.append({'camera':name,'camera_token':name,'sample_token':'sample','timestamp':10,
                                'image_path':image_path,'camera_to_global':pose.copy(),
                                'intrinsic':np.array([[16,0,16],[0,16,16],[0,0,1]],float),
                                'ego_pose':{'rotation':[1,0,0,0],'translation':[0,0,0]}})
            masks=[np.full((32,32),2,np.uint8) for _ in CAMERAS]
            output=root/'out'
            meta=project_frame(output,cameras,masks,roi_size=10,resolution=.2)
            self.assertEqual(meta['width'],50)
            self.assertGreater(meta['semantic_cells']['road'],0)
            self.assertTrue((output/'classes.json').exists())
            np.testing.assert_array_equal(load_saved_masks(output,cameras)[0],masks[0])
            with np.load(output/'ipm_grid.npz',allow_pickle=False) as grid:
                self.assertTrue(np.isin(grid['labels'],[0,2]).all())
                self.assertTrue((grid['source_camera'][grid['labels']==2]==0).all())
                self.assertTrue((grid['support_count'][grid['labels']==2]==6).all())
            with np.load(output/'cameras/CAM_FRONT_geometry.npz',allow_pickle=False) as geometry:
                self.assertEqual(geometry['pixel_xy'].shape,(50,50,2))
                self.assertTrue(geometry['valid'].any())
            with Image.open(output/'bev_ground_ipm.png') as image:self.assertEqual(image.size,(50,50))
            with self.assertRaises(FileExistsError):project_frame(output,cameras,masks)
            with self.assertRaises(ValueError):project_frame(root/'bad',cameras[:-1],masks)

    def test_saved_mask_reuse_checks_exact_frame_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'cameras').mkdir()
            cameras=[]
            for name in CAMERAS:
                record={'camera':name,'camera_token':name,'sample_token':'sample'};cameras.append(record)
                (root/'cameras'/f'{name}_source.json').write_text(json.dumps(record))
                Image.fromarray(np.array([[1,2],[0,6]],np.uint8)).save(root/'cameras'/f'{name}_class_ids.png')
            masks=load_saved_masks(root,cameras)
            self.assertEqual(len(masks),6)
            np.testing.assert_array_equal(masks[0],[[1,2],[0,6]])
            cameras[0]['camera_token']='different'
            with self.assertRaises(ValueError):load_saved_masks(root,cameras)


if __name__=='__main__':unittest.main()

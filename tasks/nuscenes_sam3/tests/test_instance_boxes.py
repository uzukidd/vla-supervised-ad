"""Upright 3D instance boxes, surface exclusion and original point provenance."""
from pathlib import Path
import json
import tempfile
import unittest

import cv2
import numpy as np
from tasks.nuscenes_sam3.python.instance_boxes import minimum_rectangle_xy,build_instance_boxes,save_instance_boxes


def contained(points,box,tolerance=1e-7):
    theta=box['yaw_rad'];axes=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    centered=(points-np.asarray(box['center_xy']))@axes
    return (np.abs(centered[:,0])<=box['length_m']/2+tolerance)&(np.abs(centered[:,1])<=box['width_m']/2+tolerance)


class InstanceBoxTests(unittest.TestCase):
    def test_known_rotated_rectangle_area_and_full_enclosure(self):
        theta=.47;rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        points=np.array([[-3,-1],[3,-1],[3,1],[-3,1],[0,0],[2,.2]],float)@rotation.T+[1000,-2000]
        box=minimum_rectangle_xy(points)
        self.assertEqual(box['geometry_type'],'rectangle')
        self.assertAlmostEqual(box['area_m2'],12.,places=8)
        self.assertAlmostEqual(box['length_m'],6.,places=8);self.assertAlmostEqual(box['width_m'],2.,places=8)
        self.assertTrue(contained(points,box).all())
        self.assertEqual(len(box['corners_xy']),4)
        self.assertAlmostEqual(box['yaw_rad'],theta,places=8)

    def test_all_degenerate_cases_and_duplicates(self):
        for points,kind,size in ((np.empty((0,2)),'empty',0),(np.array([[2,3],[2,3]]),'point',1),
                                 (np.array([[0,0],[1,1],[2,2],[1,1]]),'segment',2)):
            box=minimum_rectangle_xy(points)
            self.assertEqual(box['geometry_type'],kind);self.assertEqual(box['area_m2'],0)
            self.assertEqual(len(box['corners_xy']),size)
        with self.assertRaises(ValueError):minimum_rectangle_xy(np.array([[np.nan,1]]))

    def test_random_cloud_agrees_with_independent_opencv_minimum(self):
        rng=np.random.default_rng(17)
        for count in (3,4,10,100,1000):
            points=rng.normal(size=(count,2))*[4,1]
            original=points.copy();box=minimum_rectangle_xy(points)
            self.assertTrue(contained(points,box).all())
            _,dimensions,_=cv2.minAreaRect(points.astype(np.float32))
            self.assertAlmostEqual(box['area_m2'],dimensions[0]*dimensions[1],delta=max(1,box['area_m2'])*1e-5)
            np.testing.assert_array_equal(points,original)
            reversed_box=minimum_rectangle_xy(points[::-1])
            np.testing.assert_allclose(box['corners_xy'],reversed_box['corners_xy'],atol=1e-10)

    def test_every_instance_is_preserved_without_cross_camera_merging(self):
        xyz=np.array([[0,0,0],[2,0,1],[2,1,2],[0,1,3],[.5,.5,10]],float)
        instances=[{'global_instance_id':8,'camera_instance_id':1,'camera':'CAM_FRONT','class_id':4,'class_name':'car','score':.9,'point_indices':[0,1,2,3,3]},
                   {'global_instance_id':9,'camera_instance_id':1,'camera':'CAM_FRONT_LEFT','class_id':4,'class_name':'car','score':.8,'point_indices':[0,1,2,3]},
                   {'global_instance_id':10,'camera':'CAM_BACK','class_id':7,'point_indices':[]},
                   {'global_instance_id':11,'camera':'CAM_BACK','class_id':7,'point_indices':[4]}]
        result=build_instance_boxes(xyz,instances,'test scene')
        self.assertEqual(len(result['boxes']),4)
        self.assertEqual(result['metadata']['cuboids'],3);self.assertEqual(result['metadata']['empty'],1)
        self.assertEqual(result['metadata']['degenerate'],1)
        first=result['boxes'][0]
        self.assertEqual(first['support_count'],4);self.assertEqual(first['height_m'],3.)
        self.assertEqual(first['z_min_m'],0);self.assertEqual(first['z_max_m'],3)
        corners=np.asarray(first['corners_scene'])
        self.assertEqual(first['geometry_type'],'cuboid');self.assertEqual(corners.shape,(8,3))
        np.testing.assert_array_equal(corners[:4,2],0.)
        np.testing.assert_array_equal(corners[4:,2],3.)
        np.testing.assert_array_equal(corners[:4,:2],corners[4:,:2])
        np.testing.assert_allclose(first['center_scene'],[1.,.5,1.5])
        self.assertEqual(first['volume_m3'],6.)
        self.assertEqual(len(first['edges']),12)
        self.assertEqual(len({tuple(sorted(edge)) for edge in first['edges']}),12)
        self.assertEqual({vertex for edge in first['edges'] for vertex in edge},set(range(8)))
        self.assertEqual(instances[0]['point_indices'],[0,1,2,3,3])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'boxes.json';save_instance_boxes(path,xyz,instances)
            self.assertEqual(len(json.loads(path.read_text())['boxes']),4)
        with self.assertRaises(ValueError):build_instance_boxes(xyz,[{**instances[0],'point_indices':[20]}])

    def test_surface_classes_are_excluded_before_geometry_generation(self):
        xyz=np.array([[0,0,-1],[2,0,2],[2,1,1],[0,1,0]],float)
        instances=[{'global_instance_id':cid,'class_id':cid,'point_indices':[0,1,2,3]}
                   for cid in (1,2,4,7)]
        for surface_ids in ((1,2),(4,7),()):
            with self.subTest(surface_ids=surface_ids):
                result=build_instance_boxes(xyz,instances,surface_class_ids=surface_ids)
                self.assertEqual([box['class_id'] for box in result['boxes']],
                                 [cid for cid in (1,2,4,7) if cid not in surface_ids])
                self.assertEqual(result['metadata']['excluded_surface_class_ids'],list(surface_ids))
                self.assertEqual(result['metadata']['excluded_surface_instances'],len(surface_ids))
                for box in result['boxes']:
                    self.assertEqual(box['geometry_type'],'cuboid')
                    self.assertEqual(box['height_m'],3.)
                    self.assertTrue(contained(xyz[:,:2],box).all())

    def test_vertical_and_flat_support_keep_zero_extent_cuboids(self):
        for xyz in (np.array([[2,3,-1],[2,3,4]],float),
                    np.array([[0,0,2],[2,0,2],[2,1,2],[0,1,2]],float)):
            with self.subTest(xyz=xyz.tolist()):
                box=build_instance_boxes(xyz,[{'class_id':4,'point_indices':list(range(len(xyz)))}])['boxes'][0]
                self.assertEqual(box['geometry_type'],'cuboid')
                self.assertEqual(len(box['corners_scene']),8)
                self.assertEqual(box['volume_m3'],0.)
                self.assertEqual(box['height_m'],float(np.ptp(xyz[:,2])))
                np.testing.assert_allclose(box['center_scene'][2],(xyz[:,2].min()+xyz[:,2].max())/2)

    def test_cleaned_points_define_rectangle_and_outlier_is_not_reintroduced(self):
        xyz=np.array([[0,0,0],[2,0,0],[2,1,0],[0,1,0],[50,50,0]],float)
        record={'global_instance_id':1,'class_id':4,'point_indices':[0,1,2,3],'original_point_indices':[0,1,2,3,4],'removed_point_indices':[4]}
        box=build_instance_boxes(xyz,[record])['boxes'][0]
        self.assertAlmostEqual(box['area_m2'],2.)
        self.assertEqual(box['point_indices'],[0,1,2,3])
        self.assertFalse(contained(xyz[4:,:2],box).all())


if __name__=='__main__':unittest.main()

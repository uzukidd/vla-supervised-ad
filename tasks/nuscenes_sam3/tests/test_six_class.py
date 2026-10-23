"""Six-class prompt scope, rider disambiguation and road/sidewalk-only planes."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from shared.python.runtime import ROOT
from shared.python.sam3 import make_palette
from tasks.nuscenes_sam3.python.six_class import NAMES,RiderMask,categories_from_txt,save_frame_bev,six_ground_plane


class SixClassTests(unittest.TestCase):
    def test_configured_prompts_and_ids_are_not_fixed_to_six(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'classes.txt'
            path.write_text('sidewalk = sidewalk\nroad = road\ncrosswalk = crosswalk\npedestrian = person\nvehicle = car\nbike = bicycle\nrider = person riding a bicycle\n')
            categories=categories_from_txt(path)
            self.assertEqual([row.name for row in categories],['sidewalk','road','crosswalk','pedestrian','vehicle','bike','rider'])
            self.assertEqual([row.id for row in categories],list(range(1,8)))
        categories=categories_from_txt(ROOT/'tasks/nuscenes_sam3/config/nuscenes_classes.txt')
        self.assertEqual(len(categories),16)

    def test_rider_overrides_only_person_or_unassigned(self):
        evidence=RiderMask((1,6))
        evidence(np.ones((1,6),bool),{'class_id':6})
        initial=np.array([[0,1,2,3,4,5]],dtype=np.uint8)
        result=evidence.apply(initial)
        np.testing.assert_array_equal(result,[[6,1,2,6,4,5]])
        np.testing.assert_array_equal(initial,[[0,1,2,3,4,5]])
        no_rider=RiderMask((1,6))
        no_rider(np.ones((1,6),bool),{'class_id':3})
        np.testing.assert_array_equal(no_rider.apply(initial),initial)

    def test_only_road_sidewalk_filled_and_non_ground_cells_block(self):
        xyz=np.array([[0,0,0],[2,0,0],[0,2,0],[2,2,0],[1,1,0],[-2,-2,0],[-1,-2,0],[-2,-1,0]],float)
        prediction=np.array([2,2,2,2,4,1,1,1],np.uint8)
        result=six_ground_plane(xyz,prediction,.2,10,3)
        self.assertTrue(set(np.unique(result['filled'])).issubset({0,1,2}))
        self.assertTrue((result['filled'][result['blocked']]==0).all())
        self.assertGreater(int((result['filled']==2).sum()),4)
        self.assertGreater(int((result['filled']==1).sum()),3)
        self.assertEqual(result['metadata']['ground_class_ids'],[1,2])
        np.testing.assert_array_equal(prediction,[2,2,2,2,4,1,1,1])

    def test_bev_outputs_use_supplied_palette_for_observed_and_filled_cells(self):
        xyz=np.array([[0,0,0],[2,0,0],[0,2,0],[2,2,0],[1,1,1]],np.float32)
        labels=np.array([2,2,2,2,4],np.uint8)
        palette=make_palette();palette[2]=[19,37,211];palette[4]=[251,101,7]
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            save_frame_bev(output,xyz,labels,np.ones(len(xyz),bool),palette,.2,10,3)
            with np.load(output/'six_class_plane.npz',allow_pickle=False) as grid:
                for filename,key in (('bev_six_sparse.png','sparse_prediction'),
                                     ('bev_six_plane.png','plane_prediction'),
                                     ('road_sidewalk_plane.png','road_sidewalk_filled')):
                    with Image.open(output/filename) as image:rgb=np.array(image)
                    ids=grid[key]
                    self.assertTrue((ids==2).any())
                    np.testing.assert_array_equal(rgb[ids==2],np.tile(palette[2],((ids==2).sum(),1)))
                    if (ids==4).any():
                        np.testing.assert_array_equal(rgb[ids==4],np.tile(palette[4],((ids==4).sum(),1)))
                self.assertTrue(grid['inferred'].any())

    def test_final_plane_preserves_foreground_and_stores_no_gt_metric(self):
        xyz=np.array([[0,0,0],[2,0,0],[0,2,0],[2,2,0],[1,1,0],[3,0,0],[0,3,0]],np.float32)
        labels=np.array([2,2,2,2,4,5,6],np.uint8)
        covered=np.ones(len(xyz),bool)
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            details=save_frame_bev(output,xyz,labels,covered,make_palette(),.2,10,3)
            self.assertIsNone(details['evaluation'])
            self.assertFalse((output/'metrics.json').exists())
            with np.load(output/'six_class_plane.npz',allow_pickle=False) as grid:
                before,after=grid['sparse_prediction'],grid['plane_prediction']
                for cid in (4,5,6):
                    self.assertTrue((after[before==cid]==cid).all())
                self.assertTrue(set(np.unique(after)).issubset({0,1,2,3,4,5,6,255}))
                self.assertTrue(np.isin(after[grid['inferred']],[1,2]).all())
                self.assertTrue((after[0]==255).all())
            with Image.open(output/'bev_six_plane.png') as image:self.assertEqual(image.size,(50,50))


if __name__=='__main__':unittest.main()

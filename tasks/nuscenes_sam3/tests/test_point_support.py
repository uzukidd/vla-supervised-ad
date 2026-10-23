"""Instance memberships align with semantic overrides and retain original indices."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tasks.nuscenes_sam3.python.point_support import ProjectedInstances
from tasks.nuscenes_sam3.python.instance_filter import clean_point_support


class ProjectedSupportTests(unittest.TestCase):
    def test_semantic_preference_reassigns_instance_without_losing_coverage(self):
        collector=ProjectedInstances(np.array([4,8,10]),np.array([[0,0],[1,0],[2,0]]),'CAM_FRONT',20)
        collector(np.ones((1,3),bool),{'instance_id':1,'class_id':2,'score':.9})
        collector(np.array([[0,1,0]],bool),{'instance_id':2,'class_id':3,'score':.6})
        collector(np.array([[0,0,1]],bool),{'instance_id':3,'class_id':5,'score':.8})
        labels=np.array([[2,3,5]],np.uint8)
        records=collector.finalize(labels)
        self.assertEqual([row['point_indices'] for row in records],[[4],[8],[10]])
        self.assertEqual([row['global_instance_id'] for row in records],[21,22,23])
        xyz=np.zeros((11,3));prediction=np.zeros(11,np.uint8);prediction[[4,8,10]]=[2,3,5]
        result=clean_point_support(xyz,prediction,records,{'algorithm':'sor'})
        np.testing.assert_array_equal(result['prediction'],prediction)


if __name__=='__main__':unittest.main()

"""Ground semantic-region fusion and evidence-limited planar interpolation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from shared.python.nuscenes_data import RAW_TO_COARSE
from tasks.nuscenes_sam3.python.ground_regions import (
    _discover_frames, fuse_semantic_points, ground_planes, main, parser, process_frame, save_ground_planes,
)


class GroundRegionTests(unittest.TestCase):
    def test_camera_point_union_score_conflicts_and_no_input_mutation(self):
        rows = [dict(global_instance_id=2,class_id=13,score=.8,point_indices=[1,2,2]),
                dict(global_instance_id=1,class_id=11,score=.8,point_indices=[0,1]),
                dict(global_instance_id=3,class_id=4,score=.9,point_indices=[2])]
        labels, assigned = fuse_semantic_points(4,rows)
        np.testing.assert_array_equal(labels,[11,11,4,0])
        np.testing.assert_array_equal(assigned,[1,1,1,0])
        self.assertEqual(rows[0]['point_indices'],[1,2,2])
        np.testing.assert_array_equal(fuse_semantic_points(4,list(reversed(rows)))[0],labels)

    def test_local_surface_fills_interior_but_not_outside_support(self):
        xyz = np.array([[0,0,0],[2,0,0],[0,2,0],[2,2,0]],dtype=float)
        labels = np.full(4,11,dtype=np.uint8)
        result = ground_planes(xyz,labels,resolution=.2,roi_size=10,max_edge=3)
        self.assertEqual(result['filled'].shape,(50,50))
        self.assertEqual(int((result['observed']==11).sum()),4)
        self.assertGreater(int((result['filled']==11).sum()),4)
        self.assertTrue(result['inferred'].any())
        self.assertEqual(int(result['filled'][0,0]),0)
        self.assertTrue((result['filled'][result['observed']!=0] == result['observed'][result['observed']!=0]).all())
        np.testing.assert_array_equal(labels,np.full(4,11,dtype=np.uint8))

    def test_large_gap_is_not_bridged(self):
        xyz=np.array([[-8,-8,0],[-7,-8,0],[-8,-7,0],[7,7,0],[8,7,0],[7,8,0]],dtype=float)
        result=ground_planes(xyz,np.full(6,11,dtype=np.uint8),resolution=.2,roi_size=20,max_edge=2)
        self.assertEqual(int(result['filled'][50,50]),0)
        for row in result['triangles']:
            for triangle in row['triangles_pixel_xy']:
                points=np.array(triangle)*.2
                self.assertLessEqual(float(np.linalg.norm(points-np.roll(points,1,axis=0),axis=1).max()),2+1e-6)

    def test_non_ground_cells_veto_road_and_other_observed_ground_preserved(self):
        xyz=np.array([[0,0,0],[2,0,0],[0,2,0],[2,2,0],[1,1,0],[.8,.8,0]],dtype=float)
        labels=np.array([11,11,11,11,4,13],dtype=np.uint8)
        result=ground_planes(xyz,labels,resolution=.2,roi_size=10,max_edge=3)
        self.assertTrue(result['blocked'].any())
        self.assertTrue((result['filled'][result['blocked']]==0).all())
        self.assertTrue((result['filled'][result['observed']==13]==13).all())
        self.assertFalse((result['inferred'] & result['blocked']).any())

    def test_empty_collinear_and_roi_point_counts(self):
        result=ground_planes(np.empty((0,3)),np.empty(0,dtype=np.uint8),resolution=.2,roi_size=10)
        self.assertFalse(result['filled'].any())
        self.assertEqual(result['metadata']['roi_points'],0)
        xyz=np.array([[0,0,0],[1,0,0],[2,0,0],[5,0,0]],dtype=float)
        result=ground_planes(xyz,np.full(4,13,dtype=np.uint8),resolution=.2,roi_size=10,max_edge=3)
        self.assertEqual(result['metadata']['roi_points'],3)
        self.assertFalse(result['inferred'].any())
        self.assertEqual(int(result['point_count'].sum()),3)

    def test_class_identity_and_deduplicated_point_counts(self):
        xyz=np.array([[0,0,0],[2,0,0],[0,2,0],[2,2,0],[-2,-2,0]],dtype=float)
        result=ground_planes(xyz,np.array([11,11,11,11,0],dtype=np.uint8),resolution=.2,roi_size=10,max_edge=3)
        self.assertEqual(set(np.unique(result['filled'])),{0,11})
        self.assertEqual(int(result['ground_point_count'].sum()),4)
        self.assertEqual(int(result['point_count'].sum()),5)
        self.assertEqual(result['metadata']['filled_ground_cells'],result['metadata']['observed_ground_cells']+result['metadata']['inferred_cells'])

    def test_invalid_parameters(self):
        xyz=np.array([[0,0,0]],dtype=float); labels=np.array([11],dtype=np.uint8)
        for kwargs in ({'max_edge':0},{'resolution':0},{'roi_size':0},{'roi_size':1,'resolution':.3}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ground_planes(xyz,labels,**kwargs)


    @staticmethod
    def semantic_frame(frame, prediction=None, metadata=None):
        frame.mkdir(parents=True)
        xyz = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0]], dtype=np.float32)
        if prediction is None:
            prediction = np.array([11, 11, 11, 0], dtype=np.uint8)
        # An object target cannot be loaded with allow_pickle=False; ground must never read it.
        np.savez_compressed(frame/'points.npz', xyz=xyz, prediction=prediction,
                            target=np.array([object()], dtype=object))
        meta = {'grid_mode': 'fixed_roi', 'resolution_m_per_pixel': .2,
                'x_min_m': -5., 'y_min_m': -5., 'x_max_m': 5., 'y_max_m': 5.,
                'width': 50, 'height': 50}
        if metadata:
            meta.update(metadata)
        (frame/'bev_metadata.json').write_text(json.dumps(meta), encoding='utf-8')
        return xyz, prediction

    def test_save_without_images_matches_computation_and_preserves_inputs(self):
        xyz = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0], [1, 1, 0]], dtype=float)
        labels = np.array([11, 11, 11, 11, 0], dtype=np.uint8)
        original_xyz, original_labels = xyz.copy(), labels.copy()
        expected = ground_planes(xyz, labels, resolution=.2, roi_size=10, max_edge=3)
        source = {'kind': 'fused_semantic_points', 'path': 'saved/frame', 'details': {'camera_rule': 'majority'}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'ground'
            with patch('tasks.nuscenes_sam3.python.ground_regions._save_ground_images', side_effect=AssertionError('No image work')):
                metadata = save_ground_planes(output, xyz, labels, resolution=.2, roi_size=10,
                                              max_edge=3, make_images=False, source_info=source)
            self.assertEqual({path.name for path in output.iterdir()},
                             {'ground_grid.npz', 'semantic_points.npz', 'ground_regions.json'})
            self.assertEqual(metadata, expected['metadata'])
            with np.load(output/'ground_grid.npz', allow_pickle=False) as archive:
                for key, value in expected.items():
                    if isinstance(value, np.ndarray):
                        np.testing.assert_array_equal(archive[key], value)
            with np.load(output/'semantic_points.npz', allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive['xyz'], xyz)
                np.testing.assert_array_equal(archive['labels'], labels)
                np.testing.assert_array_equal(archive['assigned'], labels != 0)
                self.assertEqual(archive['assigned'].dtype, np.bool_)
            data = json.loads((output/'ground_regions.json').read_text(encoding='utf-8'))
            self.assertEqual(data['source_info'], source)
            self.assertEqual(data['contours'], expected['contours'])
            self.assertEqual(data['triangles'], expected['triangles'])
        np.testing.assert_array_equal(xyz, original_xyz)
        np.testing.assert_array_equal(labels, original_labels)

    def test_save_explicit_assignments_and_default_fixed_preview_dimensions(self):
        xyz = np.array([[0., 0., 0.], [1., 0., 0.]])
        labels = np.array([11, 0], dtype=np.uint8)
        assigned = np.array([True, True])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'ground'
            metadata = save_ground_planes(output, xyz, labels, assigned=assigned)
            self.assertEqual(metadata['grid_mode'], 'fixed_roi')
            self.assertEqual(metadata['bounds_xy_m'], [-50., -50., 50., 50.])
            self.assertEqual((metadata['width'], metadata['height']), (500, 500))
            for name in ('ground_observed.png', 'ground_plane.png', 'ground_sources.png'):
                with Image.open(output/name) as image:
                    self.assertEqual(image.size, (500, 500))
            with Image.open(output/'ground_comparison.png') as image:
                self.assertEqual(image.size, (1500, 640))
            with np.load(output/'semantic_points.npz', allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive['assigned'], assigned)
        np.testing.assert_array_equal(assigned, [True, True])

    def test_save_validation_before_writes_and_output_refusal_first(self):
        xyz = np.array([[0., 0., 0.]])
        labels = np.array([11], dtype=np.uint8)
        invalid = ({'assigned': np.array([1], dtype=np.uint8)},
                   {'assigned': np.array([], dtype=bool)}, {'source_info': []},
                   {'source_info': {'invalid': float('nan')}}, {'resolution': 0},
                   {'roi_size': 1, 'resolution': .3})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'ground'
            for kwargs in invalid:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    save_ground_planes(output, xyz, labels, make_images=False, **kwargs)
                self.assertFalse(output.exists())
            output.mkdir()
            sentinel = output/'existing.txt'
            sentinel.write_bytes(b'keep existing result')
            with self.assertRaises(FileExistsError):
                save_ground_planes(output, xyz, labels, resolution=0)
            with self.assertRaises(FileExistsError):
                process_frame(Path(directory)/'missing', output, max_edge=0, taxonomy='invalid')
            self.assertEqual(sentinel.read_bytes(), b'keep existing result')
            self.assertEqual(list(output.iterdir()), [sentinel])
            with self.assertRaises(FileExistsError):
                save_ground_planes(sentinel, xyz, labels)

    def test_process_instance_layout_preserves_score_fusion_and_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory)/'instance'/'frame'
            frame.mkdir(parents=True)
            xyz = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0]], dtype=float)
            rows = [dict(global_instance_id=2, camera='CAM_FRONT', class_id=13, score=.8, point_indices=[1, 2]),
                    dict(global_instance_id=1, camera='CAM_BACK', class_id=11, score=.8, point_indices=[0, 1]),
                    dict(global_instance_id=3, camera='CAM_FRONT', class_id=4, score=.9, point_indices=[2])]
            np.savez_compressed(frame/'points.npz', xyz=xyz)
            data = {'metadata': {'resolution_m_per_pixel': .2, 'roi_size_m': 10}, 'instances': rows}
            (frame/'instance_hulls.json').write_text(json.dumps(data), encoding='utf-8')
            original = {path.name: path.read_bytes() for path in frame.iterdir()}
            output = frame/'ground'
            process_frame(frame, output, make_images=False)
            with np.load(output/'semantic_points.npz', allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive['labels'], [11, 11, 4, 0])
                np.testing.assert_array_equal(archive['assigned'], [True, True, True, False])
            saved = json.loads((output/'ground_regions.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['source_info']['kind'], 'sam3_instance_points')
            self.assertEqual(saved['source_info']['path'], str(frame))
            self.assertIn('confidence', saved['source_info']['class_assignment'])
            self.assertEqual([row['id'] for row in saved['source_instances']], [2, 1])
            for name, content in original.items():
                self.assertEqual((frame/name).read_bytes(), content)

    def test_process_semantic_layout_preserves_predictions_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = root/'pointclouds'/'frame'
            xyz, prediction = self.semantic_frame(frame)
            (root/'evaluation').mkdir()
            (root/'evaluation'/'metrics.json').write_bytes(b'{"full_scan":{"point_accuracy":0.8}}\n')
            (frame/'metrics.json').write_bytes(b'{"original":true}\n')
            before = {path.relative_to(root): path.read_bytes() for path in root.rglob('*') if path.is_file()}
            output = frame/'ground'
            metadata = process_frame(frame, output, make_images=False)
            self.assertEqual((metadata['width'], metadata['height']), (50, 50))
            self.assertEqual(metadata['roi_size_m'], 10.)
            with np.load(output/'semantic_points.npz', allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive['xyz'], xyz)
                np.testing.assert_array_equal(archive['labels'], prediction)
                np.testing.assert_array_equal(archive['assigned'], prediction != 0)
            saved = json.loads((output/'ground_regions.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['source_info']['kind'], 'fused_semantic_points')
            self.assertEqual(saved['source_info']['path'], str(frame))
            self.assertIn('majority', saved['source_info']['class_assignment'])
            self.assertNotIn('source_instances', saved)
            for name, content in before.items():
                self.assertEqual((root/name).read_bytes(), content)
            after = {path.relative_to(root) for path in root.rglob('*') if path.is_file()}
            self.assertEqual(after - set(before), {path.relative_to(root) for path in output.iterdir()})
            completed = {path.name: path.read_bytes() for path in output.iterdir()}
            with self.assertRaises(FileExistsError):
                process_frame(frame, output, make_images=False)
            self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, completed)

    def test_raw32_mapping_is_explicit_and_preserves_original_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = np.array([24, 26, 30, 31], dtype=np.uint8)
            frame = root/'frame'
            self.semantic_frame(frame, prediction=raw)
            original = (frame/'points.npz').read_bytes()
            with self.assertRaises(ValueError):
                process_frame(frame, root/'not_coarse', make_images=False)
            self.assertFalse((root/'not_coarse').exists())
            process_frame(frame, root/'mapped', taxonomy='lidarseg32', make_images=False)
            with np.load(root/'mapped'/'semantic_points.npz', allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive['labels'], RAW_TO_COARSE[raw])
                np.testing.assert_array_equal(archive['assigned'], [True, True, True, False])
            info = json.loads((root/'mapped'/'ground_regions.json').read_text(encoding='utf-8'))['source_info']
            self.assertEqual(info['source_taxonomy'], 'lidarseg32')
            self.assertIn('RAW_TO_COARSE', info['class_assignment'])
            self.assertEqual((frame/'points.npz').read_bytes(), original)
            for name, prediction in (('invalid_id', np.array([32, 0, 0, 0], dtype=np.uint8)),
                                     ('invalid_dtype', raw.astype(np.int64)), ('invalid_shape', raw[:2])):
                invalid_frame = root/name
                self.semantic_frame(invalid_frame, prediction=prediction)
                with self.subTest(name=name), self.assertRaises(ValueError):
                    process_frame(invalid_frame, root/(name+'_ground'), taxonomy='lidarseg32', make_images=False)
                self.assertFalse((root/(name+'_ground')).exists())

    def test_semantic_roi_errors_never_silently_change_extent(self):
        invalid = ({'grid_mode': 'automatic_extent'}, {'x_max_m': 6},
                   {'x_min_m': 0, 'x_max_m': 10, 'y_min_m': 0, 'y_max_m': 10},
                   {'width': 49}, {'height': 49}, {'resolution_m_per_pixel': .3},
                   {'resolution_m_per_pixel': 0}, {'y_max_m': float('nan')})
        with tempfile.TemporaryDirectory() as directory:
            for i, metadata in enumerate(invalid):
                frame, output = Path(directory)/str(i), Path(directory)/f'ground_{i}'
                self.semantic_frame(frame, metadata=metadata)
                with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                    process_frame(frame, output, make_images=False)
                self.assertFalse(output.exists())

    def test_cli_discovers_all_saved_input_roots_and_single_frames(self):
        layouts = ('instances/frame', 'instance/frame', 'frame', 'pointclouds/frame',
                   'raw/pointcloud_results/pointclouds/frame')
        with tempfile.TemporaryDirectory() as directory:
            for i, layout in enumerate(layouts):
                root = Path(directory)/str(i)
                frame = root/layout
                self.semantic_frame(frame)
                if layout.startswith(('instance/', 'instances/')):
                    data = {'metadata': {'resolution_m_per_pixel': .2, 'roi_size_m': 10}, 'instances': []}
                    (frame/'instance_hulls.json').write_text(json.dumps(data), encoding='utf-8')
                self.assertEqual(_discover_frames(root), [frame])
                self.assertEqual(_discover_frames(frame), [frame])
                self.assertEqual(_discover_frames(frame/'points.npz'), [frame])
                if layout == 'pointclouds/frame':
                    self.assertEqual(_discover_frames(frame.parent), [frame])
                output = Path(directory)/f'output_{i}'
                with patch('tasks.nuscenes_sam3.python.ground_regions.ensure_environment'):
                    self.assertEqual(main(['--input-dir', str(root), '--output-dir', str(output)]), 0)
                self.assertTrue((output/'frame'/'ground_comparison.png').is_file())

    def test_cli_sample_default_limit_and_validation(self):
        self.assertEqual(parser().parse_args([]).samples, 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'source'
            self.semantic_frame(root/'pointclouds'/'a')
            self.semantic_frame(root/'pointclouds'/'b')
            output = Path(directory)/'ground'
            with patch('tasks.nuscenes_sam3.python.ground_regions.ensure_environment'):
                with self.assertRaises(ValueError):
                    main(['--input-dir', str(root), '--output-dir', str(output), '--samples', '0'])
                self.assertFalse(output.exists())
                self.assertEqual(main(['--input-dir', str(root), '--output-dir', str(output)]), 0)
                self.assertTrue((output/'a'/'ground_regions.json').is_file())
                self.assertFalse((output/'b').exists())
                before = (output/'run_config.json').read_bytes()
                with self.assertRaises(FileExistsError):
                    main(['--input-dir', str(root), '--output-dir', str(output)])
                self.assertEqual((output/'run_config.json').read_bytes(), before)


if __name__=='__main__':
    unittest.main()

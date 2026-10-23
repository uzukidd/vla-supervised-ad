"""User-configured taxonomy, selected planes and saved-ID compatibility."""
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np

from unittest.mock import patch
from PIL import Image

from tasks.nuscenes_sam3.python.semantic_schema import (categories_from_txt,taxonomy_from_categories,resolve_planes,saved_schema,palette_for,LEGACY_NAMES,
    load_display_names,saved_display_names,class_rows,load_colors,normalize_colors,saved_colors)
from tasks.nuscenes_sam3.python.six_class import RiderMask,six_ground_plane,save_frame_bev,main as run_six

NAMES={0:'unassigned',1:'sidewalk',2:'road',3:'crosswalk',4:'pedestrian',5:'vehicle',6:'bike',7:'rider'}


class PlaneSelectionTests(unittest.TestCase):
    def test_color_sidecar_normalizes_and_preserves_ids_labels_and_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            classes=Path(directory)/'classes.txt'
            classes.write_text('road = paved road\nvehicle = car\n',encoding='utf-8')
            names=taxonomy_from_categories(categories_from_txt(classes))
            self.assertEqual(load_colors(classes),{})
            default=palette_for(names)
            np.testing.assert_array_equal(default[1],[210,155,20])
            classes.with_suffix('.colors.json').write_text(json.dumps({'road':'#12ABef'}),encoding='utf-8')
            overrides=load_colors(classes)
            self.assertEqual(overrides,{'road':'#12abef'})
            palette=palette_for(names,overrides)
            np.testing.assert_array_equal(palette[1],[18,171,239])
            np.testing.assert_array_equal(palette[2:],default[2:])
            rows=class_rows(names,{'road':'路面'},overrides)
            self.assertEqual(rows[1],{'id':1,'name':'road','label':'路面','display_name':'路面','color':[18,171,239]})
            self.assertEqual(taxonomy_from_categories(categories_from_txt(classes)),names)

    def test_invalid_colors_are_rejected_before_numpy_channel_conversion(self):
        for invalid in ([],None,{'road':[1,2,3]},{'road':'red'},{'road':'#abc'},
                        {'road':'#00000g'},{'road':' #123456'},{1:'#123456'},{'':'#123456'}):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):normalize_colors(invalid)
        with self.assertRaisesRegex(ValueError,'Unknown color class names'):
            palette_for(NAMES,{'raod':'#123456'})
        with tempfile.TemporaryDirectory() as directory:
            classes=Path(directory)/'classes.txt'
            classes.with_suffix('.colors.json').write_text('{"road": false}',encoding='utf-8')
            with self.assertRaises(ValueError):load_colors(classes)

    def test_saved_colors_roundtrip_and_follow_nearest_taxonomy(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);frame=root/'frame';frame.mkdir()
            self.assertEqual(saved_colors(frame),{})
            overrides={'road':'#0012ef','vehicle':'#ff8000'}
            palette=palette_for(NAMES,overrides)
            rows=[{'id':cid,'name':name,'color_rgb':palette[cid].tolist()} for cid,name in NAMES.items()]
            classes=root/'classes.json';classes.write_text(json.dumps(rows),encoding='utf-8')
            recovered=saved_colors(frame)
            self.assertEqual(recovered['road'],overrides['road'])
            self.assertEqual(saved_colors(classes),recovered)
            np.testing.assert_array_equal(palette_for(NAMES,recovered),palette)
            (frame/'classes.json').write_text(json.dumps(LEGACY_NAMES),encoding='utf-8')
            self.assertEqual(saved_colors(frame),{})
            (frame/'classes.json').write_text(json.dumps([{'id':cid,'name':name} for cid,name in NAMES.items()]),encoding='utf-8')
            self.assertEqual(saved_colors(frame),{})
            nested=root/'old-config';nested.mkdir()
            (nested/'run_config.json').write_text(json.dumps({'taxonomy':LEGACY_NAMES}),encoding='utf-8')
            self.assertEqual(saved_colors(nested),{})
            for rgb in ([256,0,0],[-1,0,0],[True,0,0],[1.5,0,0],[1,2],'#123456'):
                classes.write_text(json.dumps([{'id':1,'name':'road','color_rgb':rgb}]),encoding='utf-8')
                with self.subTest(rgb=rgb),self.assertRaises(ValueError):saved_colors(root)

    def test_optional_chinese_display_names_preserve_taxonomy_and_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);classes=root/'classes.txt'
            classes.write_text('road = road | paved street\nvehicle = car | truck\n',encoding='utf-8')
            categories=categories_from_txt(classes)
            names=taxonomy_from_categories(categories)
            self.assertEqual(load_display_names(classes),{})
            self.assertEqual(class_rows(names)[1]['label'],'道路 / road')
            labels={'road':'可通行路面','vehicle':'机动车辆'}
            classes.with_suffix('.display_names.json').write_text(json.dumps(labels,ensure_ascii=False),encoding='utf-8')
            self.assertEqual(load_display_names(classes),labels)
            self.assertEqual(categories_from_txt(classes),categories)
            self.assertEqual(names,{0:'unassigned',1:'road',2:'vehicle'})
            self.assertEqual(categories[0].prompts,('road','paved street'))
            rows=class_rows(names,labels)
            self.assertEqual(rows[1]['label'],'可通行路面')
            self.assertEqual(rows[1]['display_name'],'可通行路面')
            self.assertEqual(rows[1]['name'],'road')
            self.assertEqual(rows[0]['label'],'未分配 / unassigned')
            self.assertEqual(resolve_planes(names,['road']),(1,))
            self.assertEqual(class_rows({0:'unassigned',1:'custom'})[1]['label'],'custom / custom')

    def test_saved_display_names_follow_saved_schema_and_support_old_configs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);frame=root/'sample';frame.mkdir()
            self.assertEqual(saved_display_names(frame),{})
            classes=root/'classes.json'
            classes.write_text(json.dumps([{'id':cid,'name':name} for cid,name in NAMES.items()]),encoding='utf-8')
            self.assertEqual(saved_display_names(frame),{})
            rows=[{'id':cid,'name':name,**({'display_name':'定制人行横道'} if name=='crosswalk' else {})}
                  for cid,name in NAMES.items()]
            classes.write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
            self.assertEqual(saved_display_names(frame),{'crosswalk':'定制人行横道'})
            self.assertEqual(saved_display_names(classes),{'crosswalk':'定制人行横道'})
            self.assertEqual(saved_schema(frame)[0],NAMES)
            # A nearer legacy taxonomy must not inherit labels from an ancestor.
            (frame/'classes.json').write_text(json.dumps(LEGACY_NAMES),encoding='utf-8')
            self.assertEqual(saved_display_names(frame),{})
            self.assertEqual(saved_schema(frame)[0],LEGACY_NAMES)
            nested=root/'old-config';nested.mkdir()
            (nested/'run_config.json').write_text(json.dumps({'taxonomy':LEGACY_NAMES,'plane_class_ids':[2]}),encoding='utf-8')
            self.assertEqual(saved_display_names(nested),{})
            self.assertEqual(saved_schema(nested),(LEGACY_NAMES,[2]))

    def test_six_class_persists_labels_and_excludes_selected_surface_boxes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);classes=root/'classes.txt';output=root/'segmentation'
            classes.write_text('road = road | paved street\nvehicle = car | truck\n',encoding='utf-8')
            classes.with_suffix('.display_names.json').write_text(json.dumps({'road':'可通行路面','vehicle':'机动车辆'},ensure_ascii=False),encoding='utf-8')
            classes.with_suffix('.colors.json').write_text(json.dumps({'road':'#12ABef','vehicle':'#987654'}),encoding='utf-8')
            image=root/'camera.png';Image.new('RGB',(2,2)).save(image)
            from shared.python.nuscenes_data import CAMERAS
            views=[{'camera':camera,'image_path':image,'lidar_token':'lidar','scene_name':'scene',
                    'sample_token':'sample','camera_token':camera} for camera in CAMERAS]
            xyz=np.array([[0,0,0],[1,0,1],[0,1,2],[1,1,3]],np.float32)
            class Dataset:
                def __init__(self,*args,**kwargs):
                    if kwargs!={'version':'v1.0-trainval','require_lidarseg':False}:raise AssertionError(kwargs)
                def select_views(self,*args):return views
                def load_lidar(self,*args):return {'xyz':xyz}
                def projection(self,*args):return {'point_indices':np.arange(4),'pixels':np.array([[0,0],[1,0],[0,1],[1,1]])}
            class Segmenter:
                def __init__(self,*args,**kwargs):pass
                def predict(self,image,categories,instance_callback):
                    self_categories=[(category.id,category.name,category.prompts) for category in categories]
                    if self_categories!=[(1,'road',('road','paved street')),(2,'vehicle',('car','truck'))]:
                        raise AssertionError('Display labels changed inference categories')
                    record={'instance_id':1,'class_id':1,'class_name':'road','score':.9}
                    instance_callback(np.ones((2,2),bool),record)
                    return np.ones((2,2),np.uint8),[record]
            with patch('tasks.nuscenes_sam3.python.six_class.ensure_environment'), \
                 patch('tasks.nuscenes_sam3.python.six_class.NuScenesMini',Dataset), \
                 patch('tasks.nuscenes_sam3.python.six_class.Sam3Segmenter',Segmenter), \
                 patch('tasks.nuscenes_sam3.python.six_class.save_frame_bev') as save_bev, \
                 patch('tasks.nuscenes_sam3.python.six_class.page'):
                self.assertEqual(run_six(['--classes-file',str(classes),'--output-dir',str(output),
                                          '--version','v1.0-trainval','--split','all','--plane-classes','vehicle','--no-scene-viewer','--point-filter','none']),0)
            saved=json.loads((output/'classes.json').read_text(encoding='utf-8'))
            self.assertEqual(saved[1]['color_rgb'],[18,171,239])
            self.assertEqual(saved[2]['color_rgb'],[152,118,84])
            self.assertEqual(saved_colors(output)['road'],'#12abef')
            np.testing.assert_array_equal(save_bev.call_args.args[4][1],[18,171,239])
            frame=next(output.glob('scene_*'))
            with Image.open(frame/'cameras'/f'{CAMERAS[0]}_semantics.png') as semantic:
                self.assertEqual(semantic.getpixel((0,0)),(18,171,239))
            self.assertEqual(saved[1]['display_name'],'可通行路面')
            self.assertEqual(saved[2]['display_name'],'机动车辆')
            self.assertEqual(saved[1]['name'],'road')
            self.assertEqual(saved[1]['prompts'],['road','paved street'])
            self.assertEqual(saved_display_names(output),{'road':'可通行路面','vehicle':'机动车辆'})
            frame=next(output.glob('scene_*'))
            boxes=json.loads((frame/'instance_boxes.json').read_text(encoding='utf-8'))
            self.assertEqual(boxes['metadata']['excluded_surface_class_ids'],[2])
            self.assertEqual(len(boxes['boxes']),1)
            box=boxes['boxes'][0]
            self.assertTrue(box['merged']);self.assertEqual(box['member_count'],6)
            self.assertEqual(box['point_indices'],[0,1,2,3])
            self.assertEqual(box['member_instance_ids'],[1,2,3,4,5,6])
            self.assertEqual(set(box['source_cameras']),set(CAMERAS))
            self.assertEqual(len(json.loads((frame/'point_instances.json').read_text())),6)
            report=json.loads((frame/'instance_merge.json').read_text())
            self.assertEqual(report['metadata']['merged_groups'],1)
            self.assertEqual(len(report['instances']),1)
            self.assertEqual(len(json.loads((frame/'merged_instances.json').read_text())),1)

    def test_named_planes_validate_and_do_not_rename_saved_ids(self):
        self.assertEqual(resolve_planes(NAMES,['road','crosswalk','sidewalk']),(1,2,3))
        self.assertEqual(resolve_planes(NAMES,['crosswalk']),(3,))
        self.assertEqual(resolve_planes(NAMES), (1,2))
        with self.assertRaises(ValueError):resolve_planes(NAMES,['crosswlk'])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);frame=root/'sample';frame.mkdir()
            (root/'classes.json').write_text(json.dumps([{'id':cid,'name':name} for cid,name in LEGACY_NAMES.items()]))
            (root/'run_config.json').write_text(json.dumps({'plane_class_ids':[1,2]}))
            names,planes=saved_schema(frame)
            self.assertEqual(names[3],'pedestrian')
            self.assertEqual(resolve_planes(names,default_ids=planes),(1,2))
            with self.assertRaises(ValueError):resolve_planes(names,['crosswalk'])

    def test_named_rider_and_crosswalk_conflicts_follow_shifted_ids(self):
        evidence=RiderMask((1,8),NAMES)
        rider=np.zeros((1,8),bool);rider[0,[0,1,2]]=True
        crosswalk=np.zeros((1,8),bool);crosswalk[0,[3,4,5,6]]=True
        evidence(rider,{'class_id':7});evidence(crosswalk,{'class_id':3})
        original=np.array([[0,4,6,2,0,5,1,4]],np.uint8)
        result=evidence.apply(original)
        np.testing.assert_array_equal(result,[[7,7,6,3,3,5,1,4]])
        np.testing.assert_array_equal(original,[[0,4,6,2,0,5,1,4]])

    def test_select_crosswalk_plane_or_keep_its_points(self):
        xyz=np.array([[0,0,0],[2,0,0],[0,2,0],[2,2,0],[1,1,1],[-2,-2,0]],np.float32)
        labels=np.array([3,3,3,3,5,2],np.uint8)
        chosen=six_ground_plane(xyz,labels,.2,10,3,NAMES,(1,2,3))
        excluded=six_ground_plane(xyz,labels,.2,10,3,NAMES,(1,2))
        self.assertGreater(int((chosen['filled']==3).sum()),4)
        self.assertFalse((excluded['filled']==3).any())
        self.assertTrue(chosen['blocked'].any())
        self.assertTrue((chosen['filled'][chosen['blocked']]==0).all())
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            save_frame_bev(root,xyz,labels,np.ones(len(xyz),bool),palette_for(NAMES),.2,10,3,NAMES,(1,2))
            with np.load(root/'six_class_plane.npz') as data:
                before,after=data['sparse_prediction'],data['plane_prediction']
                np.testing.assert_array_equal(after[before==3],before[before==3])
                self.assertTrue(np.isin(after[data['inferred']],[1,2]).all())


if __name__=='__main__':unittest.main()

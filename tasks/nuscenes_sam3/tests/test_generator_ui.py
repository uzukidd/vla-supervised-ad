"""Generator UI configuration validation, safe paths and real subprocess lifecycle."""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
import threading
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer
from unittest.mock import patch

from tasks.nuscenes_sam3.python.generator_ui import validated_config,command_for,Generator,initial_config,handler_class,validated_classes,dataset_scenes

SCENES=[{'name':'scene-0103','frames':40,'available':True},{'name':'scene-0916','frames':41,'available':True},
        {'name':'scene-missing','frames':2,'available':False}]
CONFIG={'classes':[{'name':'road','display_name':'道路','color':'#d29b14','prompts':'road','plane':True},{'name':'crosswalk','display_name':'人行横道','color':'#23becd','prompts':'crosswalk | pedestrian crossing','plane':True},
                   {'name':'person','display_name':'行人','color':'#123456','prompts':'person','plane':False}],
        'scenes':['scene-0103'],'samples':1}
BOOT={'classes':CONFIG['classes'],'scenes':SCENES,'defaults':{}}
DEFAULT_FILTER={'algorithm':'none','sor_k':16,'sor_std':2.,'dbscan_eps':1.,'dbscan_min_samples':5,
                'cluster_policy':'significant','min_cluster_size':10,'cluster_ratio':.1}


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        versions=patch('tasks.nuscenes_sam3.python.generator_ui.installed_versions',return_value=['v1.0-mini','v1.0-trainval'])
        versions.start();self.addCleanup(versions.stop)

    def test_validation_and_all_scene_estimate(self):
        config,n=validated_config({**CONFIG,'scenes':['scene-0916','scene-0103'],'samples':0},SCENES)
        self.assertEqual(n,81)
        command=command_for(config,Path('/data'),Path('/classes with space.txt'),Path('/output'))
        self.assertIn('crosswalk',command)
        self.assertEqual(command[command.index('--classes-file')+1],'/classes with space.txt')
        self.assertIn('--plane-classes',command)
        self.assertNotIn('person',command)
        self.assertEqual(command[0],sys.executable)
        self.assertEqual(config['point_filter'],DEFAULT_FILTER)
        self.assertEqual(config['occupancy_resolution'],.5)
        self.assertEqual(command[command.index('--occupancy-resolution')+1],'0.5')
        self.assertEqual(command[command.index('--point-filter')+1],'none')
        self.assertNotIn('point_filter',CONFIG)

    def test_boot_defaults_include_disabled_point_filter(self):
        with patch('tasks.nuscenes_sam3.python.generator_ui.CameraFrames') as frames,patch('tasks.nuscenes_sam3.python.generator_ui.parse_class_file',return_value=[('road',['road'])]):
            frames.return_value.frames.return_value=[]
            frames.return_value.tables={'sample_data':{},'sample':{},'scene':{}}
            boot=initial_config(Path('/data'),Path('/classes.txt'))
        self.assertEqual(boot['defaults']['point_filter'],DEFAULT_FILTER)
        self.assertEqual(boot['defaults']['occupancy_resolution'],.5)

    def test_selected_point_filters_and_command_mapping(self):
        selected={'sor_k':24,'sor_std':1.5,'dbscan_eps':.7,'dbscan_min_samples':7,
                  'cluster_policy':'largest','min_cluster_size':12,'cluster_ratio':.25}
        for algorithm in ('none','sor','dbscan','sor_dbscan'):
            with self.subTest(algorithm=algorithm):
                point_filter={**selected,'algorithm':algorithm}
                config,estimate=validated_config({**CONFIG,'point_filter':point_filter},SCENES)
                self.assertEqual(config['point_filter'],point_filter)
                self.assertEqual(estimate,1)
                command=command_for(config,Path('/data'),Path('/classes.txt'),Path('/output'))
                flags={'--point-filter':algorithm,**{'--'+key.replace('_','-'):str(value) for key,value in selected.items()}}
                for flag,value in flags.items():
                    self.assertEqual(command.count(flag),1)
                    self.assertEqual(command[command.index(flag)+1],value)
                with patch('tasks.nuscenes_sam3.python.generator_ui.initial_config',return_value=BOOT),tempfile.TemporaryDirectory() as directory:
                    generator=Generator(directory,Path(directory)/'classes.txt',Path(directory)/'jobs')
                    preview=generator.preview({**CONFIG,'point_filter':point_filter})
                self.assertEqual(preview['config']['point_filter'],point_filter)
                self.assertIn('--point-filter '+algorithm,preview['command'])
        config,_=validated_config({**CONFIG,'point_filter':{'algorithm':'sor'}},SCENES)
        self.assertEqual(config['point_filter'],{**DEFAULT_FILTER,'algorithm':'sor'})

    def test_invalid_point_filter_parameters(self):
        invalid=[[],True,'sor',{'algorithm':'unknown'},{'sor_k':0},{'sor_k':1.5},{'sor_k':True},
                 {'sor_std':0},{'sor_std':float('nan')},{'sor_std':float('inf')},{'sor_std':True},
                 {'dbscan_eps':-1},{'dbscan_eps':float('inf')},{'dbscan_min_samples':0},
                 {'dbscan_min_samples':2.5},{'dbscan_min_samples':True},{'cluster_policy':'all'},
                 {'min_cluster_size':0},{'min_cluster_size':1.5},{'cluster_ratio':0},{'cluster_ratio':-.1},
                 {'cluster_ratio':1.1},{'cluster_ratio':float('nan')},{'cluster_ratio':True}]
        for point_filter in invalid:
            with self.subTest(point_filter=point_filter),self.assertRaises(ValueError):
                validated_config({**CONFIG,'point_filter':point_filter},SCENES)

    def test_invalid_names_prompts_scenes_and_parameters(self):
        cases=[{**CONFIG,'scenes':[]},{**CONFIG,'scenes':['scene-missing']},{**CONFIG,'scenes':['unknown']},
               {**CONFIG,'classes':[{'name':'road','prompts':'road\ncar = person','plane':True}]},
               {**CONFIG,'classes':[{'name':'road','prompts':'road','plane':False}]},
               {**CONFIG,'classes':CONFIG['classes']+[CONFIG['classes'][0]]},
               {**CONFIG,'samples':-1},{**CONFIG,'start_index':40},{**CONFIG,'map_resolution':0},
               {**CONFIG,'max_edge':float('nan')},{**CONFIG,'device':'base'},
               {**CONFIG,'occupancy_resolution':0},{**CONFIG,'occupancy_resolution':-0.1},
               {**CONFIG,'occupancy_resolution':float('inf')},{**CONFIG,'occupancy_resolution':True}]
        for raw in cases:
            with self.subTest(raw=raw),self.assertRaises(ValueError):validated_config(raw,SCENES)

    def test_per_run_configuration_results_and_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch('tasks.nuscenes_sam3.python.generator_ui.initial_config',return_value=BOOT):
                generator=Generator(root,root/'original.txt',root/'jobs')
            # Run a harmless child that creates only its declared result and emits one progress line.
            script="from pathlib import Path;import sys;d=Path(sys.argv[1]);(d/'scenes').mkdir(parents=True);(d/'scenes/index.html').write_text('result');print('CAM_FRONT: configured instances=1',flush=True)"
            def command(config,dataset,classes,output):return [sys.executable,'-u','-c',script,str(output)]
            with patch('tasks.nuscenes_sam3.python.generator_ui.command_for',side_effect=command):run=generator.start(CONFIG)
            generator.runs[run['id']]['_thread'].join(timeout=10)
            result=generator.snapshot(generator.runs[run['id']])
            self.assertEqual(result['status'],'completed')
            self.assertIn('configured instances=1',result['log_tail'])
            self.assertTrue(result['result_url'].endswith('/scenes/index.html'))
            self.assertEqual((Path(run['directory'])/'classes.txt').read_text().splitlines()[1],'crosswalk = crosswalk | pedestrian crossing')
            self.assertFalse((root/'original.txt').exists())
            self.assertTrue(generator.result_path(run['id'],'scenes/index.html').is_file())
            with self.assertRaises(FileNotFoundError):generator.result_path(run['id'],'../../classes.txt')
            with self.assertRaises(FileNotFoundError):generator.result_path('unknown','index.html')
            with patch('tasks.nuscenes_sam3.python.generator_ui.initial_config',return_value=BOOT):
                restored=Generator(root,root/'original.txt',root/'jobs')
            self.assertEqual(restored.list_runs()[0]['status'],'completed')

    def test_classes_persist_order_prompts_and_planes_across_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch('tasks.nuscenes_sam3.python.generator_ui.initial_config',return_value=BOOT):
                generator=Generator(root,root/'original.txt',root/'jobs')
                rows=[CONFIG['classes'][2],{**CONFIG['classes'][1],'prompts':'crosswalk | zebra crossing'},CONFIG['classes'][0]]
                result=generator.save_classes({'classes':rows})
                saved=root/'jobs/semantic_classes.json'
                self.assertEqual(result['classes_path'],str(saved))
                self.assertEqual(json.loads(saved.read_text())['classes'],rows)
                restored=Generator(root,root/'original.txt',root/'jobs')
                self.assertEqual(restored.bootstrap['classes'],rows)
                self.assertEqual(restored.preview({**CONFIG,'classes':rows})['config']['classes'],rows)
                with patch('tasks.nuscenes_sam3.python.generator_ui.threading.Thread.start'):
                    run=restored.start({**CONFIG,'classes':rows})
                self.assertEqual((Path(run['directory'])/'classes.txt').read_text().splitlines(),
                                 [f"{row['name']} = {row['prompts']}" for row in rows])
                self.assertEqual(json.loads((Path(run['directory'])/'classes.display_names.json').read_text()),
                                 {row['name']:row['display_name'] for row in rows})
                self.assertEqual(json.loads((Path(run['directory'])/'classes.colors.json').read_text()),
                                 {row['name']:row['color'] for row in rows})
                before=saved.read_bytes()
                with self.assertRaises(ValueError):restored.save_classes({'classes':rows+[rows[0]]})
                self.assertEqual(saved.read_bytes(),before)
                self.assertFalse((root/'original.txt').exists())
                self.assertEqual(BOOT['classes'],CONFIG['classes'])

    def test_corrupt_saved_classes_report_error_and_keep_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);jobs=root/'jobs';jobs.mkdir()
            path=jobs/'semantic_classes.json';path.write_text('{broken')
            with patch('tasks.nuscenes_sam3.python.generator_ui.initial_config',return_value=BOOT):
                generator=Generator(root,root/'classes.txt',jobs)
            self.assertEqual(generator.bootstrap['classes'],CONFIG['classes'])
            self.assertIn('classes_error',generator.bootstrap)
            self.assertEqual(path.read_text(),'{broken')

    def test_display_names_are_independent_and_legacy_classes_migrate(self):
        legacy=[{key:value for key,value in row.items() if key!='display_name'} for row in CONFIG['classes']]
        normalized=validated_classes(legacy)
        self.assertEqual(normalized[0]['display_name'],'道路')
        custom=[{**CONFIG['classes'][0],'display_name':'城市道路 / Urban road'},*CONFIG['classes'][1:]]
        self.assertEqual(validated_classes(custom),custom)
        self.assertEqual(validated_classes([{**custom[0],'display_name':''}])[0]['display_name'],'道路')
        for value in (42,'bad\nlabel','x'*129):
            with self.subTest(display_name=value),self.assertRaises(ValueError):
                validated_classes([{**custom[0],'display_name':value}])

    def test_plane_parameters_and_color_validation(self):
        raw={**CONFIG,'bev_resolution':.25,'bev_roi_size':80,'max_edge':3.,'map_resolution':.5,'occupancy_resolution':.25}
        config,_=validated_config(raw,SCENES)
        command=command_for(config,Path('/data'),Path('/classes.txt'),Path('/output'))
        self.assertEqual(command[command.index('--occupancy-resolution')+1],'0.25')
        for flag,value in (('--bev-resolution','0.25'),('--bev-roi-size','80.0'),('--max-edge','3.0'),('--map-resolution','0.5')):
            self.assertEqual(command[command.index(flag)+1],value)
        self.assertNotIn('--plane-z',command);self.assertNotIn('--max-distance',command)
        for options in ({'bev_resolution':0},{'bev_roi_size':-1},{'bev_roi_size':100.1,'bev_resolution':.2},
                        {'bev_roi_size':1000,'bev_resolution':.02}):
            with self.subTest(options=options),self.assertRaises(ValueError):validated_config({**CONFIG,**options},SCENES)
        old=[{key:value for key,value in row.items() if key!='color'} for row in CONFIG['classes']]
        self.assertEqual(validated_classes(old)[0]['color'],'#d29b14')
        self.assertEqual(validated_classes([{**CONFIG['classes'][0],'color':'#ABCDEF'}])[0]['color'],'#abcdef')
        for color in ('red','#123',True,None,[1,2,3]):
            with self.subTest(color=color),self.assertRaises(ValueError):
                validated_classes([{**CONFIG['classes'][0],'color':color}])

    def test_version_selects_matching_scene_cache_and_command(self):
        trainval=[{'name':'scene-0001','frames':40,'available':True}]
        with tempfile.TemporaryDirectory() as directory,patch('tasks.nuscenes_sam3.python.generator_ui.initial_config',return_value=BOOT):
            root=Path(directory);generator=Generator(root,root/'classes.txt',root/'jobs')
            with patch('tasks.nuscenes_sam3.python.generator_ui.dataset_scenes',return_value=trainval) as discover:
                self.assertEqual(generator.scenes_for('v1.0-trainval')['scenes'],trainval)
                self.assertEqual(generator.scenes_for('v1.0-trainval')['scenes'],trainval)
                discover.assert_called_once_with(generator.dataset_dir,'v1.0-trainval')
            raw={**CONFIG,'version':'v1.0-trainval','scenes':['scene-0001']}
            preview=generator.preview(raw)
            self.assertEqual(preview['config']['version'],'v1.0-trainval')
            self.assertIn('--version v1.0-trainval',preview['command'])
            with self.assertRaises(ValueError):generator.preview({**raw,'version':'v1.0-mini'})
            with self.assertRaises(ValueError):generator.scenes_for('../bad')
            with self.assertRaises(ValueError):generator.preview({**raw,'version':'v1.0-test'})

    def test_trainval_scene_availability_without_gt_and_missing_camera(self):
        from types import SimpleNamespace
        from tasks.flat_ground.python.project_semantics import CAMERAS
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'samples/CAM_FRONT').mkdir(parents=True);(root/'samples/LIDAR_TOP').mkdir()
            (root/'samples/CAM_FRONT/ok.jpg').write_bytes(b'camera')
            (root/'samples/LIDAR_TOP/ok.bin').write_bytes(b'lidar')
            scenes={name:{'token':name,'name':name} for name in ('complete','incomplete')}
            sample={name:{'token':name,'scene_token':name} for name in scenes}
            channels={name:{cam:{'filename':('samples/CAM_FRONT/missing.jpg' if name=='incomplete' and cam=='CAM_BACK' else 'samples/CAM_FRONT/ok.jpg')} for cam in CAMERAS} for name in scenes}
            tables={'scene':scenes,'sample':sample,'sensor':{'sensor':{'channel':'LIDAR_TOP'}},
                    'calibrated_sensor':{'cal':{'sensor_token':'sensor'}},'sample_data':{name:{'sample_token':name,'is_key_frame':True,'calibrated_sensor_token':'cal','filename':'samples/LIDAR_TOP/ok.bin'} for name in scenes}}
            dataset=SimpleNamespace(root=root,tables=tables,channels=channels)
            with patch('tasks.nuscenes_sam3.python.generator_ui.CameraFrames',return_value=dataset) as loader:
                rows=dataset_scenes(root,'v1.0-trainval')
            loader.assert_called_once_with(root,version='v1.0-trainval')
            self.assertTrue(rows[0]['available']);self.assertFalse(rows[1]['available'])
            self.assertEqual(rows[0]['split'],'trainval')

    def test_optional_grid_projection_configuration_and_command(self):
        config,_=validated_config(CONFIG,SCENES)
        self.assertFalse(config['compute_grid_projection'])
        self.assertNotIn('--compute-grid-projection',command_for(config,Path('/data'),Path('/classes.txt'),Path('/output')))
        raw={**CONFIG,'compute_grid_projection':True,'grid_projection_samples':5,'grid_projection_max_distance':60.}
        config,_=validated_config(raw,SCENES)
        command=command_for(config,Path('/data'),Path('/classes.txt'),Path('/output'))
        self.assertIn('--compute-grid-projection',command)
        self.assertEqual(command[command.index('--grid-projection-samples')+1],'5')
        self.assertEqual(command[command.index('--grid-projection-max-distance')+1],'60.0')
        for options in ({'compute_grid_projection':'true'},{'grid_projection_samples':0},
                        {'grid_projection_samples':9},{'grid_projection_samples':2.5},
                        {'grid_projection_samples':True},{'grid_projection_max_distance':0},
                        {'grid_projection_max_distance':float('nan')},{'grid_projection_max_distance':True}):
            with self.subTest(options=options),self.assertRaises(ValueError):
                validated_config({**raw,**options},SCENES)

    def test_http_classes_endpoint_persists_and_reloads_display_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch('tasks.nuscenes_sam3.python.generator_ui.initial_config',return_value=BOOT):
                generator=Generator(root,root/'classes.txt',root/'jobs')
            server=ThreadingHTTPServer(('127.0.0.1',0),handler_class(generator))
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            base=f'http://127.0.0.1:{server.server_port}'
            def post(rows,token):
                request=urllib.request.Request(base+'/api/classes',json.dumps({'classes':rows}).encode(),
                                               headers={'Content-Type':'application/json','X-Generator-Token':token})
                return urllib.request.urlopen(request,timeout=5)
            try:
                with urllib.request.urlopen(base+'/api/config',timeout=5) as response:boot=json.load(response)
                self.assertTrue(boot['capabilities']['class_display_names'])
                rows=list(reversed(CONFIG['classes']))
                with post(rows,boot['token']) as response:self.assertEqual(json.load(response)['classes'],rows)
                with urllib.request.urlopen(base+'/api/config',timeout=5) as response:self.assertEqual(json.load(response)['classes'],rows)
                with self.assertRaises(urllib.error.HTTPError) as error:post(rows,'wrong-token')
                self.assertEqual(error.exception.code,403)
                self.assertEqual(json.loads(generator.classes_path.read_text())['classes'],rows)
            finally:server.shutdown();server.server_close();thread.join(timeout=5)

    def test_stop_terminates_only_managed_child_and_marks_stopped(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch('tasks.nuscenes_sam3.python.generator_ui.initial_config',return_value=BOOT):generator=Generator(root,root/'c.txt',root/'jobs')
            with patch('tasks.nuscenes_sam3.python.generator_ui.command_for',return_value=[sys.executable,'-u','-c','import time;print("started",flush=True);time.sleep(30)']):
                run=generator.start(CONFIG)
            deadline=time.monotonic()+5
            while generator.runs[run['id']]['status']=='starting' and time.monotonic()<deadline:time.sleep(.01)
            with self.assertRaises(ValueError):generator.start(CONFIG)
            generator.stop(run['id']);generator.runs[run['id']]['_thread'].join(timeout=10)
            self.assertEqual(generator.runs[run['id']]['status'],'stopped')


if __name__=='__main__':unittest.main()

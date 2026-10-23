#!/usr/bin/env python
"""Local nuScenes generator UI: configuration, scene selection and managed inference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import secrets
import shlex
import signal
import subprocess
import sys
import threading
from urllib.parse import urlsplit,unquote,parse_qs
from collections import defaultdict

from shared.python.runtime import ROOT,ensure_environment,save_json
from shared.python.sam3 import parse_class_file
from tasks.flat_ground.python.project_semantics import CameraFrames
from tasks.nuscenes_sam3.python.semantic_schema import ZH,palette_for,normalize_colors
from tasks.nuscenes_sam3.python.instance_filter import validate_filter


def utcnow():return datetime.now(timezone.utc).isoformat()


DATASET_VERSIONS=('v1.0-mini','v1.0-trainval')
REQUIRED_TABLES=('scene','sample','sample_data','sensor','calibrated_sensor','ego_pose')


def installed_versions(root):
    root=Path(root)
    return [version for version in DATASET_VERSIONS if all((root/version/f'{table}.json').is_file() for table in REQUIRED_TABLES)]


def dataset_scenes(dataset_dir,version='v1.0-mini'):
    if version not in DATASET_VERSIONS:raise ValueError('不支持的数据版本 / Unsupported dataset version')
    dataset=CameraFrames(dataset_dir,version=version)
    train={row['scene_name'] for row in dataset.frames('mini_train')} if version=='v1.0-mini' else set()
    lidars={}
    for record in dataset.tables['sample_data'].values():
        calibration=dataset.tables['calibrated_sensor'][record['calibrated_sensor_token']]
        if record['is_key_frame'] and dataset.tables['sensor'][calibration['sensor_token']]['channel']=='LIDAR_TOP':
            lidars[record['sample_token']]=record
    samples_by_scene=defaultdict(list)
    for sample in dataset.tables['sample'].values():samples_by_scene[sample['scene_token']].append(sample)
    # Enumerate each media channel once. This avoids thousands of slow per-file
    # metadata calls through a Windows-mounted dataset during trainval discovery.
    filenames={}
    def exists(filename):
        relative=Path(filename);parent=relative.parent
        if parent not in filenames:
            directory=dataset.root/parent
            filenames[parent]={entry.name for entry in os.scandir(directory) if entry.is_file()} if directory.is_dir() else set()
        return relative.name in filenames[parent]
    scenes=[]
    for scene in sorted(dataset.tables['scene'].values(),key=lambda row:row['name']):
        samples=samples_by_scene[scene['token']]
        available=bool(samples);reason='' if samples else '没有关键帧 / No keyframes'
        for sample in samples:
            channels=dataset.channels.get(sample['token'],{})
            if len(channels)!=6 or any(not exists(record['filename']) for record in channels.values()):
                available=False;reason='缺少相机关键帧或图像 / Missing camera keyframes';break
            lidar=lidars.get(sample['token'])
            if lidar is None or not exists(lidar['filename']):
                available=False;reason='缺少LiDAR关键帧 / Missing LiDAR keyframes';break
        scenes.append({'name':scene['name'],'description':scene.get('description',''),'frames':len(samples),
                       'split':('mini_train' if scene['name'] in train else 'mini_val') if version=='v1.0-mini' else 'trainval',
                       'available':available,'reason':reason})
    return scenes


def initial_config(dataset_dir,classes_file,version='v1.0-mini'):
    scenes=dataset_scenes(dataset_dir,version)
    return {'version':version,'classes':[{'name':name,'display_name':ZH.get(name,name),'prompts':' | '.join(prompts),'plane':name in ('sidewalk','road')} for name,prompts in parse_class_file(classes_file)],
            'scenes':scenes,'defaults':{'samples':1,'start_index':0,'map_resolution':.2,'bev_resolution':.2,'bev_roi_size':100.,'max_edge':6.,'compute_grid_projection':False,'grid_projection_samples':3,'grid_projection_max_distance':80.,'occupancy_resolution':.5,'device':'cuda','point_filter':validate_filter(None)}}


def validated_classes(rows):
    if not isinstance(rows,list) or not 1<=len(rows)<=254:raise ValueError('请设置1至254个类别')
    normalized=[];seen=set()
    for row in rows:
        if not isinstance(row,dict):raise ValueError('类别格式不正确')
        name=str(row.get('name','')).strip();prompts=str(row.get('prompts','')).strip()
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_.-]{0,63}',name) or name.lower() in ('unassigned','unlabeled','ignore'):
            raise ValueError('类别名请使用英文字母开头的字母/数字/下划线，不能使用保留类别名')
        if name.casefold() in seen:raise ValueError(f'类别名重复：{name}')
        parts=[part.strip() for part in prompts.split('|')]
        if not all(parts) or len(prompts)>4000 or any(c in prompts for c in ('\n','\r','#','=')):
            raise ValueError(f'{name} 的提示词不能为空，多个提示词用 | 分隔，不能包含换行/#/=')
        if not isinstance(row.get('plane',False),bool):raise ValueError('平面选择必须是勾选值')
        display_name=row.get('display_name',ZH.get(name,name))
        if not isinstance(display_name,str):raise ValueError(f'{name} 的显示名必须是文本')
        display_name=display_name.strip() or ZH.get(name,name)
        if len(display_name)>128 or any(ord(char)<32 for char in display_name):
            raise ValueError(f'{name} 的显示名不能超过128个字符或包含控制字符')
        default_rgb=palette_for({len(normalized)+1:name})[len(normalized)+1]
        default_color='#'+''.join(f'{int(value):02x}' for value in default_rgb)
        color=normalize_colors({name:row.get('color',default_color)})[name]
        normalized.append({'name':name,'display_name':display_name,'color':color,'prompts':' | '.join(dict.fromkeys(parts)),'plane':row.get('plane',False)})
        seen.add(name.casefold())
    if not any(row['plane'] for row in normalized):raise ValueError('请至少勾选一个生成平面的类别')
    return normalized


def validated_config(raw,available):
    if not isinstance(raw,dict):raise ValueError('配置应为JSON对象')
    normalized=validated_classes(raw.get('classes'))
    scenes=raw.get('scenes')
    options={row['name']:row for row in available}
    if not isinstance(scenes,list) or not scenes or len(set(scenes))!=len(scenes):raise ValueError('请勾选至少一个场景，且不要重复')
    if any(not isinstance(name,str) or name not in options or not options[name]['available'] for name in scenes):raise ValueError('所选场景不存在或数据不完整')
    def integer(name,default,minimum):
        value=raw.get(name,default)
        if isinstance(value,bool) or not isinstance(value,int) or value<minimum:raise ValueError(f'{name} 必须是大于等于{minimum}的整数')
        return value
    count=integer('samples',1,0);start=integer('start_index',0,0)
    total=sum(options[name]['frames'] for name in scenes)
    if start>=total:raise ValueError('起始帧超出了所选场景总帧数')
    version=raw.get('version','v1.0-mini')
    if version not in DATASET_VERSIONS:raise ValueError('不支持的数据版本 / Unsupported dataset version')
    config={'version':version,'classes':normalized,'scenes':sorted(scenes),'samples':count,'start_index':start}
    for name,default,positive in (('map_resolution',.2,True),('bev_resolution',.2,True),('bev_roi_size',100.,True),('max_edge',6.,True),('occupancy_resolution',.5,True)):
        value=raw.get(name,default)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not -1e8<float(value)<1e8 or positive and value<=0:raise ValueError(f'{name} 数值无效')
        config[name]=float(value)
    cells=config['bev_roi_size']/config['bev_resolution']
    if abs(cells-round(cells))>1e-8 or cells*cells>16000000:
        raise ValueError('平面范围 / 分辨率必须为整数，且不能超过1600万格')
    if config['map_resolution']<.02:raise ValueError('地图分辨率最小0.02米/格')
    config['compute_grid_projection']=raw.get('compute_grid_projection',False)
    if not isinstance(config['compute_grid_projection'],bool):raise ValueError('栅格投影计算必须是勾选值')
    config['grid_projection_samples']=integer('grid_projection_samples',3,1)
    if config['grid_projection_samples']>8:raise ValueError('每轴采样数须为1至8的整数')
    distance=raw.get('grid_projection_max_distance',80.)
    if isinstance(distance,bool) or not isinstance(distance,(int,float)) or not 0<float(distance)<1e8:
        raise ValueError('栅格投影最大距离必须是有限正数')
    config['grid_projection_max_distance']=float(distance)
    config['device']=raw.get('device','cuda')
    if config['device'] not in ('cuda','cpu'):raise ValueError('device只能为cuda或cpu')
    config['point_filter']=validate_filter(raw.get('point_filter'))
    estimate=total-start if count==0 else min(count,total-start)
    return config,estimate


def command_for(config,dataset_dir,classes_file,output):
    point_filter=validate_filter(config.get('point_filter'))
    filter_args=[argument for key in ('sor_k','sor_std','dbscan_eps','dbscan_min_samples','cluster_policy','min_cluster_size','cluster_ratio')
                 for argument in ('--'+key.replace('_','-'),str(point_filter[key]))]
    projection_args=(['--compute-grid-projection','--grid-projection-samples',str(config.get('grid_projection_samples',3)),
                      '--grid-projection-max-distance',str(config.get('grid_projection_max_distance',80.))]
                     if config.get('compute_grid_projection',False) else [])
    return [sys.executable,'-u','-m','tasks.nuscenes_sam3.python.scene_export',
            '--dataset-dir',str(dataset_dir),'--version',config.get('version','v1.0-mini'),'--classes-file',str(classes_file),'--split','all',
            '--scenes',*config['scenes'],'--samples',str(config['samples']),'--start-index',str(config['start_index']),
            '--plane-classes',*[row['name'] for row in config['classes'] if row['plane']],
            '--map-resolution',str(config['map_resolution']),'--bev-resolution',str(config['bev_resolution']),
            '--bev-roi-size',str(config['bev_roi_size']),'--max-edge',str(config['max_edge']),'--occupancy-resolution',str(config['occupancy_resolution']),
            '--device',config['device'],'--output-dir',str(output),
            '--point-filter',point_filter['algorithm'],*filter_args,*projection_args]


class Generator:
    def __init__(self,dataset_dir,classes_file,output_root):
        self.dataset_dir=Path(dataset_dir).resolve();self.classes_file=Path(classes_file).resolve()
        versions=installed_versions(self.dataset_dir)
        if not versions:raise FileNotFoundError(f'没有可用的nuScenes元数据版本 / No supported nuScenes metadata under {self.dataset_dir}')
        bootstrap_version='v1.0-mini' if 'v1.0-mini' in versions else versions[0]
        self.output_root=Path(output_root).resolve();self.bootstrap=dict(initial_config(self.dataset_dir,self.classes_file,bootstrap_version))
        self.bootstrap['versions']=versions
        self.bootstrap['classes']=validated_classes(self.bootstrap['classes'])
        self.bootstrap['capabilities']={'class_persistence':True,'class_display_names':True,'class_colors':True,'grid_projection':True}
        self.token=secrets.token_urlsafe(32);self.runs={};self.lock=threading.RLock()
        self.scene_lock=threading.RLock()
        self.scene_cache={bootstrap_version:self.bootstrap['scenes']}
        self.output_root.mkdir(parents=True,exist_ok=True)
        self.classes_path=self.output_root/'semantic_classes.json'
        self.bootstrap['classes_path']=str(self.classes_path)
        if self.classes_path.is_file():
            try:self.bootstrap['classes']=validated_classes(json.loads(self.classes_path.read_text(encoding='utf-8'))['classes'])
            except (ValueError,KeyError,TypeError,OSError) as error:
                self.bootstrap['classes_error']=f'本地类别读取失败 / Cannot load saved classes: {error}'
        for path in self.output_root.glob('*/job.json'):
            try:
                run=json.loads(path.read_text());identifier=run['id']
                if re.fullmatch(r'[A-Za-z0-9_-]+',identifier):
                    run['directory']=str(path.parent.resolve())
                    if run.get('status') in ('running','stopping','starting'):run['status']='interrupted'
                    self.runs[identifier]=run
            except (ValueError,KeyError,OSError):continue

    def scenes_for(self,version):
        if version not in self.bootstrap['versions']:raise ValueError('所选数据版本未安装或元数据不完整 / Dataset version is not installed')
        with self.scene_lock:
            if version not in self.scene_cache:self.scene_cache[version]=dataset_scenes(self.dataset_dir,version)
            return {'version':version,'scenes':self.scene_cache[version]}

    def validate_run(self,raw):
        if not isinstance(raw,dict):raise ValueError('配置应为JSON对象')
        version=raw.get('version',self.bootstrap.get('version','v1.0-mini'))
        return validated_config({**raw,'version':version},self.scenes_for(version)['scenes'])

    def save_classes(self,raw):
        if not isinstance(raw,dict):raise ValueError('配置应为JSON对象')
        classes=validated_classes(raw.get('classes'))
        with self.lock:
            temporary=self.classes_path.with_suffix('.json.tmp')
            save_json(temporary,{'classes':classes})
            os.replace(temporary,self.classes_path)
            self.bootstrap['classes']=classes
            self.bootstrap.pop('classes_error',None)
        return {'classes':classes,'classes_path':str(self.classes_path)}

    def snapshot(self,run):
        data={key:value for key,value in run.items() if not key.startswith('_')}
        log=Path(run['directory'])/'run.log'
        if log.is_file():
            with log.open('rb') as stream:
                stream.seek(0,2);end=stream.tell();stream.seek(max(0,end-48000));data['log_tail']=stream.read().decode('utf-8',errors='replace')
        else:data['log_tail']=''
        return data

    def list_runs(self):
        with self.lock:return [self.snapshot(run) for run in sorted(self.runs.values(),key=lambda row:row['created_at'],reverse=True)]

    def preview(self,raw):
        config,estimated=self.validate_run(raw)
        command=command_for(config,self.dataset_dir,Path('<本次类别配置.txt>'),Path('<自动创建的新输出目录>'))
        return {'command':shlex.join(command),'estimated_frames':estimated,'total_images':estimated*6,
                'selected_scenes':config['scenes'],'config':config}

    def persist(self,run):save_json(Path(run['directory'])/'job.json',{key:value for key,value in run.items() if not key.startswith('_')})

    def start(self,raw):
        config,estimate=self.validate_run(raw)
        with self.lock:
            if any(row['status'] in ('starting','running','stopping') for row in self.runs.values()):raise ValueError('已有任务运行中，请等待完成或停止后再生成')
            identifier=datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+secrets.token_hex(2)
            directory=self.output_root/identifier;directory.mkdir()
            classes=directory/'classes.txt';classes.write_text('\n'.join(f"{row['name']} = {row['prompts']}" for row in config['classes'])+'\n',encoding='utf-8')
            save_json(classes.with_suffix('.display_names.json'),{row['name']:row['display_name'] for row in config['classes']})
            save_json(classes.with_suffix('.colors.json'),{row['name']:row['color'] for row in config['classes']})
            save_json(directory/'ui_config.json',config)
            output=directory/'output';command=command_for(config,self.dataset_dir,classes,output)
            run={'id':identifier,'directory':str(directory),'output_dir':str(output),'created_at':utcnow(),
                 'status':'starting','scenes':config['scenes'],'config':config,'command':shlex.join(command),
                 'progress':{'current':0,'total':estimate*6,'unit':'camera_images'},'estimated_frames':estimate,
                 'result_url':None,'exit_code':None}
            self.runs[identifier]=run;self.persist(run)
            worker=threading.Thread(target=self._worker,args=(run,command),daemon=True);run['_thread']=worker;worker.start()
            return self.snapshot(run)

    def _worker(self,run,command):
        directory=Path(run['directory'])
        try:
            env=dict(os.environ,PYTHONUNBUFFERED='1')
            with (directory/'run.log').open('w',encoding='utf-8',buffering=1) as log:
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                                         encoding='utf-8',errors='replace',bufsize=1,start_new_session=True)
                with self.lock:
                    run['_process']=process;run['status']='running';run['pid']=process.pid;self.persist(run)
                with process.stdout:
                    for line in process.stdout:
                        log.write(line)
                        if re.search(r'^CAM_[A-Z_]+: (?:six-class|configured).*instances=',line):
                            with self.lock:run['progress']['current']=min(run['progress']['total'],run['progress']['current']+1)
                code=process.wait()
            with self.lock:
                cancelled=run['status']=='stopping'
                run.update(status='stopped' if cancelled else 'completed' if code==0 else 'failed',exit_code=code,finished_at=utcnow())
                if code==0 and (Path(run['output_dir'])/'scenes/index.html').is_file():
                    run['result_url']=f"/results/{run['id']}/scenes/index.html";run['progress']['current']=run['progress']['total']
                self.persist(run)
        except BaseException as error:
            with self.lock:
                run.update(status='failed',error=str(error),finished_at=utcnow());self.persist(run)

    def stop(self,identifier):
        with self.lock:
            if identifier not in self.runs:raise KeyError(identifier)
            run=self.runs[identifier];process=run.get('_process')
            if run['status']=='starting' and process is None:raise ValueError('任务正在启动，请稍后停止')
            if process is not None and process.poll() is None:
                run['status']='stopping';self.persist(run)
                os.killpg(process.pid,signal.SIGTERM)
                threading.Thread(target=self._finish_stop,args=(process,),daemon=True).start()
            return self.snapshot(run)

    @staticmethod
    def _finish_stop(process):
        try:process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass

    def result_path(self,identifier,relative):
        if identifier not in self.runs:raise FileNotFoundError('未知任务')
        base=Path(self.runs[identifier]['output_dir']).resolve();target=(base/relative).resolve()
        if not target.is_relative_to(base) or not target.is_file():raise FileNotFoundError('结果文件不存在')
        return target


def handler_class(generator):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def json(self,value,status=200):
            body=json.dumps(value,ensure_ascii=False,allow_nan=False).encode('utf-8')
            self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Cache-Control','no-store')
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):
            path=unquote(urlsplit(self.path).path)
            try:
                if path=='/api/config':return self.json({**generator.bootstrap,'token':generator.token,'runs':generator.list_runs()})
                if path=='/api/scenes':
                    version=parse_qs(urlsplit(self.path).query).get('version',[generator.bootstrap.get('version','v1.0-mini')])[0]
                    return self.json(generator.scenes_for(version))
                if path=='/api/state':return self.json({'runs':generator.list_runs()})
                if path.startswith('/api/runs/'):
                    identifier=path.split('/')[-1]
                    with generator.lock:
                        if identifier not in generator.runs:raise FileNotFoundError('未知任务')
                        return self.json(generator.snapshot(generator.runs[identifier]))
                if path in ('/','/index.html'):target=ROOT/'tasks/nuscenes_sam3/web/generator.html'
                elif path.startswith('/results/'):
                    parts=path.split('/',3)
                    if len(parts)!=4:raise FileNotFoundError('结果路径无效')
                    target=generator.result_path(parts[2],parts[3])
                else:raise FileNotFoundError('未找到页面')
                body=target.read_bytes();self.send_response(200)
                content=mimetypes.guess_type(target.name)[0] or 'application/octet-stream'
                self.send_header('Content-Type',content+('; charset=utf-8' if content.startswith('text/') else ''))
                self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
            except (KeyError,FileNotFoundError) as error:self.json({'error':str(error)},404)
            except (ValueError,OSError) as error:self.json({'error':str(error)},400)
        def do_POST(self):
            if self.headers.get('X-Generator-Token')!=generator.token:return self.json({'error':'请求令牌无效，请刷新页面'},403)
            try:
                length=int(self.headers.get('Content-Length','0'))
                if length<0 or length>200000:raise ValueError('配置过大')
                raw=json.loads(self.rfile.read(length) or b'{}');path=urlsplit(self.path).path
                if path=='/api/classes':return self.json(generator.save_classes(raw))
                if path=='/api/preview':return self.json(generator.preview(raw))
                if path=='/api/start':return self.json(generator.start(raw),201)
                match=re.fullmatch(r'/api/runs/([A-Za-z0-9_-]+)/stop',path)
                if match:return self.json(generator.stop(match.group(1)))
                self.json({'error':'未找到接口'},404)
            except (ValueError,TypeError,KeyError) as error:self.json({'error':str(error)},400)
            except OSError as error:self.json({'error':str(error)},500)
    return Handler


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir',type=Path,default=ROOT/'datasets/nuscenes')
    p.add_argument('--classes-file',type=Path,default=ROOT/'tasks/nuscenes_sam3/config/six_classes.txt')
    p.add_argument('--output-root',type=Path,default=ROOT/'outputs/nuscenes_generator')
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8765)
    args=p.parse_args(argv);ensure_environment()
    generator=Generator(args.dataset_dir,args.classes_file,args.output_root)
    server=ThreadingHTTPServer((args.host,args.port),handler_class(generator))
    print(f'nuScenes 生成器 UI: http://{args.host}:{server.server_port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        for identifier in list(generator.runs):
            if generator.runs[identifier]['status'] in ('running','stopping'):generator.stop(identifier)
        server.server_close()
    return 0


if __name__=='__main__':raise SystemExit(main())

'use strict';
// node dynamic_classes_browser.cjs [http://127.0.0.1:9339]
// Uses a running Chromium CDP endpoint; fixtures contain one frame and no model output.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const os=require('node:os');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
const endpoint=process.argv[2]||'http://127.0.0.1:9339';
const packed=(values,type)=>Buffer.from(new type(values).buffer).toString('base64');
const template=fs.readFileSync(path.resolve(__dirname,'../web/scene_viewer.html'),'utf8');
assert.equal(template.split('__SCENE_DATA__').length,2);
function fixture(planes,maxId=7,projection=false){
 const classes=Array.from({length:maxId+1},(_,id)=>({id,name:id===3?'crosswalk':id===7?'rider':`class-${id}`,label:id===3?'人行横道 / Crosswalk':`类别 ${id}`,color:id===0?[255,255,255]:[(id*53)%220+20,(id*79)%220+20,(id*31)%220+20]}));
 const surface={xyz:packed([-1,-1,0,1,-1,0,1,1,0],Float32Array),count:3};
 const surfaces={polygon:{}};for(const id of planes)surfaces.polygon[id]=surface;
 if(projection)surfaces.grid_projection={...surfaces.polygon};
 const mapClass=planes.includes(3)?3:2;const runs=packed([0,1,mapClass],Uint32Array),map={runs,cells:1,conflict_cells:0};
 return {scene:'dynamic-fixture',classes,plane_class_ids:planes,metadata:{plane_class_ids:[1,2]},bounds:{min:[-2,-2,-1],max:[2,2,1]},frames:[{sample_token:'one-frame',ego:[0,0,0],points:{xyz:packed([0,0,0,1,0,0,0,1,0],Float32Array),labels:packed([3,7,maxId],Uint8Array),count:3},surfaces}],bev_map:{grid:{width:2,height:2,resolution:1,x_min:-1,y_min:-1},frames:[{polygon:map,...(projection?{grid_projection:{runs:packed([1,2,mapClass],Uint32Array),cells:2,conflict_cells:1}}:{})}],fused:{polygon:map,...(projection?{grid_projection:{runs:packed([1,2,mapClass],Uint32Array),cells:2,conflict_cells:1}}:{})}}};
}
async function verify(file,planes,maxId,projection=false){
 const target=await fetch(endpoint+'/json/new?'+encodeURIComponent(pathToFileURL(file).href),{method:'PUT'}).then(r=>r.json());
 const ws=new WebSocket(target.webSocketDebuggerUrl);let seq=0;const pending=new Map(),errors=[];
 await new Promise((resolve,reject)=>{ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true});});
 ws.addEventListener('message',event=>{const m=JSON.parse(event.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);if(m.error)p.reject(Error(JSON.stringify(m.error)));else p.resolve(m.result);}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);});
 const send=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
 const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
 const state=()=>evaluate('window.__sceneViewer.getState()');
 const action=async expression=>evaluate(`(()=>{${expression};window.__sceneViewer.render();return window.__sceneViewer.getState()})()`);
 const check=async(id,value)=>action(`const e=document.getElementById(${JSON.stringify(id)});e.checked=${value};e.dispatchEvent(new Event('change',{bubbles:true}))`);
 const classCheck=async(id,value)=>action(`const e=document.querySelector('[data-class="${id}"]');e.checked=${value};e.dispatchEvent(new Event('change',{bubbles:true}))`);
 try{
  await send('Runtime.enable');await send('Page.enable');await send('Page.navigate',{url:pathToFileURL(file).href});
  await evaluate(`new Promise((resolve,reject)=>{const end=performance.now()+30000;function tick(){if(window.__sceneViewerReady||document.getElementById('error')&&!document.getElementById('error').hidden)resolve();else if(performance.now()>end)reject(Error('Viewer timeout'));else requestAnimationFrame(tick)}tick()})`);
  assert.equal(await evaluate("document.getElementById('error').hidden?'':document.getElementById('error').textContent"),'');
  assert.equal(await evaluate('document.querySelectorAll("#ipmLayer, #mapMethod").length'),0,'Removed IPM and map method controls must be absent');
  let s=await state();assert.deepEqual(s.methods,['polygon']);assert.deepEqual(s.planeClassIds,planes);assert.equal(s.availableClassIds.length,maxId+1);assert.equal(s.pointCount,planes.includes(3)?2:3);assert.equal(s.triangleCount,planes.length);assert.ok(!s.classIds.includes(0));
  s=await check('polygonLayer',false);assert.deepEqual(s.methods,[]);assert.equal(s.triangleCount,0);assert.equal(s.surfaceDrawCalls,0);
  s=await check('polygonLayer',true);assert.deepEqual(s.methods,['polygon']);assert.equal(s.triangleCount,planes.length);
  assert.equal(await evaluate('document.getElementById("gridProjectionControl").hidden'),!projection);
  if(projection){
   s=await check('grid_projectionLayer',true);assert.equal(s.triangleCount,planes.length*2);
   s=await check('polygonLayer',false);assert.equal(s.triangleCount,planes.length);assert.deepEqual(s.methods,['grid_projection']);
   s=await check('grid_projectionLayer',false);assert.equal(s.triangleCount,0);
   s=await check('polygonLayer',true);
  }
  await action("document.getElementById('selectNone').click()");s=await classCheck(3,true);
  assert.equal(s.triangleCount,planes.includes(3)?1:0);assert.equal(s.pointCount,planes.includes(3)?0:1);
  s=await check('groundPointLayer',true);assert.equal(s.pointCount,1);
  await classCheck(3,false);s=await classCheck(maxId,true);assert.equal(s.pointCount,maxId===7?2:1);assert.equal(s.triangleCount,0);
  await action("document.getElementById('selectAll').click()");assert.equal((await state()).classIds.length,maxId+1);
  await action("document.getElementById('selectNone').click()");await classCheck(planes.includes(3)?3:2,true);
  s=await action("document.getElementById('bevMapView').click()");assert.equal(s.mode,'bev-map');assert.equal(s.mapMethod,'polygon');assert.equal(s.mapCells,1);
  if(projection){
   s=await action("const selector=document.getElementById('groundMapMethod');selector.value='grid_projection';selector.dispatchEvent(new Event('change'))");
   assert.equal(s.mapMethod,'grid_projection');assert.equal(s.mapCells,2);assert.equal(s.mapConflictCells,1);
   await action('window.__sceneViewer.setAllFrames(true)');assert.equal((await state()).mapCells,2);
   await classCheck(planes.includes(3)?3:2,false);assert.equal((await state()).mapCells,0);
   await classCheck(planes.includes(3)?3:2,true);
   s=await action("document.getElementById('scene3dView').click()");assert.equal(s.mode,'3d');
   await action("document.getElementById('bevMapView').click()");
  }
  s=await classCheck(planes.includes(3)?3:2,false);assert.equal(s.mapCells,0);
  assert.deepEqual(errors,[]);console.log(JSON.stringify({planes,maxId,checks:'polygon-only fixture, removed controls absent, polygon visibility, dynamic shader, point and plane membership, class toggles, select all, BEV cells, one-frame fixture'}));
 }finally{await send('Page.close').catch(()=>{});ws.close();}
}
async function main(){
 const directory=fs.mkdtempSync(path.join(os.tmpdir(),'scene-dynamic-classes-'));
 try{for(const [planes,maxId,projection] of [[[1,2,3],7,false],[[1,2],7,false],[[1,2,3],254,false],[[1,2,3],7,true]]){const file=path.join(directory,`fixture-${planes.length}-${maxId}-${projection}.html`);fs.writeFileSync(file,template.replace('__SCENE_DATA__',JSON.stringify(fixture(planes,maxId,projection))));await verify(file,planes,maxId,projection);}}
 finally{const intended=path.resolve(directory);assert.equal(path.dirname(intended),path.resolve(os.tmpdir()));assert.ok(path.basename(intended).startsWith('scene-dynamic-classes-'));fs.rmSync(intended,{recursive:true,force:true});}
}
main().catch(error=>{console.error(error.stack);process.exitCode=1;});

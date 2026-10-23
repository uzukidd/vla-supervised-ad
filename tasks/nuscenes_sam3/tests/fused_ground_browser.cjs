'use strict';
// node tasks/nuscenes_sam3/tests/fused_ground_browser.cjs [http://127.0.0.1:9341]
// Native Node/CDP regression; requires an already running Chromium, no model output.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const os=require('node:os');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
const endpoint=process.argv[2]||'http://127.0.0.1:9341';
const packed=(values,type)=>Buffer.from(new type(values).buffer).toString('base64');
const classes=[{id:0,name:'unknown',color:[255,255,255]},{id:1,name:'ground-one',color:[240,80,40]},{id:2,name:'ground-two',color:[40,160,240]},{id:3,name:'object',color:[80,220,80]}];
const triangle=(frame,id,projection=false)=>[10,20,projection?0:frame+id,12,20,projection?0:frame+id,12,22,projection?0:frame+id];
const surface=xyz=>({xyz:packed(xyz,Float32Array),count:xyz.length/3});
const map=(runs,heights)=>({runs:packed(runs,Uint32Array),cells:3,conflict_cells:1,class_heights_scene:heights});
function fixture(projection){
 const frames=Array.from({length:3},(_,index)=>{
  const surfaces={polygon:{1:surface(triangle(index,1)),2:surface(triangle(index,2))}};
  if(projection)surfaces.grid_projection={1:surface(triangle(index,1,true)),2:surface(triangle(index,2,true))};
  return {sample_token:`frame-${index}`,ego:[10+index,20,0],points:{xyz:packed([10+index,20,3],Float32Array),labels:packed([3],Uint8Array),count:1},surfaces};
 });
 // Both per-frame classes occupy the same cell. The saved fused winners are authoritative.
 // Polygon cell 0 is class 2; cells 1 and 2 are class 1 in ONE run crossing a row.
 const polygon=map([0,1,2,1,2,1],{'1':1.25,'2':2.5});
 // Projection has different winners and footprint, with no class 1 winner anywhere.
 const projected=map([1,3,2],{'1':0,'2':0});
 return {scene:'fused-ground-test',classes,plane_class_ids:[1,2],bounds:{min:[9,19,-1],max:[15,25,5]},frames,bev_map:{grid:{width:2,height:2,resolution:2,x_min:10,y_min:20},frames:frames.map((_,index)=>({polygon:{runs:packed([0,1,index===1?2:1],Uint32Array),cells:1,conflict_cells:1},...(projection?{grid_projection:{runs:packed([0,1,index===1?1:2],Uint32Array),cells:1,conflict_cells:1}}:{})})),fused:{polygon,...(projection?{grid_projection:projected}:{})}}};
}
// Observe real uploads and native draw calls; WebGL 1 has no getBufferSubData.
function installProbe(){
 const probe=window.__groundProbe={draws:[],uploads:[],raster:null};
 const buffers=new WeakMap(),locations=new WeakMap();let color=[],opacity=1,offset=0;
 const proto=WebGLRenderingContext.prototype;
 const bufferData=proto.bufferData;
 proto.bufferData=function(target,data,...rest){
  if(target===this.ARRAY_BUFFER&&data instanceof Float32Array){const xyz=Array.from(data);buffers.set(this.getParameter(this.ARRAY_BUFFER_BINDING),xyz);probe.uploads.push(xyz);}
  return bufferData.call(this,target,data,...rest);
 };
 const getUniformLocation=proto.getUniformLocation;
 proto.getUniformLocation=function(program,name){const location=getUniformLocation.call(this,program,name);if(location)locations.set(location,name);return location;};
 const uniform3fv=proto.uniform3fv;
 proto.uniform3fv=function(location,value){if(locations.get(location)==='u_color')color=Array.from(value);return uniform3fv.call(this,location,value);};
 const uniform1f=proto.uniform1f;
 proto.uniform1f=function(location,value){if(locations.get(location)==='u_opacity')opacity=value;if(locations.get(location)==='u_offset')offset=value;return uniform1f.call(this,location,value);};
 const drawArrays=proto.drawArrays;
 proto.drawArrays=function(mode,first,count){
  if(mode===this.TRIANGLES)probe.draws.push({first,count,xyz:buffers.get(this.getParameter(this.ARRAY_BUFFER_BINDING)),color:color.slice(),opacity,offset});
  return drawArrays.call(this,mode,first,count);
 };
 const drawImage=CanvasRenderingContext2D.prototype.drawImage;
 CanvasRenderingContext2D.prototype.drawImage=function(source,...args){
  if(this.canvas.id==='bevMapCanvas'&&source.width===2&&source.height===2)probe.raster=Array.from(source.getContext('2d').getImageData(0,0,2,2).data);
  return drawImage.call(this,source,...args);
 };
}
const quad=(x0,y0,x1,y1,z)=>[x0,y0,z,x1,y0,z,x1,y1,z,x0,y0,z,x1,y1,z,x0,y1,z];
const polygonGeometry=[{id:1,xyz:[...quad(12,22,14,24,1.25),...quad(10,20,12,22,1.25)]},{id:2,xyz:quad(10,22,12,24,2.5)}];
const projectionGeometry=[{id:2,xyz:[...quad(12,22,14,24,0),...quad(10,20,14,22,0)]}];
function near(actual,expected,message){assert.equal(actual.length,expected.length,message);actual.forEach((value,i)=>assert.ok(Math.abs(value-expected[i])<1e-6,`${message}: component ${i}: ${value} != ${expected[i]}`));}
function assertDraws(result,expected,opacity=result.state.planeOpacity,offset=.035){
 assert.equal(result.state.surfaceDrawCalls,expected.length,'One draw per visible winning class, independent of frame count');
 assert.equal(result.draws.length,expected.length,'Native TRIANGLES calls match state');
 assert.equal(result.state.triangleCount,expected.reduce((n,g)=>n+g.xyz.length/9,0));
 expected.forEach((group,index)=>{const draw=result.draws[index];assert.equal(draw.first,0);assert.equal(draw.count,group.xyz.length/3);assert.deepEqual(draw.xyz,group.xyz,'Uploaded xyz uses the saved winners, row bounds and scene height');near(draw.color,classes[group.id].color.map(v=>v/255),'Saved class color');assert.equal(draw.opacity,opacity);assert.equal(draw.offset,group.offset??offset);});
}
function raster(ids){return ids.flatMap(id=>id===null?[0,0,0,0]:[...classes[id].color,255]);}
async function verify(file,projection){
 let target,ws;const pending=new Map(),errors=[];let seq=0;
 const send=(method,params={})=>new Promise((resolve,reject)=>{
  const id=++seq,timer=setTimeout(()=>{pending.delete(id);reject(Error('CDP timeout: '+method));},15000);
  pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}));
 });
 const evaluate=async expression=>{const response=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(response.exceptionDetails)throw Error(JSON.stringify(response.exceptionDetails));return response.result.value;};
 const action=expression=>evaluate(`(()=>{const p=window.__groundProbe;p.draws=[];p.uploads=[];p.raster=null;${expression};window.__sceneViewer.render();return {state:window.__sceneViewer.getState(),draws:p.draws,uploads:p.uploads,raster:p.raster};})()`);
 const check=(id,value)=>`{const e=document.getElementById(${JSON.stringify(id)});e.checked=${value};e.dispatchEvent(new Event('change',{bubbles:true}));}`;
 const classCheck=(id,value)=>`{const e=document.querySelector('[data-class="${id}"]');e.checked=${value};e.dispatchEvent(new Event('change',{bubbles:true}));}`;
 const selectMap=method=>`{const e=document.getElementById('groundMapMethod');e.value=${JSON.stringify(method)};e.dispatchEvent(new Event('change'));}`;
 try{
  const response=await fetch(endpoint+'/json/new?about:blank',{method:'PUT',signal:AbortSignal.timeout(15000)});assert.ok(response.ok,'CDP target creation');target=await response.json();
  ws=new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(Error('CDP WebSocket open timeout')),15000);ws.addEventListener('open',()=>{clearTimeout(timer);resolve();},{once:true});ws.addEventListener('error',event=>{clearTimeout(timer);reject(Error(event.message||'CDP WebSocket error'));},{once:true});});
  ws.addEventListener('message',event=>{const message=JSON.parse(event.data);if(message.id){const p=pending.get(message.id);if(!p)return;pending.delete(message.id);clearTimeout(p.timer);message.error?p.reject(Error(JSON.stringify(message.error))):p.resolve(message.result);}else if(message.method==='Runtime.exceptionThrown')errors.push(message.params.exceptionDetails);});
  await send('Runtime.enable');await send('Page.enable');await send('Emulation.setDeviceMetricsOverride',{width:1400,height:1000,deviceScaleFactor:1,mobile:false});
  await send('Page.addScriptToEvaluateOnNewDocument',{source:`(${installProbe.toString()})()`});
  await send('Page.navigate',{url:pathToFileURL(file).href});
  await evaluate(`new Promise((resolve,reject)=>{const end=performance.now()+20000;function tick(){const error=document.getElementById('error');if(error&&!error.hidden)reject(Error(error.textContent));else if(window.__sceneViewerReady)resolve();else if(performance.now()>end)reject(Error('Viewer ready timeout'));else setTimeout(tick,20);}tick();})`);
  let result=await action('');
  assertDraws(result,[1,2].map(id=>({id,xyz:triangle(0,id)})));assert.equal(result.state.pointCount,1);assert.equal(result.state.pointDrawCalls,1);
  assert.equal(await evaluate('document.getElementById("gridProjectionControl").hidden'),!projection);
  result=await action('window.__sceneViewer.setAllFrames(true)');
  assertDraws(result,polygonGeometry);assert.equal(result.state.pointCount,3);assert.equal(result.state.pointDrawCalls,3);
  for(const group of polygonGeometry)assert.ok(result.uploads.some(xyz=>JSON.stringify(xyz)===JSON.stringify(group.xyz)),'First fused render uploads the exact row-split mesh');
  const fusedDraws=result.draws;
  for(const frame of [1,2,0]){result=await action(`window.__sceneViewer.setFrame(${frame})`);assertDraws(result,polygonGeometry);assert.deepEqual(result.draws,fusedDraws,'Timeline changes leave fused geometry unchanged');assert.equal(result.state.pointCount,3);assert.equal(result.state.pointDrawCalls,3);}
  result=await action("const opacity=document.getElementById('planeOpacity');opacity.value='0.35';opacity.dispatchEvent(new Event('input'))");assertDraws(result,polygonGeometry,.35);assert.equal(result.state.planeOpacity,.35);
  result=await action(classCheck(1,false));assertDraws(result,[polygonGeometry[1]],.35);assert.equal(result.state.pointCount,3);
  result=await action("document.getElementById('bevMapView').click()");assert.equal(result.state.mapCells,1);assert.deepEqual(result.raster,raster([2,null,null,null]),'Hiding the winner leaves empty cells, with no losing-class fallback');assert.equal(result.state.pointCount,3);
  result=await action(classCheck(1,true));assert.equal(result.state.mapCells,3);assert.deepEqual(result.raster,raster([2,1,1,null]));assert.equal(result.state.mapConflictCells,1);
  result=await action("document.getElementById('scene3dView').click()");assertDraws(result,polygonGeometry,.35);
  result=await action(classCheck(2,false));assertDraws(result,[polygonGeometry[0]],.35);
  result=await action("document.getElementById('bevMapView').click()");assert.equal(result.state.mapCells,2);assert.deepEqual(result.raster,raster([null,1,1,null]));
  result=await action(classCheck(1,false));assert.equal(result.state.mapCells,0);assert.deepEqual(result.raster,raster([null,null,null,null]));
  result=await action("document.getElementById('scene3dView').click()");assertDraws(result,[],.35);assert.equal(result.state.pointCount,3);
  await action(classCheck(1,true)+classCheck(2,true));
  if(projection){
   result=await action(check('grid_projectionLayer',true));assertDraws(result,[...polygonGeometry,...projectionGeometry.map(group=>({...group,offset:.055}))],.35);
   assert.ok(result.uploads.some(xyz=>JSON.stringify(xyz)===JSON.stringify(projectionGeometry[0].xyz)),'Projection first render uploads zero-height fused mesh');
  }
  // Restore single-frame mode at a different frame to prove original meshes remain available.
  result=await action(check('grid_projectionLayer',false)+'window.__sceneViewer.setFrame(2);window.__sceneViewer.setAllFrames(false)');
  assertDraws(result,[1,2].map(id=>({id,xyz:triangle(2,id)})),.35);assert.equal(result.state.pointCount,1);assert.equal(result.state.pointDrawCalls,1);
  if(projection){
   result=await action(check('polygonLayer',false)+check('grid_projectionLayer',true));assertDraws(result,[1,2].map(id=>({id,xyz:triangle(2,id,true)})),.35,.055);
   result=await action('window.__sceneViewer.setAllFrames(true)');assertDraws(result,projectionGeometry,.35,.055);assert.equal(result.state.pointCount,3);assert.equal(result.state.pointDrawCalls,3);
   result=await action(classCheck(2,false));assertDraws(result,[],.35,.055);assert.equal(result.state.pointCount,3);
   result=await action("document.getElementById('bevMapView').click();"+selectMap('grid_projection'));assert.equal(result.state.mapCells,0);assert.deepEqual(result.raster,raster([null,null,null,null]),'Class 1 loses every projection cell and must never reappear');
   result=await action(classCheck(2,true));assert.equal(result.state.mapCells,3);assert.deepEqual(result.raster,raster([null,2,2,2]));assert.equal(result.state.pointCount,3);
   result=await action("document.getElementById('scene3dView').click();window.__sceneViewer.setFrame(1)");assertDraws(result,projectionGeometry,.35,.055);assert.equal(result.state.pointCount,3);
   result=await action('window.__sceneViewer.setAllFrames(false)');assertDraws(result,[1,2].map(id=>({id,xyz:triangle(1,id,true)})),.35,.055);assert.equal(result.state.pointCount,1);
   result=await action(check('grid_projectionLayer',false));assertDraws(result,[],.35);assert.equal(result.state.pointCount,1);
  }
  assert.deepEqual(errors,[],'No browser runtime exceptions');
  console.log(JSON.stringify({ok:true,projection,checks:'single-frame restoration; authoritative fused winners; row-crossing RLE split and exact xyz/heights; native TRIANGLES counts/colors/opacity; class filters with no fallback; frame invariance; all three point frames; BEV/3D cell parity'}));
 }finally{
  if(ws&&ws.readyState===WebSocket.OPEN)await send('Page.close').catch(()=>{});
  if(ws)ws.close();
  for(const p of pending.values()){clearTimeout(p.timer);p.reject(Error('CDP session closed'));}pending.clear();
  if(target)await fetch(endpoint+'/json/close/'+target.id,{signal:AbortSignal.timeout(5000)}).catch(()=>{});
 }
}
async function main(){
 const template=fs.readFileSync(path.resolve(__dirname,'../web/scene_viewer.html'),'utf8');assert.equal(template.split('__SCENE_DATA__').length,2);
 const directory=fs.mkdtempSync(path.join(os.tmpdir(),'scene-fused-ground-'));
 try{for(const projection of [false,true]){const file=path.join(directory,`fixture-${projection}.html`);fs.writeFileSync(file,template.replace('__SCENE_DATA__',JSON.stringify(fixture(projection))));await verify(file,projection);}}
 finally{const intended=path.resolve(directory);assert.equal(path.dirname(intended),path.resolve(os.tmpdir()));assert.ok(path.basename(intended).startsWith('scene-fused-ground-'));fs.rmSync(intended,{recursive:true,force:true});}
}
main().catch(error=>{console.error(error.stack);process.exitCode=1;});

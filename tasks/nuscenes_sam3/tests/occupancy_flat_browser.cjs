'use strict';
// node occupancy_flat_browser.cjs [http://127.0.0.1:9341]
// Uses the parent-managed Chromium CDP endpoint; writes only a temporary fixture.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const os=require('node:os');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
const endpoint=process.argv[2]||'http://127.0.0.1:9341';
const packed=(values,type)=>Buffer.from(new type(values).buffer).toString('base64');
const template=fs.readFileSync(path.resolve(__dirname,'../web/scene_viewer.html'),'utf8');
assert.equal(template.split('__SCENE_DATA__').length,2);
const square=(x,y)=>({xyz:packed([x,y,0,x+1,y,0,x+1,y+1,0,x,y,0,x+1,y+1,0,x,y+1,0],Float32Array),count:6});
const triangle=(x,y)=>({xyz:packed([x,y,0,x+1,y,0,x,y+1,0],Float32Array),count:3});
function fixture(){
 const classes=[{id:0,name:'unassigned',label:'Unassigned',color:[255,255,255]},{id:1,name:'surface-a',label:'Surface A',color:[235,40,30]},{id:2,name:'object',label:'Object',color:[20,80,230]},{id:3,name:'surface-b',label:'Surface B',color:[25,190,70]}];
 const empty={xyz:packed([],Float32Array),count:0};
 const frame=(token,x)=>({sample_token:token,ego:[0,0,0],points:{xyz:packed([x,0,1,x+.2,0,1,x+.4,0,1],Float32Array),labels:packed([1,2,3],Uint8Array),count:3},surfaces:{polygon:{'1':empty,'3':empty},occupancy_flat:{'1':square(x,0),'3':square(x+2,0)}}});
 const mapRuns=packed([0,1,1,1,1,3],Uint32Array),emptyRuns=packed([],Uint32Array),map={runs:mapRuns,cells:2,conflict_cells:0},polygon={runs:emptyRuns,cells:0,conflict_cells:0};
 const occupancyFrame={indices:[[0,0,0]],point_counts:[1],class_ids:[2],voxel_count:1,point_count:1};
 return {scene:'occupancy-flat-contract',classes,plane_class_ids:[1,3],bounds:{min:[-1,-1,-1],max:[5,3,2]},frames:[frame('frame-a',0),frame('frame-b',10)],fused_surfaces:{occupancy_flat:{'1':triangle(-3,0),'3':triangle(3,0)}},occupancy_grid:{resolution:1,origin:[0,0,0],frames:[occupancyFrame,{indices:[[1,0,0]],point_counts:[1],class_ids:[2],voxel_count:1,point_count:1}],fused:occupancyFrame},bev_map:{grid:{width:2,height:1,resolution:1,x_min:0,y_min:0},frames:[{polygon,occupancy_flat:map},{polygon,occupancy_flat:map}],fused:{polygon,occupancy_flat:map}}};
}
const instrumentation=`(()=>{const P=WebGLRenderingContext.prototype;const old={};const state={buffers:new WeakMap(),positionAttrib:-1,positionBuffer:null,colorLocation:null,offsetLocation:null,color:[0,0,0],offset:0,draws:[]};window.__gpuTrace=state;
 old.getAttribLocation=P.getAttribLocation;P.getAttribLocation=function(program,name){const result=old.getAttribLocation.call(this,program,name);if(name==='a_position')state.positionAttrib=result;return result;};
 old.getUniformLocation=P.getUniformLocation;P.getUniformLocation=function(program,name){const result=old.getUniformLocation.call(this,program,name);if(name==='u_color')state.colorLocation=result;if(name==='u_offset')state.offsetLocation=result;return result;};
 old.bindBuffer=P.bindBuffer;P.bindBuffer=function(target,buffer){if(target===this.ARRAY_BUFFER)this.__traceArrayBuffer=buffer;return old.bindBuffer.call(this,target,buffer);};
 old.bufferData=P.bufferData;P.bufferData=function(target,data,usage){if(target===this.ARRAY_BUFFER&&data instanceof Float32Array&&this.__traceArrayBuffer)state.buffers.set(this.__traceArrayBuffer,Array.from(data));return old.bufferData.call(this,target,data,usage);};
 old.vertexAttribPointer=P.vertexAttribPointer;P.vertexAttribPointer=function(index){if(index===state.positionAttrib)state.positionBuffer=this.__traceArrayBuffer;return old.vertexAttribPointer.apply(this,arguments);};
 old.uniform3fv=P.uniform3fv;P.uniform3fv=function(location,value){if(location===state.colorLocation)state.color=Array.from(value);return old.uniform3fv.call(this,location,value);};
 old.uniform1f=P.uniform1f;P.uniform1f=function(location,value){if(location===state.offsetLocation)state.offset=value;return old.uniform1f.call(this,location,value);};
 old.drawArrays=P.drawArrays;P.drawArrays=function(mode,first,count){if(mode===this.TRIANGLES){const data=state.buffers.get(state.positionBuffer)||[];state.draws.push({count,positions:data.slice(first*3,(first+count)*3),color:state.color.slice(),offset:state.offset});}return old.drawArrays.call(this,mode,first,count);};})();`;
async function main(){
 const directory=fs.mkdtempSync(path.join(os.tmpdir(),'scene-occupancy-flat-'));
 const file=path.join(directory,'fixture.html');fs.writeFileSync(file,template.replace('__SCENE_DATA__',JSON.stringify(fixture())));
 const target=await fetch(endpoint+'/json/new?'+encodeURIComponent(pathToFileURL(file).href),{method:'PUT'}).then(r=>{if(!r.ok)throw Error(`CDP new target failed: ${r.status}`);return r.json();});
 const ws=new WebSocket(target.webSocketDebuggerUrl);let seq=0;const pending=new Map(),errors=[];
 await new Promise((resolve,reject)=>{ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true});});
 ws.addEventListener('message',event=>{const m=JSON.parse(event.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);if(m.error)p.reject(Error(JSON.stringify(m.error)));else p.resolve(m.result);}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);});
 const send=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
 const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
 const state=()=>evaluate('window.__sceneViewer.getState()');
 const waitRender=async before=>evaluate(`new Promise(resolve=>{const end=performance.now()+5000;function tick(){const s=window.__sceneViewer.getState();if(s.renderCount>${before})resolve(s);else if(performance.now()>end)resolve(s);else requestAnimationFrame(tick)}tick()})`);
 const action=async expression=>{const before=(await state()).renderCount;await evaluate(`(()=>{${expression}})()`);const s=await waitRender(before);assert.ok(s.renderCount>before,'UI action must trigger a render');return s;};
 const check=async(id,value)=>action(`const e=document.getElementById(${JSON.stringify(id)});e.checked=${value};e.dispatchEvent(new Event('change',{bubbles:true}))`);
 const classCheck=async(id,value)=>action(`const e=document.querySelector('[data-class="${id}"]');e.checked=${value};e.dispatchEvent(new Event('change',{bubbles:true}))`);
 const flush=()=>evaluate('window.__sceneViewer.render()');
 try{
  await send('Runtime.enable');await send('Page.enable');await send('Page.addScriptToEvaluateOnNewDocument',{source:instrumentation});await send('Page.navigate',{url:pathToFileURL(file).href});
  await evaluate(`new Promise((resolve,reject)=>{const end=performance.now()+30000;function tick(){if(window.__sceneViewerReady||document.getElementById('error')&&!document.getElementById('error').hidden)resolve();else if(performance.now()>end)reject(Error('Viewer timeout'));else requestAnimationFrame(tick)}tick()})`);
  assert.equal(await evaluate("document.getElementById('error').hidden?'':document.getElementById('error').textContent"),'');
  assert.equal(await evaluate("document.getElementById('occupancyFlatControl').hidden"),false,'Flattened occupancy control is visible');
  assert.equal(await evaluate("document.getElementById('occupancy_flatLayer').checked"),false,'Flattened occupancy defaults off');
  let s=await state();assert.deepEqual(s.planeClassIds,[1,3]);assert.deepEqual(s.methods,['polygon']);assert.equal(s.triangleCount,0);assert.equal(s.availableOccupancy,true);
  s=await check('occupancy_flatLayer',true);assert.deepEqual(s.methods,['occupancy_flat','polygon']);assert.equal(s.triangleCount,4);assert.equal(s.surfaceDrawCalls,2);
  // Surface payloads are uploaded at z=0 and drawn with each saved semantic color.
  let trace=await evaluate('window.__gpuTrace.draws');let tri=trace.filter(d=>d.count===6);assert.equal(tri.length,2);
  const red=[235/255,40/255,30/255],green=[25/255,190/255,70/255];
  assert.ok(tri.some(d=>JSON.stringify(d.color)===JSON.stringify(red)));assert.ok(tri.some(d=>JSON.stringify(d.color)===JSON.stringify(green)));
  assert.ok(tri.every(d=>d.positions.length===18&&d.positions.filter((_,i)=>i%3===2).every(z=>z===0)),'Uploaded frame triangles must lie at Z=0');
  await action("document.getElementById('selectNone').click()");s=await classCheck(2,true);assert.equal(s.triangleCount,0,'Non-surface class 2 cannot contribute surface triangles');
  await action("document.getElementById('selectNone').click()");await classCheck(1,true);s=await state();assert.equal(s.triangleCount,2);await evaluate('window.__gpuTrace.draws.length=0');await flush();
  trace=await evaluate('window.__gpuTrace.draws');tri=trace.filter(d=>d.count===6);assert.ok(tri.length>=1&&tri.every(d=>JSON.stringify(d.color)===JSON.stringify(red)),'Class 1 draw uses its saved color');
  await classCheck(2,true);s=await check('pointLayer',false);assert.equal(s.pointCount,0);assert.equal(s.triangleCount,2,'Point visibility is independent of flattened surfaces');
  s=await check('pointLayer',true);assert.ok(s.pointCount>0);s=await check('occupancyLayer',true);assert.equal(s.occupancyCount,1);assert.equal(s.triangleCount,2,'Occupancy voxels are independent of flattened surfaces');
  s=await check('occupancyLayer',false);assert.equal(s.triangleCount,2);await classCheck(2,false);
  s=await action("document.getElementById('timeline').value='1';document.getElementById('timeline').dispatchEvent(new Event('input',{bubbles:true}))");assert.equal(s.frame,1);assert.equal(s.triangleCount,2);
  await action("document.getElementById('selectNone').click()");await classCheck(3,true);s=await state();assert.equal(s.triangleCount,2);await evaluate('window.__gpuTrace.draws.length=0');await flush();
  trace=await evaluate('window.__gpuTrace.draws');tri=trace.filter(d=>d.count===6);assert.ok(tri.length>=1&&tri.every(d=>JSON.stringify(d.color)===JSON.stringify(green)),'Frame 2 class 3 draw uses its saved color');
  s=await action("document.getElementById('bevMapView').click()");assert.equal(s.mode,'bev-map');
  s=await action("const e=document.getElementById('groundMapMethod');e.value='occupancy_flat';e.dispatchEvent(new Event('change',{bubbles:true}))");assert.equal(s.mapMethod,'occupancy_flat');assert.equal(s.mapCells,1);
  await classCheck(1,true);assert.equal((await state()).mapCells,2);await classCheck(3,false);assert.equal((await state()).mapCells,1);await classCheck(3,true);assert.equal((await state()).mapCells,2);
  await action("document.getElementById('selectNone').click()");await classCheck(3,true);await action("document.getElementById('scene3dView').click()");await evaluate('window.__gpuTrace.draws.length=0');
  await action('window.__sceneViewer.setAllFrames(true)');s=await state();assert.equal(s.allFrames,true);assert.equal(s.triangleCount,1,'Direct fused geometry contributes the class-3 triangle');assert.equal(s.surfaceDrawCalls,1,'Fused geometry is drawn once, not once per frame');
  trace=await evaluate('window.__gpuTrace.draws');const fused=trace.filter(d=>d.count===3&&JSON.stringify(d.color)===JSON.stringify(green));assert.equal(fused.length,1,'Authoritative fused class-3 mesh is uploaded and drawn exactly once');assert.deepEqual(fused[0].positions,[3,0,0,4,0,0,3,1,0]);
  assert.deepEqual(errors,[]);console.log(JSON.stringify({ok:true,fixture:'2 frames, surface classes 1/3, object class 2',singleFrameTriangles:2,fusedTriangles:s.triangleCount,fusedDrawCalls:s.surfaceDrawCalls,mapCells:2,checks:'control default, uploaded z=0 triangles and class colors, class filtering, point/occupancy independence, timeline, direct fused mesh once, BEV class filters'}));
 }finally{await send('Page.close').catch(()=>{});ws.close();const intended=path.resolve(directory);assert.equal(path.dirname(intended),path.resolve(os.tmpdir()));assert.ok(path.basename(intended).startsWith('scene-occupancy-flat-'));fs.rmSync(intended,{recursive:true,force:true});}
}
main().catch(error=>{console.error(error.stack||String(error));process.exitCode=1;});

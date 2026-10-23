'use strict';
// node tasks/nuscenes_sam3/tests/occupancy_browser.cjs [CDP endpoint]
const assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path');
const {pathToFileURL}=require('node:url');
const endpoint=process.argv[2]||'http://127.0.0.1:9341';
const template=fs.readFileSync(path.resolve(__dirname,'../web/scene_viewer.html'),'utf8');
const packed=(values,type)=>Buffer.from(new type(values).buffer).toString('base64');
const occupancy={resolution:1,origin:[0,0,0],frames:[{indices:[[-1,0,0],[0,0,0]],point_counts:[2,3],class_ids:[1,2],voxel_count:2,point_count:5},{indices:[[0,0,1]],point_counts:[4],class_ids:[2],voxel_count:1,point_count:4}],fused:{indices:[[-1,0,0],[0,0,0],[0,0,1]],point_counts:[2,3,4],class_ids:[1,2,2],voxel_count:3,point_count:9}};
// Legacy unassigned records omit class_ids; semantic and unassigned voxels may overlap.
const unassignedOccupancy={resolution:1,origin:[0,0,0],frames:[{indices:[[-1,0,0],[1,0,0]],point_counts:[4,1],voxel_count:2,point_count:5},{indices:[[0,0,1]],point_counts:[3],voxel_count:1,point_count:3}],fused:{indices:[[-1,0,0],[1,0,0],[0,0,1]],point_counts:[4,1,3],voxel_count:3,point_count:8}};
const scene={scene:'occupancy-browser',classes:[{id:0,name:'unknown',color:[255,255,255]},{id:1,name:'road',color:[60,160,230]},{id:2,name:'object',color:[230,120,50]}],plane_class_ids:[1],bounds:{min:[-3,-3,-2],max:[3,3,3]},frames:Array.from({length:2},(_,i)=>({sample_token:`frame-${i}`,ego:[0,0,0],points:{xyz:packed([i,0,0],Float32Array),labels:packed([i===0?0:2],Uint8Array),count:1},surfaces:{polygon:{1:{xyz:'',count:0}}}})),occupancy_grid:occupancy};
function probe(fallback){
 const original=WebGLRenderingContext.prototype.getExtension,uploads=[],draws=[];
 const data=WebGLRenderingContext.prototype.bufferData,draw=WebGLRenderingContext.prototype.drawArrays,uniform=WebGLRenderingContext.prototype.uniform3fv;
 let currentColor=[];
 WebGLRenderingContext.prototype.bufferData=function(target,values,usage){if(values instanceof Float32Array)uploads.push(Array.from(values));return data.call(this,target,values,usage);};
 WebGLRenderingContext.prototype.uniform3fv=function(location,value){currentColor=Array.from(value);return uniform.call(this,location,value);};
 WebGLRenderingContext.prototype.drawArrays=function(mode,first,count){draws.push({mode,count,color:currentColor.slice()});return draw.call(this,mode,first,count);};
 WebGLRenderingContext.prototype.getExtension=function(name){if(name!=='ANGLE_instanced_arrays')return original.call(this,name);if(fallback)return null;const ext=original.call(this,name);if(!ext)return null;return{vertexAttribDivisorANGLE:ext.vertexAttribDivisorANGLE.bind(ext),drawArraysInstancedANGLE:(mode,first,count,instances)=>{draws.push({mode,count,instances,color:currentColor.slice()});return ext.drawArraysInstancedANGLE(mode,first,count,instances);}};};
 window.__occProbe={uploads,draws};
}
async function run(file,fallback,hasUnassigned){
 let target,ws,seq=0;const pending=new Map(),errors=[];
 const send=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq,timer=setTimeout(()=>{pending.delete(id);reject(Error('Timeout '+method));},20000);pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}));});
 const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
 try{
  target=await fetch(endpoint+'/json/new?about:blank',{method:'PUT'}).then(r=>r.json());ws=new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((r,j)=>{ws.addEventListener('open',r,{once:true});ws.addEventListener('error',j,{once:true});});
  ws.addEventListener('message',event=>{const m=JSON.parse(event.data);if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);if(m.id){const p=pending.get(m.id);if(p){clearTimeout(p.timer);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}}});
  await send('Runtime.enable');await send('Page.enable');await send('Page.addScriptToEvaluateOnNewDocument',{source:`(${probe.toString()})(${fallback})`});
  await send('Page.navigate',{url:pathToFileURL(file).href});
  await evaluate(`new Promise((resolve,reject)=>{const t=performance.now()+15000;(function poll(){if(window.__sceneViewerReady)resolve();else if(performance.now()>t)reject(Error(document.getElementById('error')?.textContent||'Viewer timeout'));else setTimeout(poll,20)})()})`);
  const state=()=>evaluate('window.__sceneViewer.getState()');
  const action=expression=>evaluate(`(()=>{${expression};window.__sceneViewer.render()})()`);
  const change=(id,value)=>action(`const e=document.getElementById(${JSON.stringify(id)});e.checked=${value};e.dispatchEvent(new Event('change',{bubbles:true}))`);
  const changeClass=(id,value)=>action(`const e=document.querySelector('[data-class="${id}"]');e.checked=${value};e.dispatchEvent(new Event('change',{bubbles:true}))`);
  const voxelDraws=()=>evaluate('window.__occProbe.draws.filter(d=>d.mode===4)');
  const road=[60/255,160/255,230/255],object=[230/255,120/255,50/255],gray=[.55,.60,.65];
  const assertColors=(actual,expected)=>{assert.equal(actual.length,expected.length);for(let i=0;i<actual.length;i++)for(let j=0;j<3;j++)assert.ok(Math.abs(actual[i][j]-expected[i][j])<1e-6,`draw ${i} color channel ${j}: ${actual[i][j]}`);};
  const assertBatches=(actual,counts)=>{if(fallback)assert.deepEqual(actual.map(d=>d.count),counts.map(n=>36*n));else assert.deepEqual(actual.map(d=>[d.count,d.instances]),counts.map(n=>[36,n]));};
  const assertCounts=async(semanticVoxels,semanticPoints,unassignedVoxels,unassignedPoints)=>{const s=await state();assert.deepEqual([s.occupancyCount,s.occupancyPointCount,s.unassignedOccupancyCount,s.unassignedOccupancyPointCount],[semanticVoxels,semanticPoints,unassignedVoxels,unassignedPoints]);return s;};
  const assertGray=async count=>{const draws=(await voxelDraws()).slice(-1);assertColors(draws.map(d=>d.color),[gray]);assertBatches(draws,[count]);};
  let s=await state();assert.equal(s.occupancy,false);assert.equal(s.unassignedOccupancy,false);assert.equal(s.availableUnassignedOccupancy,hasUnassigned);assert.equal(s.occupancyInstanced,!fallback);
  await assertCounts(0,0,0,0);
  assert.equal(await evaluate('document.getElementById("occupancyLayer").disabled'),false);
  assert.deepEqual(await evaluate('(()=>{const e=document.getElementById("unassignedOccupancyLayer");return [e.disabled,e.checked]})()'),[!hasUnassigned,false]);
  await action('window.__occProbe.draws.length=0;document.getElementById("occupancyLayer").click()');
  s=await state();assert.equal(s.occupancyCount,2);assert.equal(s.occupancyPointCount,5);
  const first=(await voxelDraws()).slice(-2);assert.equal(first.length,2);
  assertColors(first.map(d=>d.color),[road,object]);assertBatches(first,[1,1]);
  const uploads=await evaluate('window.__occProbe.uploads');
  if(fallback){const vertices=uploads.filter(a=>a.length===108).slice(-2);assert.equal(vertices.length,2);for(let axis=0;axis<3;axis++){const roadValues=vertices[0].filter((_,i)=>i%3===axis),objectValues=vertices[1].filter((_,i)=>i%3===axis);assert.equal(Math.min(...roadValues),axis===0?-1:0);assert.equal(Math.max(...roadValues),axis===0?0:1);assert.equal(Math.min(...objectValues),0);assert.equal(Math.max(...objectValues),1);}}
  else{assert.ok(uploads.some(a=>JSON.stringify(a)===JSON.stringify([-.5,.5,.5])),'road center includes negative XYZ voxel');assert.ok(uploads.some(a=>JSON.stringify(a)===JSON.stringify([.5,.5,.5])),'object center includes positive XYZ voxel');}
  await changeClass(2,false);s=await state();assert.equal(s.occupancyCount,1);assert.equal(s.occupancyPointCount,2);assertColors((await voxelDraws()).slice(-1).map(d=>d.color),[road]);
  await action('document.getElementById("selectNone").click()');s=await state();assert.equal(s.occupancyCount,0);assert.equal(s.occupancyPointCount,0);
  await change('pointLayer',false);assert.equal((await state()).occupancyCount,0);
  await action('document.getElementById("selectAll").click()');s=await state();assert.equal(s.occupancyCount,2);assert.equal(s.occupancyPointCount,5);assert.equal(s.pointCount,0);
  await change('pointLayer',true);assert.equal((await state()).pointCount,1);assert.equal((await state()).occupancyCount,2);
  await action('window.__sceneViewer.setFrame(1)');s=await state();assert.equal(s.occupancyCount,1);assert.equal(s.occupancyPointCount,4);
  await action('window.__sceneViewer.setAllFrames(true)');s=await state();assert.equal(s.occupancyCount,3);assert.equal(s.occupancyPointCount,9);
  const fusedDraws=(await voxelDraws()).slice(-2);assertColors(fusedDraws.map(d=>d.color),[road,object]);assertBatches(fusedDraws,[1,2]);
  await change('pointLayer',false);s=await state();assert.equal(s.occupancyCount,3);assert.equal(s.occupancyPointCount,9);assert.equal(s.pointCount,0);
  await action('document.getElementById("selectNone").click()');s=await state();assert.equal(s.occupancyCount,0);assert.equal(s.pointCount,0);
  await action('document.getElementById("selectAll").click()');s=await state();assert.equal(s.pointCount,0);assert.equal(s.occupancyCount,3);
  await change('pointLayer',true);assert.equal((await state()).pointCount,2);assert.equal((await state()).occupancyCount,3);
  await change('trajectoryLayer',false);await action('window.__sceneViewer.setFrame(0)');assert.equal((await state()).occupancyCount,3);
  await change('occupancyLayer',false);assert.equal((await state()).occupancyCount,0);assert.equal((await state()).occupancyPointCount,0);
  await change('occupancyLayer',true);await action('window.__sceneViewer.setAllFrames(false)');assert.equal((await state()).occupancyCount,2);
  if(hasUnassigned){
   await change('unassignedOccupancyLayer',true);s=await assertCounts(2,5,2,5);assert.equal(s.unassignedOccupancy,true);
   const both=(await voxelDraws()).slice(-3);assertColors(both.map(d=>d.color),[road,object,gray]);assertBatches(both,[1,1,2]);
   await changeClass(0,false);s=await assertCounts(2,5,2,5);assert.ok(!s.classIds.includes(0));assert.equal(s.pointCount,0);await assertGray(2);
   await changeClass(0,true);s=await assertCounts(2,5,2,5);assert.ok(s.classIds.includes(0));assert.equal(s.pointCount,1);
   await action('document.getElementById("selectNone").click()');s=await assertCounts(0,0,2,5);assert.deepEqual(s.classIds,[]);assert.equal(s.pointCount,0);await assertGray(2);
   await changeClass(0,true);s=await assertCounts(0,0,2,5);assert.deepEqual(s.classIds,[0]);assert.equal(s.pointCount,1);await assertGray(2);
   await changeClass(0,false);await assertCounts(0,0,2,5);
   await action('document.getElementById("selectAll").click()');await assertCounts(2,5,2,5);
   await change('pointLayer',false);s=await assertCounts(2,5,2,5);assert.equal(s.pointCount,0);await assertGray(2);
   await change('occupancyLayer',false);s=await assertCounts(0,0,2,5);assert.equal(s.occupancy,false);await assertGray(2);
   await change('unassignedOccupancyLayer',false);s=await assertCounts(0,0,0,0);assert.equal(s.unassignedOccupancy,false);
   await action('window.__occProbe.draws.length=0');assert.deepEqual(await voxelDraws(),[]);
   await change('pointLayer',true);s=await assertCounts(0,0,0,0);assert.equal(s.pointCount,1);
   await change('unassignedOccupancyLayer',true);await assertCounts(0,0,2,5);await assertGray(2);
   await change('occupancyLayer',true);await assertCounts(2,5,2,5);
   await change('unassignedOccupancyLayer',false);await assertCounts(2,5,0,0);
   await action('window.__occProbe.draws.length=0');const semanticOnly=await voxelDraws();assertColors(semanticOnly.map(d=>d.color),[road,object]);assertBatches(semanticOnly,[1,1]);
   await change('unassignedOccupancyLayer',true);
   await action('window.__sceneViewer.setFrame(1)');await assertCounts(1,4,1,3);
   const second=(await voxelDraws()).slice(-2);assertColors(second.map(d=>d.color),[object,gray]);assertBatches(second,[1,1]);
   await action('window.__sceneViewer.setAllFrames(true)');await assertCounts(3,9,3,8);
   const all=(await voxelDraws()).slice(-3);assertColors(all.map(d=>d.color),[road,object,gray]);assertBatches(all,[1,2,3]);
   await action('window.__sceneViewer.setFrame(0)');await assertCounts(3,9,3,8);await assertGray(3);
   await change('occupancyLayer',false);await assertCounts(0,0,3,8);await assertGray(3);
   await action('document.getElementById("selectNone").click()');await assertCounts(0,0,3,8);await assertGray(3);
   await change('pointLayer',false);s=await assertCounts(0,0,3,8);assert.equal(s.pointCount,0);
   await change('unassignedOccupancyLayer',false);await assertCounts(0,0,0,0);
   await action('window.__sceneViewer.setAllFrames(false)');await assertCounts(0,0,0,0);
   await action('document.getElementById("selectAll").click()');await change('occupancyLayer',true);await assertCounts(2,5,0,0);
   await change('unassignedOccupancyLayer',true);await assertCounts(2,5,2,5);await assertGray(2);
   await change('unassignedOccupancyLayer',false);await assertCounts(2,5,0,0);
  }else{
   await action('document.getElementById("unassignedOccupancyLayer").click()');s=await assertCounts(2,5,0,0);assert.equal(s.unassignedOccupancy,false);
   assert.equal(await evaluate('document.getElementById("unassignedOccupancyLayer").disabled'),true);
  }
  assert.deepEqual(errors,[]);console.log(JSON.stringify({ok:true,fallback,hasUnassigned,checks:'XYZ cubes, negative indices, semantic colors/counts, legacy unassigned gray batches, class-0/selectNone independence, separate layer toggles, frame/fused counts and off resets'}));
 }finally{if(ws&&ws.readyState===WebSocket.OPEN)await send('Page.close').catch(()=>{});if(ws)ws.close();for(const p of pending.values())clearTimeout(p.timer);}
}
async function main(){const dir=fs.mkdtempSync(path.join(os.tmpdir(),'occupancy-browser-'));try{for(const hasUnassigned of [true,false]){const file=path.join(dir,hasUnassigned?'unassigned.html':'legacy.html'),data=hasUnassigned?{...scene,unassigned_occupancy_grid:unassignedOccupancy}:scene;fs.writeFileSync(file,template.replace('__SCENE_DATA__',JSON.stringify(data)));await run(file,false,hasUnassigned);await run(file,true,hasUnassigned);}}finally{const resolved=path.resolve(dir);assert.equal(path.dirname(resolved),path.resolve(os.tmpdir()));assert.ok(path.basename(resolved).startsWith('occupancy-browser-'));fs.rmSync(resolved,{recursive:true,force:true});}}
main().catch(e=>{console.error(e.stack);process.exitCode=1;});

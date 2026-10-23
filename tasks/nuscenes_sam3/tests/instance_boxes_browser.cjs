'use strict';
// Native Node CDP test; reuse existing Chromium, without installs or browser startup.
// node instance_boxes_browser.cjs scene.html [boxes.png] [9339 | http://127.0.0.1:9339]
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
async function main(){
 if(!process.argv[2])throw Error('Usage: node instance_boxes_browser.cjs scene.html [boxes.png] [CDP port/endpoint]');
 const html=path.resolve(process.argv[2]),screenshot=path.resolve(process.argv[3]||html.replace(/\.html?$/i,'')+'.boxes.png');
 const arg=process.argv[4]||'9339',endpoint=(/^\d+$/.test(arg)?'http://127.0.0.1:'+arg:arg).replace(/\/$/,'');
 assert.ok(fs.statSync(html).isFile());
 const target=await fetch(endpoint+'/json/new?about:blank',{method:'PUT'}).then(r=>{if(!r.ok)throw Error('CDP target: '+r.status);return r.json();});
 const ws=new WebSocket(target.webSocketDebuggerUrl),pending=new Map(),errors=[],requests=[];let seq=0;
 await new Promise((resolve,reject)=>{ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true});});
 ws.addEventListener('message',event=>{const m=JSON.parse(event.data);if(m.id){const p=pending.get(m.id);if(!p)return;pending.delete(m.id);clearTimeout(p.timer);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}
  else if(m.method==='Runtime.exceptionThrown')errors.push(JSON.stringify(m.params.exceptionDetails));
  else if(m.method==='Runtime.consoleAPICalled'&&m.params.type==='error')errors.push(m.params.args.map(v=>v.description||v.value||'').join(' '));
  else if(m.method==='Network.requestWillBeSent'&&/^https?:/i.test(m.params.request.url))requests.push(m.params.request.url);});
 const send=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq,timer=setTimeout(()=>{pending.delete(id);reject(Error('CDP timed out: '+method));},65000);pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}));});
 const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
 const wait=expression=>evaluate(`new Promise((resolve,reject)=>{const end=performance.now()+60000;function tick(){if(${expression})resolve(true);else if(performance.now()>end)reject(Error('Viewer wait timed out'));else requestAnimationFrame(tick)}tick()})`);
 const state=()=>evaluate('window.__sceneViewer.getState()');
 const changed=async expression=>{const n=(await state()).renderCount;await evaluate(expression);await wait(`window.__sceneViewer.getState().renderCount>${n}`);};
 const click=id=>changed(`document.getElementById(${JSON.stringify(id)}).click()`);
 const check=(id,value)=>changed(`(()=>{const e=document.getElementById(${JSON.stringify(id)});e.checked=${value};e.dispatchEvent(new Event('change',{bubbles:true}));})()`);
 const classCheck=(id,value)=>changed(`(()=>{const e=document.querySelector('[data-class="${id}"]');e.checked=${value};e.dispatchEvent(new Event('change',{bubbles:true}));})()`);
 const expected=()=>evaluate(`(()=>{const v=window.__sceneViewer,s=v.getState();let count=0,drawCalls=0;const lineVertices=[];for(const f of s.allFrames?v.data.frames:[v.data.frames[s.frame]]){const groups=new Map();for(const b of f.bounding_boxes||[]){if(!s.boxes||s.planeClassIds.includes(b.class_id)||!s.classIds.includes(b.class_id)||!b.corners_scene.length)continue;count++;const g=groups.get(b.class_id)||{vertices:0,points:false};if(b.corners_scene.length===1)g.points=true;else g.vertices+=b.geometry_type==='cuboid'?24:b.geometry_type==='rectangle'?8:2;groups.set(b.class_id,g);}for(const g of groups.values()){drawCalls+=Number(g.vertices>0)+Number(g.points);if(g.vertices)lineVertices.push(g.vertices);}}return{count,drawCalls,lineVertices};})()`);
 const assertBoxes=async()=>{
  const s=await state(),e=await expected();
  assert.equal(s.boxCount,e.count,'Visible boxes must match non-surface frame/class payload selection');
  assert.equal(s.boxDrawCalls,e.drawCalls,'One line/point draw per visible frame/class batch');
  if(s.mode==='3d'){
   const vertices=await evaluate(`(()=>{const v=window.__sceneViewer,gl=document.getElementById('viewer').getContext('webgl'),draw=gl.drawArrays,vertices=[];gl.drawArrays=function(mode,first,count){if(mode===gl.LINES)vertices.push(count);return draw.call(this,mode,first,count);};try{v.render();}finally{gl.drawArrays=draw;}return vertices.slice(1);})()`);
   assert.deepEqual(vertices,e.lineVertices,'Each cuboid contributes exactly 12 edges / 24 vertices to its class batch');
  }
  return e;
 };
 try{
  await send('Runtime.enable');await send('Network.enable');await send('Page.enable');
  await send('Emulation.setDeviceMetricsOverride',{width:1500,height:1000,deviceScaleFactor:1,mobile:false});
  await send('Page.navigate',{url:pathToFileURL(html).href});
  await wait('window.__sceneViewerReady===true || (document.getElementById("error") && !document.getElementById("error").hidden)');
  assert.equal(await evaluate('document.getElementById("error").hidden?"":document.getElementById("error").textContent'),'');
  const payload=await evaluate(`(()=>{const v=window.__sceneViewer,s=v.getState();return{frames:v.data.frames.map(f=>(f.bounding_boxes||[])),classes:s.availableClassIds,planeClasses:s.planeClassIds};})()`);
  assert.ok(payload.frames.length>=1,'Supports scenes with one or more frames');
  let available=0,positiveHeight=0;
  for(const records of payload.frames)for(const b of records){
   assert.ok(payload.classes.includes(b.class_id),'Box class ID must exist in saved classes');
   const sizes={empty:0,point:1,segment:2,rectangle:4,cuboid:8};assert.ok(Object.hasOwn(sizes,b.geometry_type));
   assert.ok(Array.isArray(b.corners_scene));assert.equal(b.corners_scene.length,sizes[b.geometry_type]);
   for(const row of b.corners_scene){assert.equal(row.length,3);assert.ok(row.every(Number.isFinite));}
   if(b.geometry_type==='cuboid'){
    const c=b.corners_scene,zmin=Math.min(...c.map(p=>p[2])),zmax=Math.max(...c.map(p=>p[2]));
    for(let i=0;i<4;i++){assert.equal(c[i][2],zmin,'Bottom corners must use Z minimum');assert.equal(c[i+4][2],zmax,'Top corners must use Z maximum');assert.deepEqual(c[i].slice(0,2),c[i+4].slice(0,2),'Top/bottom XY must correspond');}
    const area2=c.slice(0,4).reduce((sum,p,i)=>sum+p[0]*c[(i+1)%4][1]-p[1]*c[(i+1)%4][0],0);
    assert.ok(area2>=-1e-7,'Bottom footprint must be counterclockwise, allowing degenerate boxes');
    if(b.footprint_corners_scene!==undefined)assert.deepEqual(b.footprint_corners_scene,c.slice(0,4),'Optional BEV footprint must match bottom corners');
    if(!payload.planeClasses.includes(b.class_id)&&zmax>zmin)positiveHeight++;
   }
   if(!payload.planeClasses.includes(b.class_id)&&b.corners_scene.length)available++;
  }
  assert.equal(await evaluate('document.querySelectorAll("#ipmLayer, #mapMethod").length'),0,'Removed IPM and map method controls must be absent');
  const initial=await state();assert.deepEqual(initial.methods,['polygon']);assert.equal(initial.boxes,false);assert.equal(initial.boxCount,0);assert.equal(initial.boxDrawCalls,0);assert.equal(initial.availableBoxCount,available);
  assert.ok(available>0,'Real scene must include at least one drawable non-surface exported instance box');
  assert.ok(positiveHeight>0,'Real fixture must contain at least one non-surface cuboid with genuine positive height');
  assert.equal(await evaluate('document.getElementById("bboxLayer").disabled'),false);
  await assertBoxes();await check('bboxLayer',true);await assertBoxes();
  const first=await state();await check('pointLayer',false);assert.equal((await state()).boxCount,first.boxCount);assert.equal((await state()).pointCount,0);await assertBoxes();
  await check('groundPointLayer',true);assert.equal((await state()).boxCount,first.boxCount);await assertBoxes();
  await check('polygonLayer',false);assert.deepEqual((await state()).methods,[]);assert.equal((await state()).boxCount,first.boxCount);await assertBoxes();
  await click('selectNone');assert.equal((await state()).boxCount,0);assert.equal((await state()).boxDrawCalls,0);await assertBoxes();
  await click('selectAll');assert.deepEqual((await state()).classIds,payload.classes.slice().sort((a,b)=>a-b));await assertBoxes();
  // Test every saved class independently, including unassigned and all configured planes.
  await click('selectNone');
  for(const id of payload.classes){await classCheck(id,true);await assertBoxes();if(payload.planeClasses.includes(id)){assert.equal((await state()).boxCount,0,'Selected surface class must never display boxes');assert.equal((await state()).boxDrawCalls,0);}await classCheck(id,false);}
  await click('selectAll');
  if(payload.frames.length>=2){await changed('window.__sceneViewer.setFrame(1)');assert.equal((await state()).frame,1);await assertBoxes();}
  await check('allFrames',true);const all=await assertBoxes();assert.equal(all.count,available,'All-frame boxes retain each non-surface camera-local record');
  assert.match(await evaluate('document.getElementById("bboxNote").textContent'),/not tracks/);
  const hasMap=await evaluate('!document.getElementById("bevMapView").disabled');
  if(hasMap){
   await click('bevMapView');assert.equal((await state()).mode,'bev-map');assert.equal((await state()).boxCount,available);await assertBoxes();
   await click('selectNone');assert.equal((await state()).boxCount,0);await assertBoxes();await click('selectAll');await assertBoxes();
   await check('allFrames',false);await changed('window.__sceneViewer.setFrame(0)');await assertBoxes();
   if(payload.frames.length>=2){await changed('window.__sceneViewer.setFrame(1)');await assertBoxes();}
   const before=await state();await check('pointLayer',true);await check('groundPointLayer',false);assert.equal((await state()).boxCount,before.boxCount);await assertBoxes();
   await check('bboxLayer',false);assert.equal((await state()).boxCount,0);assert.equal((await state()).boxDrawCalls,0);await assertBoxes();
   await check('bboxLayer',true);await click('scene3dView');assert.equal((await state()).boxCount,before.boxCount);await assertBoxes();
  }
  await check('allFrames',false);await changed('window.__sceneViewer.setFrame(0)');await click('selectAll');
  await check('bboxLayer',false);assert.equal((await state()).boxCount,0);assert.equal((await state()).boxDrawCalls,0);await assertBoxes();
  await check('bboxLayer',true);await check('pointLayer',true);await check('groundPointLayer',false);await check('polygonLayer',true);assert.deepEqual((await state()).methods,['polygon']);
  await classCheck(0,false);await click('fitView');await click('orbitView');const final=await assertBoxes();
  assert.ok((await state()).camera.pitch<Math.PI/2-.01,'Screenshot must show 3D height');
  assert.deepEqual(requests,[],'Offline scene must make no HTTP(S) requests');assert.deepEqual(errors,[],'No browser exceptions or console errors');
  const image=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.mkdirSync(path.dirname(screenshot),{recursive:true});fs.writeFileSync(screenshot,Buffer.from(image.data,'base64'));
  console.log(JSON.stringify({ok:true,scene:await evaluate('window.__sceneViewer.data.scene'),frames:payload.frames.length,availableBoxCount:available,positiveHeightCuboids:positiveHeight,firstFrameBoxes:final.count,mapChecked:hasMap,screenshot,finalState:await state(),checks:'removed controls absent, polygon-only, default off, 8-corner cuboid bottom/top contract, positive height, 12 edges per cuboid in class batches, selected frame/classes, all classes including planes and unassigned, surface boxes excluded, independent points/ground/surfaces, all frames camera-local count, BEV/3D parity, off resets counts, orbit screenshot excluding unassigned, no HTTP/errors'},null,2));
 }finally{await send('Page.close').catch(()=>{});ws.close();for(const p of pending.values())clearTimeout(p.timer);}
}
main().catch(error=>{console.error(error.stack||String(error));process.exitCode=1;});

'use strict';
// Native Node CDP verification using an already running Chromium, no npm packages.
// node scene_viewer_cdp.cjs viewer.html screenshot.png [http://127.0.0.1:9339]
const fs=require('node:fs');const path=require('node:path');const assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
async function main(){
 const html=path.resolve(process.argv[2]);const screenshot=path.resolve(process.argv[3]);const endpoint=process.argv[4]||'http://127.0.0.1:9339';
 const target=await fetch(endpoint+'/json/new?'+encodeURIComponent(pathToFileURL(html).href),{method:'PUT'}).then(r=>r.json());
 const ws=new WebSocket(target.webSocketDebuggerUrl);let seq=0;const pending=new Map();const errors=[],requests=[];
 await new Promise((resolve,reject)=>{ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true});});
 ws.addEventListener('message',event=>{const m=JSON.parse(event.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);if(m.error)p.reject(new Error(JSON.stringify(m.error)));else p.resolve(m.result);}
   else if(m.method==='Runtime.exceptionThrown')errors.push(JSON.stringify(m.params.exceptionDetails));
   else if(m.method==='Network.requestWillBeSent' && /^https?:/.test(m.params.request.url))requests.push(m.params.request.url);});
 const send=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
 const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
 const wait=async expression=>evaluate(`new Promise((resolve,reject)=>{const end=performance.now()+60000;function tick(){if(${expression})resolve(true);else if(performance.now()>end)reject(new Error('Viewer wait timed out: '+${JSON.stringify(expression)}));else requestAnimationFrame(tick)}tick()})`);
 const state=()=>evaluate('window.__sceneViewer.getState()');
 const changed=async expression=>{const old=(await state()).renderCount;await evaluate(expression);await wait(`window.__sceneViewer.getState().renderCount>${old}`);};
 const check=(id,value)=>changed(`(()=>{const e=document.getElementById(${JSON.stringify(id)});e.checked=${value};e.dispatchEvent(new Event('change',{bubbles:true}));})()`);
 try{
  await send('Runtime.enable');await send('Network.enable');await send('Page.enable');
  await send('Emulation.setDeviceMetricsOverride',{width:1500,height:1000,deviceScaleFactor:1,mobile:false});
  await send('Page.navigate',{url:pathToFileURL(html).href});
  await wait('window.__sceneViewerReady===true || (document.getElementById("error") && !document.getElementById("error").hidden)');
  const problem=await evaluate('document.getElementById("error").hidden?"":document.getElementById("error").textContent');assert.equal(problem,'');
  assert.equal(await evaluate('document.querySelectorAll("#ipmLayer, #mapMethod").length'),0,'Removed IPM and map method controls must be absent');
  let initial=await state();assert.ok(initial.drawCalls>0 && initial.pointCount>0 && initial.triangleCount>0);assert.deepEqual(initial.methods,['polygon']);
  const count=await evaluate('window.__sceneViewer.data.frames.length');assert.ok(count>=2);
  await changed('(()=>{const e=document.getElementById("timeline");e.value="1";e.dispatchEvent(new Event("input",{bubbles:true}));})()');assert.equal((await state()).frame,1);
  await changed('document.getElementById("prevFrame").click()');assert.equal((await state()).frame,0);
  await check('allFrames',true);let all=await state();assert.ok(all.pointCount>initial.pointCount);assert.ok(all.triangleCount>=initial.triangleCount);
  assert.deepEqual((await state()).methods,['polygon']);
  const expected=await evaluate('(()=>{const v=window.__sceneViewer,s=v.getState();return v.data.frames.reduce((n,f)=>n+s.planeClassIds.reduce((z,c)=>z+(s.classIds.includes(c)?(f.surfaces.polygon[c]?.count||0)/3:0),0),0)})()');assert.equal((await state()).triangleCount,expected);
  await check('polygonLayer',false);assert.deepEqual((await state()).methods,[]);assert.equal((await state()).triangleCount,0);assert.equal((await state()).surfaceDrawCalls,0);
  await check('polygonLayer',true);assert.deepEqual((await state()).methods,['polygon']);assert.equal((await state()).triangleCount,expected);
  await changed('document.getElementById("selectNone").click()');assert.equal((await state()).pointCount,0);assert.equal((await state()).triangleCount,0);
  for(const cid of initial.availableClassIds){
   await changed(`(()=>{const e=document.querySelector('[data-class="${cid}"]');e.checked=true;e.dispatchEvent(new Event('change',{bubbles:true}));})()`);
   assert.deepEqual((await state()).classIds,[cid]);
   if(!initial.planeClassIds.includes(cid))assert.equal((await state()).triangleCount,0);
   await changed(`(()=>{const e=document.querySelector('[data-class="${cid}"]');e.checked=false;e.dispatchEvent(new Event('change',{bubbles:true}));})()`);
  }
  await changed('document.getElementById("selectAll").click()');await check('groundPointLayer',true);
  const total=await evaluate('window.__sceneViewer.data.frames.reduce((n,f)=>n+f.points.count,0)');assert.equal((await state()).pointCount,total);
  await check('pointLayer',false);assert.equal((await state()).pointCount,0);await check('pointLayer',true);
  await check('allFrames',false);
  const previous=(await state()).frame;await evaluate('document.getElementById("playPause").click()');await wait(`window.__sceneViewer.getState().frame!==${previous}`);await evaluate('document.getElementById("playPause").click()');
  await changed('document.getElementById("topView").click()');assert.ok((await state()).camera.pitch>1.4);
  await changed('document.getElementById("orbitView").click()');
  await check('allFrames',true);await check('groundPointLayer',false);
  await changed('(()=>{const e=document.querySelector(\'[data-class="0"]\');e.checked=false;e.dispatchEvent(new Event("change",{bubbles:true}));})()');
  await changed('document.getElementById("fitView").click()');
  const pixels=await evaluate('(()=>{const canvas=document.getElementById("viewer"),gl=canvas.getContext("webgl"),a=new Uint8Array(canvas.width*canvas.height*4);gl.readPixels(0,0,canvas.width,canvas.height,gl.RGBA,gl.UNSIGNED_BYTE,a);let colored=0;for(let i=0;i<a.length;i+=4)if(Math.max(a[i],a[i+1],a[i+2])-Math.min(a[i],a[i+1],a[i+2])>45)colored++;return colored})()');
  assert.ok(pixels>100,'Canvas must contain colored surfaces/points');
  const image=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(screenshot,Buffer.from(image.data,'base64'));
  assert.equal(requests.length,0,'Offline viewer must perform no network requests');assert.deepEqual(errors,[]);
  console.log(JSON.stringify({scene:await evaluate('window.__sceneViewer.data.scene'),frames:count,totalOriginalPoints:total,visibleColoredPixels:pixels,finalState:await state(),screenshot,checks:'WebGL, timeline, play, all frames, polygon visibility/counts, removed controls absent, class filters, point visibility, camera views, zero HTTP requests'},null,2));
 }finally{await send('Page.close').catch(()=>{});ws.close();}
}
main().catch(error=>{console.error(error.stack);process.exitCode=1;});

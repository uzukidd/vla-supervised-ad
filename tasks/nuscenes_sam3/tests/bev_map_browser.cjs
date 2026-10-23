'use strict';
// Native Node CDP contract test; reuse Chromium on port 9339, with no npm dependencies.
// node bev_map_browser.cjs scene.html [map.png] [http://127.0.0.1:9339]
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
async function main(){
 if(!process.argv[2])throw new Error('Usage: node bev_map_browser.cjs scene.html [map.png] [CDP endpoint]');
 const html=path.resolve(process.argv[2]),screenshot=path.resolve(process.argv[3]||html.replace(/\.html?$/i,'')+'.bev.png');
 const endpoint=(process.argv[4]||'http://127.0.0.1:9339').replace(/\/$/,'');assert.ok(fs.statSync(html).isFile());
 const target=await fetch(endpoint+'/json/new?about:blank',{method:'PUT'}).then(r=>{if(!r.ok)throw Error('CDP target: '+r.status);return r.json();});
 const ws=new WebSocket(target.webSocketDebuggerUrl),pending=new Map(),errors=[],requests=[];let seq=0;
 await new Promise((resolve,reject)=>{ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true});});
 ws.addEventListener('message',event=>{const m=JSON.parse(event.data);if(m.id){const p=pending.get(m.id);if(!p)return;pending.delete(m.id);clearTimeout(p.timer);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}
  else if(m.method==='Runtime.exceptionThrown')errors.push(JSON.stringify(m.params.exceptionDetails));
  else if(m.method==='Runtime.consoleAPICalled'&&m.params.type==='error')errors.push(m.params.args.map(v=>v.description||v.value||'').join(' '));
  else if(m.method==='Network.requestWillBeSent'&&/^https?:/i.test(m.params.request.url))requests.push(m.params.request.url);});
 const send=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq,timer=setTimeout(()=>{pending.delete(id);reject(Error('CDP timed out: '+method));},65000);pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}));});
 const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
 const wait=expression=>evaluate(`new Promise((resolve,reject)=>{const end=performance.now()+60000;function tick(){if(${expression})resolve(true);else if(performance.now()>end)reject(Error('Viewer wait timed out: '+${JSON.stringify(expression)}));else requestAnimationFrame(tick)}tick()})`);
 const state=()=>evaluate('window.__sceneViewer.getState()');
 const changed=async expression=>{const n=(await state()).renderCount;await evaluate(expression);await wait(`window.__sceneViewer.getState().renderCount>${n}`);};
 const click=id=>changed(`document.getElementById(${JSON.stringify(id)}).click()`);
 const check=(id,value)=>changed(`(()=>{const e=document.getElementById(${JSON.stringify(id)});e.checked=${value};e.dispatchEvent(new Event('change',{bubbles:true}));})()`);
 const classCheck=(id,value)=>changed(`(()=>{const e=document.querySelector('[data-class="${id}"]');e.checked=${value};e.dispatchEvent(new Event('change',{bubbles:true}));})()`);
 const expectedMap=()=>evaluate(`(()=>{const v=window.__sceneViewer,s=v.getState(),m=v.data.bev_map,source=(s.allFrames?m.fused:m.frames[s.frame]).polygon,text=atob(source.runs),bytes=Uint8Array.from(text,c=>c.charCodeAt(0)),dv=new DataView(bytes.buffer);let cells=0,total=0;for(let i=0;i<bytes.length;i+=12){const length=dv.getUint32(i+4,true),id=dv.getUint32(i+8,true);if(s.planeClassIds.includes(id)){total+=length;if(s.classIds.includes(id))cells+=length;}}return{cells,total,declared:source.cells,conflicts:source.conflict_cells||0};})()`);
 const assertMap=async()=>{const s=await state(),e=await expectedMap();assert.equal(s.mapMethod,'polygon');assert.equal(s.mapCells,e.cells,'Visible cells must match selected polygon class runs');assert.equal(s.mapConflictCells,e.conflicts);assert.equal(e.total,e.declared,'Exported cell metadata must match run lengths');assert.equal(s.triangleCount,0);assert.equal(s.surfaceDrawCalls,0);return e;};
 const expectedPoints=()=>evaluate(`(()=>{const v=window.__sceneViewer,s=v.getState();if(!s.points)return 0;let count=0;for(const f of s.allFrames?v.data.frames:[v.data.frames[s.frame]]){const labels=atob(f.points.labels);for(let i=0;i<labels.length;i++){const id=s.availableClassIds.includes(labels.charCodeAt(i))?labels.charCodeAt(i):0;if(s.classIds.includes(id)&&(s.groundPoints||!s.planeClassIds.includes(id)))count++;}}return count;})()`);
 try{
  await send('Runtime.enable');await send('Network.enable');await send('Page.enable');
  await send('Emulation.setDeviceMetricsOverride',{width:1500,height:1000,deviceScaleFactor:1,mobile:false});
  await send('Page.navigate',{url:pathToFileURL(html).href});
  await wait('window.__sceneViewerReady===true || (document.getElementById("error") && !document.getElementById("error").hidden)');
  assert.equal(await evaluate('document.getElementById("error").hidden?"":document.getElementById("error").textContent'),'');
  const frameCount=await evaluate('window.__sceneViewer.data.frames.length');assert.ok(frameCount>=2,'Contract fixture needs at least2 frames');
  assert.equal(await evaluate('document.getElementById("bevMapView").disabled'),false);
  assert.equal(await evaluate('document.querySelectorAll("#ipmLayer, #mapMethod").length'),0,'Removed IPM and map method controls must be absent');
  const initial=await state();assert.equal(initial.mode,'3d');assert.deepEqual(initial.methods,['polygon']);
  await click('topView');const original3d=await state();
  await click('bevMapView');let s=await state();assert.equal(s.mode,'bev-map');assert.equal(s.mapMethod,'polygon');assert.deepEqual(s.methods,original3d.methods);assert.deepEqual(s.camera,original3d.camera);
  assert.ok(s.mapCells>0);assert.ok(s.mapWidth>0&&s.mapHeight>0&&s.resolution>0);await assertMap();
  assert.equal(await evaluate('document.getElementById("viewer").hidden&&!document.getElementById("bevMapCanvas").hidden'),true);
  assert.equal(await evaluate('document.getElementById("bevMapCanvas").getContext("2d").imageSmoothingEnabled'),false);
  assert.equal(s.pointCount,await expectedPoints());
  await changed('(()=>{const e=document.getElementById("timeline");e.value="1";e.dispatchEvent(new Event("input",{bubbles:true}));})()');assert.equal((await state()).frame,1);await assertMap();
  await check('allFrames',true);const fusedPolygon=await assertMap();assert.equal((await state()).playing,false);assert.match(await evaluate('document.getElementById("trailWarning").textContent'),/全帧类别投票融合，点云为多时刻叠加/);
  assert.deepEqual((await state()).methods,['polygon']);assert.equal((await state()).pointCount,await expectedPoints());
  for(const id of initial.planeClassIds)await classCheck(id,false);
  assert.equal((await state()).mapCells,0);await assertMap();
  await classCheck(0,true);assert.equal((await state()).mapCells,0,'Unknown must never become free/road cells');assert.equal((await state()).pointCount,await expectedPoints());
  for(const id of initial.planeClassIds)await classCheck(id,true);await assertMap();
  await check('groundPointLayer',true);assert.equal((await state()).pointCount,await expectedPoints());
  const originalPoints=await evaluate('window.__sceneViewer.data.frames.reduce((n,f)=>n+f.points.count,0)');assert.equal((await state()).pointCount,originalPoints);
  await check('pointLayer',false);assert.equal((await state()).pointCount,0);assert.equal((await state()).pointDrawCalls,0);await assertMap();
  await check('trajectoryLayer',false);await check('mapGrid',false);assert.equal((await state()).mapGrid,false);
  const beforePan=await state(),box=await evaluate('(()=>{const r=document.getElementById("bevMapCanvas").getBoundingClientRect();return{x:r.x+r.width*.45,y:r.y+r.height*.55};})()');
  const pan=async(button,modifiers=0)=>{await send('Input.dispatchMouseEvent',{type:'mousePressed',x:box.x,y:box.y,button,clickCount:1,modifiers});await send('Input.dispatchMouseEvent',{type:'mouseMoved',x:box.x+30,y:box.y+20,button,buttons:button==='right'?2:1,modifiers});await send('Input.dispatchMouseEvent',{type:'mouseReleased',x:box.x+30,y:box.y+20,button,clickCount:1,modifiers});};
  await pan('left');await wait(`window.__sceneViewer.getState().renderCount>${beforePan.renderCount}`);assert.notDeepEqual((await state()).mapCenter,beforePan.mapCenter);
  const leftPan=await state();await pan('right');await wait(`window.__sceneViewer.getState().renderCount>${leftPan.renderCount}`);assert.notDeepEqual((await state()).mapCenter,leftPan.mapCenter);
  const rightPan=await state();await pan('left',8);await wait(`window.__sceneViewer.getState().renderCount>${rightPan.renderCount}`);assert.notDeepEqual((await state()).mapCenter,rightPan.mapCenter);
  const zoomBefore=await state();await send('Input.dispatchMouseEvent',{type:'mouseWheel',x:box.x,y:box.y,deltaX:0,deltaY:-300});await wait(`window.__sceneViewer.getState().mapZoom>${zoomBefore.mapZoom}`);assert.equal((await state()).mapCells,zoomBefore.mapCells);
  await check('mapGrid',true);assert.equal((await state()).mapGrid,true);await click('fitView');
  // Verify an actual selected run cell in the raster, with points/trajectory/grid removed.
  await check('mapGrid',false);
  const probe=await evaluate(`(()=>{const v=window.__sceneViewer,s=v.getState(),g=v.data.bev_map.grid,source=v.data.bev_map.fused.polygon,t=atob(source.runs),b=Uint8Array.from(t,c=>c.charCodeAt(0)),d=new DataView(b.buffer);let best=null;for(let i=0;i<b.length;i+=12){const start=d.getUint32(i,true),length=d.getUint32(i+4,true),id=d.getUint32(i+8,true);if(!s.classIds.includes(id)||!length)continue;const cell=start+Math.floor(length/2),col=cell%g.width,row=Math.floor(cell/g.width),score=Math.abs(col-g.width/2)+Math.abs(row-g.height/2);if(!best||score<best.score)best={col,row,id,score};}const r=document.getElementById('bevMapCanvas').getBoundingClientRect();return{...best,x:g.x_min+(best.col+.5)*g.resolution,y:g.y_min+(g.height-best.row-.5)*g.resolution,rect:{x:r.x,y:r.y,width:r.width,height:r.height}};})()`);
  // Center the known cell using native pan, then zoom around the map's projection center.
  s=await state();const dx=(s.mapCenter[0]-probe.x)*s.mapZoom,dy=(probe.y-s.mapCenter[1])*s.mapZoom;
  const center={x:probe.rect.x+probe.rect.width/2,y:probe.rect.y+(probe.rect.height-50)/2};
  const probeRender=(await state()).renderCount;
  await send('Input.dispatchMouseEvent',{type:'mousePressed',x:center.x,y:center.y,button:'left',clickCount:1});
  await send('Input.dispatchMouseEvent',{type:'mouseMoved',x:center.x+dx,y:center.y+dy,button:'left',buttons:1});
  await send('Input.dispatchMouseEvent',{type:'mouseReleased',x:center.x+dx,y:center.y+dy,button:'left',clickCount:1});
  await wait(`window.__sceneViewer.getState().renderCount>${probeRender}`);
  s=await state();if(s.mapPixelsPerCell<8){await changed(`document.getElementById('bevMapCanvas').dispatchEvent(new WheelEvent('wheel',{clientX:${center.x},clientY:${center.y},deltaY:${-Math.log(8/s.mapPixelsPerCell)*1000},cancelable:true}))`);}
  const pixel=await evaluate(`(()=>{const c=document.getElementById('bevMapCanvas'),a=c.getContext('2d').getImageData(Math.floor(c.width/2),Math.floor((c.height-50)/2),1,1).data;return Array.from(a);})()`);
  const color=await evaluate(`(window.__sceneViewer.data.classes||[]).find(c=>Number(c.id)===${probe.id})?.color || [[145,151,159],[246,169,59],[80,175,225]][${probe.id}]`);assert.deepEqual(pixel.slice(0,3),color,'Cell center must use the exact semantic color');
  await click('fitView');await check('mapGrid',true);const pixels=await evaluate('(()=>{const c=document.getElementById("bevMapCanvas"),a=c.getContext("2d").getImageData(0,0,c.width,c.height).data;let n=0;for(let i=0;i<a.length;i+=4)if(Math.max(a[i],a[i+1],a[i+2])-Math.min(a[i],a[i+1],a[i+2])>45)n++;return n;})()');assert.ok(pixels>100,'Map must contain colored semantic cells');
  await send('Browser.setDownloadBehavior',{behavior:'deny'});await click('exportMap');assert.equal(await evaluate('document.getElementById("bevMapCanvas").toDataURL("image/png").startsWith("data:image/png;base64,")'),true);
  const image=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.mkdirSync(path.dirname(screenshot),{recursive:true});fs.writeFileSync(screenshot,Buffer.from(image.data,'base64'));
  await click('scene3dView');s=await state();assert.equal(s.mode,'3d');assert.deepEqual(s.camera,original3d.camera);assert.deepEqual(s.methods,original3d.methods);assert.ok(s.triangleCount>0);assert.equal(await evaluate('document.getElementById("bevMapCanvas").hidden&&!document.getElementById("viewer").hidden'),true);
  await check('pointLayer',true);await check('allFrames',false);await check('groundPointLayer',false);await classCheck(0,false);
  await click('bevMapView');await check('allFrames',false);await evaluate('document.getElementById("playPause").click()');const startFrame=(await state()).frame;await wait(`window.__sceneViewer.getState().frame!==${startFrame}`);await changed('document.getElementById("playPause").click()');assert.equal((await state()).playing,false);await assertMap();
  assert.deepEqual(requests,[],'Offline viewer must make no HTTP(S) requests');assert.deepEqual(errors,[],'No page/console errors');
  console.log(JSON.stringify({ok:true,scene:await evaluate('window.__sceneViewer.data.scene'),frames:frameCount,fusedPolygon,originalPoints,coloredPixels:pixels,screenshot,finalState:await state(),checks:'polygon map runs/counts, fused polygon map, removed controls absent, class masks, unknown, points, ground points, timeline/play, pan left/right/shift, zoom, grid, exact cell color, PNG, 3D camera restoration, no remote requests/errors'},null,2));
 }finally{await send('Page.close').catch(()=>{});ws.close();for(const p of pending.values())clearTimeout(p.timer);}
}
main().catch(error=>{console.error(error.stack||String(error));process.exitCode=1;});

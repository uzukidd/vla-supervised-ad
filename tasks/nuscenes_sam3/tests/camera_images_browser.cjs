'use strict';
// node tasks/nuscenes_sam3/tests/camera_images_browser.cjs [CDP endpoint]
const fs=require('node:fs'),os=require('node:os'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
async function main(){
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'scene-cameras-'));
 const names=['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT'];
 const frames=[0,1].map(index=>({sample_token:'frame-'+index,frame_name:'frame-'+index,time_s:index*.5,ego:[index,0,0],points:{xyz:'',labels:'',count:0},surfaces:{},camera_images:names.map((camera,i)=>{
  const name=`${index}-${camera}.svg`;
  fs.writeFileSync(path.join(root,name),`<svg xmlns="http://www.w3.org/2000/svg" width="320" height="180"><rect width="320" height="180" fill="${index?'navy':'teal'}"/><text x="10" y="90" fill="white">${camera} ${index}</text></svg>`);
  const overlay=`${index}-${camera}-overlay.svg`;
  fs.writeFileSync(path.join(root,overlay),`<svg xmlns="http://www.w3.org/2000/svg" width="320" height="180"><rect width="320" height="180" fill="purple"/><text x="10" y="90" fill="white">SAM3 ${camera} ${index}</text></svg>`);
  const points=`${index}-${camera}-points.svg`,overlayPoints=`${index}-${camera}-overlay_points.svg`;
  fs.writeFileSync(path.join(root,points),`<svg xmlns="http://www.w3.org/2000/svg" width="320" height="180"><rect width="320" height="180" fill="darkgreen"/><text x="10" y="90" fill="white">LiDAR ${camera} ${index}</text></svg>`);
  fs.writeFileSync(path.join(root,overlayPoints),`<svg xmlns="http://www.w3.org/2000/svg" width="320" height="180"><rect width="320" height="180" fill="darkred"/><text x="10" y="90" fill="white">SAM3 + LiDAR ${camera} ${index}</text></svg>`);
  return {camera,src:name,overlay_src:overlay,points_src:points,overlay_points_src:overlayPoints,width:320,height:180,timestamp_us:index*500000+i};
 })}));
 const data={scene:'camera-test',frames,bounds:{min:[-5,-5,-1],max:[5,5,2]}};
 const template=fs.readFileSync(path.join(__dirname,'../web/scene_viewer.html'),'utf8');
 const file=path.join(root,'viewer.html');fs.writeFileSync(file,template.replace('__SCENE_DATA__',JSON.stringify(data)));
 const endpoint=process.argv[2]||'http://127.0.0.1:9341';
 const target=await fetch(endpoint+'/json/new?about:blank',{method:'PUT'}).then(r=>r.json());
 const ws=new WebSocket(target.webSocketDebuggerUrl),pending=new Map(),errors=[];let seq=0;
 await new Promise((resolve,reject)=>{ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true});});
 ws.addEventListener('message',event=>{const m=JSON.parse(event.data);if(m.id){const p=pending.get(m.id);if(!p)return;pending.delete(m.id);clearTimeout(p.timer);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);});
 const send=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq,timer=setTimeout(()=>reject(Error('Timeout: '+method)),15000);pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}));});
 const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
 const wait=expression=>evaluate(`new Promise((resolve,reject)=>{const end=performance.now()+10000;function tick(){if(${expression})resolve(true);else if(performance.now()>end)reject(Error('Wait timed out'));else setTimeout(tick,20);}tick();})`);
 const imageSource=(index,camera,semantics=false,points=false)=>`${index}-${camera}${points?(semantics?'-overlay_points':'-points'):(semantics?'-overlay':'')}.svg`;
 const assertFrame=async (index,semantics=false,points=false)=>{
  await wait(`document.getElementById('cameraFrame').textContent==='${index+1} / 2' && [...document.querySelectorAll('.camera-tile img')].every(i=>i.complete&&i.naturalWidth>0)`);
  assert.deepEqual(await evaluate(`[...document.querySelectorAll('.camera-tile')].map(t=>({camera:t.dataset.camera,src:t.querySelector('img').getAttribute('src'),disabled:t.disabled}))`),['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT'].map(camera=>({camera,src:imageSource(index,camera,semantics,points),disabled:false})));
 };
 const assertToggles=async (semantics,points,semanticsDisabled=false,pointsDisabled=false)=>{
  assert.deepEqual(await evaluate(`['cameraSemantics','cameraDialogSemantics','cameraPoints','cameraDialogPoints'].map(id=>{const input=document.getElementById(id);return {id,checked:input.checked,disabled:input.disabled};})`),[
   {id:'cameraSemantics',checked:semantics,disabled:semanticsDisabled},{id:'cameraDialogSemantics',checked:semantics,disabled:semanticsDisabled},
   {id:'cameraPoints',checked:points,disabled:pointsDisabled},{id:'cameraDialogPoints',checked:points,disabled:pointsDisabled}
  ]);
 };
 const assertEnlarged=async (index,camera,semantics=false,points=false)=>{
  const src=imageSource(index,camera,semantics,points);
  await wait(`document.getElementById('cameraDialog').open && document.getElementById('cameraFullImage').getAttribute('src')===${JSON.stringify(src)} && document.getElementById('cameraFullImage').complete && document.getElementById('cameraFullImage').naturalWidth>0`);
  assert.ok(await evaluate(`document.getElementById('cameraDialogTitle').textContent.includes(${JSON.stringify(camera)}) && document.getElementById('cameraDialogTitle').textContent.endsWith('${index+1} / 2') && document.getElementById('cameraImageInfo').textContent.startsWith('frame-${index} ·')`),'Enlarged camera title and metadata must follow the selected frame');
 };
 const setMode=async (semantics,points)=>{
  for(const [id,checked] of [['cameraSemantics',semantics],['cameraPoints',points]]){
   if(await evaluate(`document.getElementById('${id}').checked`)!==checked)await evaluate(`document.getElementById('${id}').click()`);
  }
  await assertToggles(semantics,points);
 };
 try{
  await send('Runtime.enable');await send('Page.enable');await send('Emulation.setDeviceMetricsOverride',{width:1500,height:1000,deviceScaleFactor:1,mobile:false});
  await send('Page.navigate',{url:pathToFileURL(file).href});await wait('window.__sceneViewerReady===true');await assertFrame(0);
  await evaluate(`(()=>{const t=document.getElementById('timeline');t.value='1';t.dispatchEvent(new Event('input'));})()`);await assertFrame(1);
  for(const [semantics,points] of [[false,false],[true,false],[false,true],[true,true]]){
   await setMode(semantics,points);await assertFrame(1,semantics,points);
   for(const name of names){
    await evaluate(`document.querySelector('[data-camera="${name}"]').click()`);
    assert.equal(await evaluate('document.getElementById("cameraDialog").open'),true);
    await assertEnlarged(1,name,semantics,points);await assertToggles(semantics,points);
    await evaluate('document.getElementById("closeCamera").click()');
   }
  }
  await setMode(false,false);await assertFrame(1);
  assert.equal(await evaluate('document.getElementById("cameraSemantics").checked'),false);
  await evaluate('document.getElementById("cameraSemantics").click()');await assertFrame(1,true);
  await evaluate('window.__sceneViewer.setFrame(0)');await assertFrame(0,true);
  await evaluate('document.querySelector(".camera-tile").click()');
  assert.equal(await evaluate('document.getElementById("cameraFullImage").getAttribute("src")'),'0-CAM_FRONT_LEFT-overlay.svg');
  assert.equal(await evaluate('document.getElementById("cameraDialogSemantics").checked'),true);
  await evaluate('document.getElementById("cameraDialogSemantics").click()');await assertFrame(0);
  assert.equal(await evaluate('document.getElementById("cameraFullImage").getAttribute("src")'),'0-CAM_FRONT_LEFT.svg');
  assert.equal(await evaluate('document.getElementById("cameraSemantics").checked'),false);
  await evaluate('document.getElementById("cameraDialogSemantics").click()');await assertFrame(0,true);
  await evaluate('window.__sceneViewer.setFrame(1)');await assertFrame(1,true);
  assert.equal(await evaluate('document.getElementById("cameraFullImage").getAttribute("src")'),'1-CAM_FRONT_LEFT-overlay.svg');
  await evaluate('document.getElementById("closeCamera").click()');
  await evaluate('document.getElementById("cameraSemantics").click()');await assertFrame(1);
  await evaluate('document.querySelector(".camera-tile").click()');
  for(const [id,semantics,points] of [
   ['cameraPoints',false,true],['cameraDialogPoints',false,false],['cameraDialogPoints',false,true],
   ['cameraSemantics',true,true],['cameraDialogSemantics',false,true],['cameraDialogSemantics',true,true],
   ['cameraDialogPoints',true,false],['cameraPoints',true,true]
  ]){
   await evaluate(`document.getElementById('${id}').click()`);
   await assertToggles(semantics,points);await assertFrame(1,semantics,points);await assertEnlarged(1,'CAM_FRONT_LEFT',semantics,points);
  }
  await evaluate('window.__sceneViewer.setFrame(0)');await assertFrame(0,true,true);await assertEnlarged(0,'CAM_FRONT_LEFT',true,true);await assertToggles(true,true);
  await evaluate(`(()=>{const t=document.getElementById('timeline');t.value='1';t.dispatchEvent(new Event('input'));})()`);await assertFrame(1,true,true);await assertEnlarged(1,'CAM_FRONT_LEFT',true,true);
  await evaluate('document.getElementById("prevFrame").click()');await assertFrame(0,true,true);await assertEnlarged(0,'CAM_FRONT_LEFT',true,true);
  await evaluate('document.getElementById("nextFrame").click()');await assertFrame(1,true,true);await assertEnlarged(1,'CAM_FRONT_LEFT',true,true);
  await evaluate('document.getElementById("closeCamera").click()');await setMode(false,false);await assertFrame(1);
  await evaluate('document.getElementById("playPause").click()');
  await evaluate('document.querySelector(".camera-tile").click()');
  assert.equal(await evaluate('window.__sceneViewer.getState().playing'),false);
  await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});
  assert.equal(await evaluate('document.getElementById("cameraDialog").open'),false);
  await evaluate('window.__sceneViewer.setFrame(0)');await assertFrame(0);
  await evaluate('document.getElementById("nextFrame").click()');await assertFrame(1);
  await evaluate('document.getElementById("prevFrame").click()');await assertFrame(0);
  await evaluate('window.__sceneViewer.setAllFrames(true)');await assertFrame(0);
  await evaluate('document.getElementById("cameraSemantics").click()');await assertFrame(0,true);
  await evaluate('window.__sceneViewer.data.frames[1].camera_images.forEach(image=>delete image.overlay_src);window.__sceneViewer.setFrame(1)');await assertFrame(1);
  await assertToggles(true,false,true,false);
  await evaluate('document.querySelector(".camera-tile").click()');await assertEnlarged(1,'CAM_FRONT_LEFT');
  await evaluate('document.getElementById("cameraPoints").click()');await assertFrame(1,false,true);await assertEnlarged(1,'CAM_FRONT_LEFT',false,true);await assertToggles(true,true,true,false);
  await evaluate('window.__sceneViewer.setFrame(0)');await assertFrame(0,true,true);await assertEnlarged(0,'CAM_FRONT_LEFT',true,true);await assertToggles(true,true);
  await evaluate('document.getElementById("cameraDialogPoints").click()');await assertFrame(0,true);await assertEnlarged(0,'CAM_FRONT_LEFT',true);await assertToggles(true,false);
  await evaluate('document.getElementById("cameraDialogPoints").click()');await assertFrame(0,true,true);
  await evaluate(`window.__sceneViewer.data.frames[1].camera_images.forEach(image=>{image.overlay_src='1-'+image.camera+'-overlay.svg';delete image.points_src;delete image.overlay_points_src;});window.__sceneViewer.setFrame(1)`);
  await assertFrame(1,true);await assertEnlarged(1,'CAM_FRONT_LEFT',true);await assertToggles(true,true,false,true);
  await evaluate('document.getElementById("cameraPoints").click();document.getElementById("cameraDialogPoints").click()');await assertFrame(1,true);await assertToggles(true,true,false,true);
  await evaluate('document.getElementById("cameraDialogSemantics").click()');await assertFrame(1);await assertEnlarged(1,'CAM_FRONT_LEFT');await assertToggles(false,true,false,true);
  await evaluate('document.getElementById("cameraSemantics").click()');await assertFrame(1,true);await assertEnlarged(1,'CAM_FRONT_LEFT',true);await assertToggles(true,true,false,true);
  await evaluate('window.__sceneViewer.setFrame(0)');await assertFrame(0,true,true);await assertEnlarged(0,'CAM_FRONT_LEFT',true,true);await assertToggles(true,true);
  await evaluate('document.getElementById("closeCamera").click()');await setMode(true,false);await assertFrame(0,true);
  const layout=await evaluate(`(()=>{const panel=document.getElementById('cameraPanel').getBoundingClientRect(),legend=document.querySelector('.legend').getBoundingClientRect();return{panelBottom:panel.bottom,legendTop:legend.top};})()`);
  assert.ok(layout.panelBottom<layout.legendTop,'Camera panel must not overlap statistics at desktop size');
  assert.deepEqual(errors,[]);
  const screenshot=path.join(root,'cameras.png'),image=await send('Page.captureScreenshot',{format:'png'});fs.writeFileSync(screenshot,Buffer.from(image.data,'base64'));
  console.log(JSON.stringify({ok:true,checks:'six linked images in original/overlay/points/overlay_points modes, all six enlargements in all four modes, independent shared semantic/projected-point toggles both directions, timeline and previous/next synchronization including open enlargement, both choices retained across frames, missing-point controls disabled with overlay/original fallback, original missing-overlay fallback plus points-only fallback, playback pauses, Escape close, all-frame selection, panel layout, no runtime errors',screenshot},null,2));
 }finally{await send('Page.close').catch(()=>{});ws.close();for(const p of pending.values())clearTimeout(p.timer);}
}
main().catch(error=>{console.error(error.stack);process.exitCode=1;});

'use strict';
// node tasks/nuscenes_sam3/tests/class_persistence_browser.cjs [CDP endpoint]
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),http=require('node:http');
async function main(){
 let classes=[{name:'road',display_name:'道路',color:'#d29b14',prompts:'road',plane:true},{name:'vehicle',display_name:'车辆',color:'#c8372d',prompts:'car | vehicle',plane:false},{name:'person',display_name:'行人',color:'#236ee1',prompts:'person',plane:false}],saves=0,legacyMode=false,rejectToken=false;
 const template=fs.readFileSync(path.join(__dirname,'../web/generator.html'),'utf8');
 const server=http.createServer(async(req,res)=>{
  res.setHeader('Content-Type',req.url==='/'?'text/html; charset=utf-8':'application/json');
  if(req.url==='/')return res.end(template);
  if(req.url==='/api/config')return res.end(JSON.stringify({token:'test-token',classes,classes_path:'/local/jobs/semantic_classes.json',scenes:[],defaults:{},runs:[]}));
  if(req.url==='/api/classes'){
   if(legacyMode){res.statusCode=404;return res.end(JSON.stringify({error:'未找到接口'}));}
   if(rejectToken){rejectToken=false;res.statusCode=403;return res.end(JSON.stringify({error:'请求令牌无效'}));}
   assert.equal(req.headers['x-generator-token'],'test-token');let text='';for await(const chunk of req)text+=chunk;
   classes=JSON.parse(text).classes;saves++;return res.end(JSON.stringify({classes,classes_path:'/local/jobs/semantic_classes.json'}));
  }
  res.end('{}');
 });
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
 const endpoint=process.argv[2]||'http://127.0.0.1:9341',url=`http://127.0.0.1:${server.address().port}/`;
 let ws;const pending=new Map();
 try{
  const target=await fetch(endpoint+'/json/new?about:blank',{method:'PUT'}).then(r=>r.json());
  ws=new WebSocket(target.webSocketDebuggerUrl);let seq=0;const errors=[];
  await new Promise((resolve,reject)=>{ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true});});
  ws.addEventListener('message',event=>{const m=JSON.parse(event.data);if(m.id){const p=pending.get(m.id);if(!p)return;pending.delete(m.id);clearTimeout(p.timer);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);});
  const send=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq,timer=setTimeout(()=>reject(Error('Timeout '+method)),15000);pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}));});
  const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
  const wait=expression=>evaluate(`new Promise((resolve,reject)=>{const end=performance.now()+10000;function tick(){if(${expression})resolve(true);else if(performance.now()>end)reject(Error('Wait timeout'));else setTimeout(tick,40);}tick();})`);
  const saved=()=>wait(`document.getElementById('classesSaveStatus').textContent.includes('Saved locally')`);
  await send('Runtime.enable');await send('Page.enable');await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await send('Page.navigate',{url});await wait('window.__generatorUI?.getState().ready');
  await evaluate(`(()=>{const input=document.querySelector('[data-field="prompts"]');input.value='road | street';input.dispatchEvent(new Event('input',{bubbles:true}));})()`);await saved();
  assert.equal(classes[0].prompts,'road | street');
  await evaluate(`(()=>{const input=document.querySelector('[data-field="display_name"]');input.value='城市道路 / Urban road';input.dispatchEvent(new Event('input',{bubbles:true}));})()`);await saved();
  assert.equal(classes[0].display_name,'城市道路 / Urban road');assert.equal(classes[0].name,'road');
  await evaluate(`(()=>{const input=document.querySelector('[data-field="color"]');input.value='#e62991';input.dispatchEvent(new Event('input',{bubbles:true}));})()`);await saved();
  assert.equal(classes[0].color,'#e62991');
  const params=await evaluate(`(()=>{for(const [id,value] of Object.entries({bevResolution:'.25',bevRoiSize:'80',maxEdge:'3',mapResolution:'.5',occupancyResolution:'.25'})){const input=document.getElementById(id);input.value=value;input.dispatchEvent(new Event('input',{bubbles:true}));}return window.__generatorUI.getConfig();})()`);
  assert.equal(params.bev_resolution,.25);assert.equal(params.bev_roi_size,80);assert.equal(params.max_edge,3);assert.equal(params.map_resolution,.5);assert.equal(params.occupancy_resolution,.25);
  assert.equal(params.compute_grid_projection,false);
  assert.equal(await evaluate('document.getElementById("gridProjectionSamples").disabled'),true);
  await evaluate('document.getElementById("computeGridProjection").click()');
  assert.equal(await evaluate('document.getElementById("gridProjectionParams").hidden'),false);
  const projection=await evaluate(`(()=>{for(const [id,value] of Object.entries({gridProjectionSamples:'5',gridProjectionMaxDistance:'60'})){const input=document.getElementById(id);input.value=value;input.dispatchEvent(new Event('input',{bubbles:true}));}return window.__generatorUI.getConfig();})()`);
  assert.equal(projection.compute_grid_projection,true);assert.equal(projection.grid_projection_samples,5);assert.equal(projection.grid_projection_max_distance,60);
  await evaluate('document.getElementById("computeGridProjection").click()');
  assert.equal(await evaluate('document.getElementById("gridProjectionSamples").disabled'),true);
  assert.ok(!('plane_z' in params));assert.ok(!('max_distance' in params));
  await evaluate(`(()=>{const list=document.getElementById('classRows'),row=list.firstElementChild,handle=row.querySelector('.drag-handle'),target=list.lastElementChild,transfer=new DataTransfer();handle.dispatchEvent(new DragEvent('dragstart',{bubbles:true,dataTransfer:transfer}));target.dispatchEvent(new DragEvent('dragover',{bubbles:true,dataTransfer:transfer,clientY:target.getBoundingClientRect().bottom-1}));list.dispatchEvent(new DragEvent('drop',{bubbles:true,dataTransfer:transfer}));})()`);await saved();
  assert.deepEqual(classes.map(c=>c.name),['vehicle','person','road']);
  assert.equal(classes[2].plane,true);assert.equal(classes[2].prompts,'road | street');assert.equal(classes[2].display_name,'城市道路 / Urban road');assert.equal(classes[2].color,'#e62991');
  assert.deepEqual(await evaluate('window.__generatorUI.getConfig().classes'),classes);
  await send('Page.reload');await wait('window.__generatorUI?.getState().ready');
  assert.deepEqual(await evaluate('window.__generatorUI.getConfig().classes'),classes);
  await evaluate(`document.querySelector('.drag-handle').dispatchEvent(new KeyboardEvent('keydown',{bubbles:true,altKey:true,code:'ArrowDown'}))`);await saved();
  assert.deepEqual(classes.map(c=>c.name),['person','vehicle','road']);
  const count=saves;await evaluate('document.getElementById("saveClasses").click()');await saved();assert.ok(saves>count);
  legacyMode=true;
  await evaluate(`(()=>{const input=document.querySelector('[data-field="display_name"]');input.value='未保存的显示名';input.dispatchEvent(new Event('input',{bubbles:true}));})()`);
  await wait(`document.getElementById('classesSaveStatus').textContent.includes('Restart generator')`);
  await send('Page.reload');await wait('window.__generatorUI?.getState().ready');
  assert.equal(await evaluate('window.__generatorUI.getConfig().classes[0].display_name'),'未保存的显示名');
  legacyMode=false;rejectToken=true;
  await evaluate('document.getElementById("saveClasses").click()');await saved();
  assert.equal(classes[0].display_name,'未保存的显示名');
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({ok:true,checks:'class colors saved/reloaded/reordered, plane resolution/ROI/max-edge/map controls, removed IPM parameters, independent Chinese display names, autosave prompt, drag reorder, labels/prompts/planes preserved, reload recovery, keyboard reorder, explicit save, old-backend 404 diagnosis, unsaved draft recovery, token refresh after restart, no browser errors',saves,classes}));
  if(process.argv[3]){
   await send('Page.navigate',{url:process.argv[3]});await wait('window.__generatorUI?.getState().ready');
   const actual=await evaluate(`fetch('/api/config').then(r=>r.json()).then(data=>({supported:data.capabilities?.class_display_names,classes:data.classes}))`);
   assert.equal(actual.supported,true);
   assert.deepEqual(await evaluate('window.__generatorUI.getConfig().classes'),actual.classes);
   assert.equal(await evaluate('document.querySelectorAll("[data-field=display_name]").length'),actual.classes.length);
   const layout=await evaluate(`(()=>{const row=document.querySelector('.class-row'),last=row.lastElementChild.getBoundingClientRect(),panel=row.closest('.panel').getBoundingClientRect();return{lastRight:last.right,panelRight:panel.right};})()`);
   assert.ok(layout.lastRight<=layout.panelRight,'Class editor columns must fit the panel');
   console.log(JSON.stringify({actualGenerator:process.argv[3],displayNames:actual.classes.map(c=>c.display_name),layout}));
  }
  await send('Page.close');
 }finally{if(ws)ws.close();for(const p of pending.values())clearTimeout(p.timer);await new Promise(resolve=>server.close(resolve));}
}
main().catch(error=>{console.error(error.stack);process.exitCode=1;});
